"""
DiT-FM: Diffusion Transformer for Flow Matching

This module extends the existing DiT model to support Flow Matching (FM) paradigm.
It reuses most components from dit.py while adding continuous time embedding 
and velocity prediction capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Optional, Any, Tuple
from einops import rearrange

# Reuse components from existing DiT
from .dit import (
    DiTBlock, DiTDoubleStreamBlock, DiTSingleParallelBlock, GraspTokenizer, PositionalEmbedding, TimestepEmbedding,
    AdaptiveLayerNorm, AdaLNZero, OutputProjection, DiTModel,
    DiTValidationError, DiTInputError, DiTDimensionError,
    DiTDeviceError, DiTConditioningError, DiTGracefulFallback
)
from .dit_utils import adjust_backbone_config, init_weights
from .dit_conditioning import prepare_scene_features, pool_scene_features
from models.backbone import build_backbone
from models.utils.text_encoder import TextConditionProcessor, PosNegTextEncoder
from .dit_conditioning import ensure_text_encoder, prepare_text_features
from .scene_pool import GlobalScenePool
from .local_selector import build_local_selector
from .adaln_cond import build_adaln_cond_vector


class ContinuousTimeEmbedding(nn.Module):
    """
    Continuous time embedding for t ∈ [0, 1] using Gaussian random Fourier features.
    
    This is more suitable for Flow Matching compared to the discrete timestep 
    embeddings used in DDPM.
    """
    def __init__(self, dim: int, freq_dim: int = 256, max_period: float = 10.0):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        
        # Random Fourier features
        self.register_buffer('frequencies', 
                           torch.randn(freq_dim) * math.pi * 2.0)
        
        # MLP to process concatenated features
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim * 2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Continuous time in [0, 1], shape [B] or [B, 1]
        Returns:
            Time embeddings of shape [B, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
            
        # Ensure t is in [0, 1]
        t = torch.clamp(t, 0.0, 1.0)
        
        # Compute Fourier features
        t_expanded = t.unsqueeze(1)  # [B, 1]
        freqs = self.frequencies.unsqueeze(0)  # [1, freq_dim]
        
        # [B, freq_dim]
        fourier_features = t_expanded * freqs
        
        # Concatenate sin, cos, and raw t
        features = torch.cat([
            torch.sin(fourier_features),
            torch.cos(fourier_features),
            t_expanded
        ], dim=-1)  # [B, freq_dim*2 + 1]
        
        # Pass through MLP
        emb = self.mlp(features)
        
        logging.debug(f"[ContinuousTimeEmbedding] t: {t.shape}, emb: {emb.shape}, "
                     f"||emb||_2: {torch.norm(emb, dim=-1).mean().item():.4f}")
        
        return emb


class DiTFM(nn.Module):
    """
    Diffusion Transformer for Flow Matching.
    
    This model extends DiT to support:
    1. Continuous time embedding for t ∈ [0, 1]
    2. Velocity field prediction for Flow Matching
    3. Compatibility with existing DDPM mode
    """
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        # Prediction mode: velocity (FM), epsilon (DDPM), or pose (direct)
        self.pred_mode = cfg.get('pred_mode', 'velocity')
        
        # Input Dimension Config (reuse from DiT)
        rot_to_dim = {'quat': 23, 'r6d': 25}
        self._rot_type = cfg.rot_type
        self.d_x = rot_to_dim[cfg.rot_type]
        
        # Model Architecture Config
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.d_head = cfg.d_head
        self.dropout = cfg.dropout
        self.max_sequence_length = cfg.max_sequence_length
        self.use_learnable_pos_embedding = cfg.use_learnable_pos_embedding
        
        # Time embedding config
        self.time_embed_dim = cfg.time_embed_dim
        self.use_adaptive_norm = cfg.use_adaptive_norm
        
        # AdaLN-Zero 多条件融合配置
        self.use_adaln_zero = getattr(cfg, 'use_adaln_zero', False)
        self.use_scene_pooling = getattr(cfg, 'use_scene_pooling', True)
        # AdaLN 模式：'multi' 使用多条件(time+scene+text)，'simple' 仅使用 time_emb
        self.adaln_mode = getattr(cfg, 'adaln_mode', 'multi')
        self.use_double_stream = getattr(cfg, 'use_double_stream', False)
        self.num_double_blocks = int(getattr(cfg, 'num_double_blocks', 0))
        self.single_block_variant = getattr(cfg, 'single_block_variant', 'legacy')
        
        # FM-specific config
        self.continuous_time = cfg.get('continuous_time', True)
        self.freq_dim = cfg.get('freq_dim', 256)
        
        # Conditioning config
        self.use_text_condition = cfg.use_text_condition
        self.text_dropout_prob = cfg.text_dropout_prob
        self.use_negative_prompts = getattr(cfg, 'use_negative_prompts', True)
        self.use_object_mask = cfg.use_object_mask
        self.use_rgb = cfg.use_rgb
        
        # Text token 级别特征配置
        self.use_text_tokens = getattr(cfg, 'use_text_tokens', False)
        self.scene_to_time: Optional[nn.Module] = None
        self.text_to_time: Optional[nn.Module] = None
        
        # Debug config
        self.debug_check_nan = cfg.get('debug', {}).get('check_nan', True)
        self.debug_log_stats = cfg.get('debug', {}).get('log_tensor_stats', False)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self._warned_missing_scene_cond = False
        self._warned_missing_text_cond = False
        
        # Attention optimization config (pass-through to DiT blocks)
        self.use_flash_attention = getattr(cfg, 'use_flash_attention', False)
        self.attention_chunk_size = getattr(cfg, 'attention_chunk_size', 512)
        
        # Attention dropout config (for regularization)
        self.attention_dropout = getattr(cfg, 'attention_dropout', 0.0)
        self.cross_attention_dropout = getattr(cfg, 'cross_attention_dropout', 0.0)
        # MLP ratio for fused single-stream variants
        self.mlp_ratio = getattr(cfg, 'mlp_ratio', 4.0)
        
        # Geometric attention bias config
        self.use_geometric_bias = getattr(cfg, 'use_geometric_bias', False)
        self.geometric_bias_hidden_dims = getattr(cfg, 'geometric_bias_hidden_dims', [128, 64])
        self.geometric_bias_feature_types = getattr(cfg, 'geometric_bias_feature_types', ['relative_pos', 'distance'])
        
        # Time-aware conditioning config
        self.use_t_aware_conditioning = getattr(cfg, 'use_t_aware_conditioning', False)
        self.t_gate_config = getattr(cfg, 't_gate', None)

        # Core components
        self.grasp_tokenizer = GraspTokenizer(self.d_x, self.d_model)
        
        if self.use_learnable_pos_embedding:
            self.pos_embedding = PositionalEmbedding(self.d_model, self.max_sequence_length)
        else:
            self.pos_embedding = None
            
        # Time embeddings - support both continuous and discrete
        if self.pred_mode == 'velocity' and self.continuous_time:
            self.time_embedding = ContinuousTimeEmbedding(
                self.d_model, self.freq_dim
            )
            self.logger.info("Using continuous time embedding for Flow Matching")
        else:
            self.time_embedding = TimestepEmbedding(
                self.d_model, self.time_embed_dim
            )
            self.logger.info("Using discrete timestep embedding")
        
        # Additional time projection for adaptive norm / AdaLN-Zero
        self.time_proj = None
        if self.use_adaptive_norm or self.use_adaln_zero:
            self.time_proj = nn.Sequential(
                nn.Linear(self.d_model, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim)
            )

        # 当未启用 adaptive_norm 但启用时间门控且门控为 MLP 时，仍需为门控提供 time_emb
        self.time_proj_gate = None
        if (not self.use_adaptive_norm) and self.use_t_aware_conditioning and self.t_gate_config is not None:
            try:
                gate_type = self.t_gate_config.get('type', 'cos2')
            except Exception:
                gate_type = 'cos2'
            if gate_type == 'mlp':
                self.time_proj_gate = nn.Sequential(
                    nn.Linear(self.d_model, self.time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.time_embed_dim, self.time_embed_dim)
                )
        
        # 计算 AdaLN-Zero 的条件向量维度
        cond_dim: Optional[int] = None
        if self.use_adaln_zero:
            cond_dim = self.time_embed_dim
            if self.adaln_mode == 'multi':
                if self.use_scene_pooling:
                    self.scene_to_time = nn.Linear(self.d_model, self.time_embed_dim)
                    nn.init.zeros_(self.scene_to_time.weight)
                    nn.init.zeros_(self.scene_to_time.bias)
                if self.use_text_condition:
                    self.text_to_time = nn.Linear(self.d_model, self.time_embed_dim)
                    nn.init.zeros_(self.text_to_time.weight)
                    nn.init.zeros_(self.text_to_time.bias)
                self.logger.info(
                    "AdaLN-Zero enabled for DiT-FM (mode=multi) with projected cond_dim=%d (time=%d + projected scene=%s + "
                    "projected text=%s)",
                    cond_dim,
                    self.time_embed_dim,
                    "yes" if self.scene_to_time is not None else "no",
                    "yes" if self.text_to_time is not None else "no",
                )
                self.logger.info(
                    "AdaLN-Zero config (multi): use_scene_pooling=%s, use_text_condition=%s",
                    bool(self.use_scene_pooling),
                    bool(self.use_text_condition),
                )
            else:
                self.logger.info(
                    f"AdaLN-Zero enabled for DiT-FM (mode=simple) with cond_dim={cond_dim} (time={self.time_embed_dim})"
                )
            self.logger.info(
                "AdaLN-Zero activation summary: mode=%s, cond_dim=%s, t_gate=%s",
                self.adaln_mode,
                cond_dim,
                type(self.t_gate_config).__name__ if self.t_gate_config is not None else "None",
            )
        if not self.use_double_stream:
            self.num_double_blocks = 0
        else:
            if not self.use_adaln_zero:
                raise ValueError("use_double_stream=True requires use_adaln_zero=True")
            if cond_dim is None:
                raise ValueError("Double-stream blocks require a valid AdaLN cond_dim.")
            if self.num_double_blocks <= 0:
                self.logger.warning(
                    "use_double_stream=True but num_double_blocks<=0, falling back to legacy single-stream blocks."
                )
                self.num_double_blocks = 0
            else:
                if self.num_double_blocks > self.num_layers:
                    self.logger.warning(
                        f"num_double_blocks ({self.num_double_blocks}) exceeds total layers {self.num_layers}; clipping."
                    )
                self.num_double_blocks = min(self.num_double_blocks, self.num_layers)
            self.logger.info(
                "Double-stream config: enabled=True, num_double_blocks=%d, single_block_variant=%s",
                self.num_double_blocks,
                self.single_block_variant,
            )
        # 规范化 single_block_variant，并处理兼容映射
        valid_variants = {"legacy", "fused_parallel", "fused_serial", "parallel"}
        if self.single_block_variant not in valid_variants:
            self.logger.warning(
                f"single_block_variant='{self.single_block_variant}' is unsupported. Falling back to 'legacy'."
            )
            self.single_block_variant = "legacy"
        if self.single_block_variant == "parallel":
            self.logger.warning("single_block_variant='parallel' 已弃用，映射为 'fused_parallel'")
            self.single_block_variant = "fused_parallel"

        # 前置条件：fused_* 仅允许在启用双流且 num_double_blocks>0 时使用
        if self.single_block_variant in ("fused_parallel", "fused_serial"):
            if not self.use_double_stream or self.num_double_blocks <= 0:
                raise ValueError(
                    "single_block_variant='fused_*' 需要 use_double_stream=True 且 num_double_blocks>0"
                )
        
        # Initialize geometric attention bias module if enabled
        if self.use_geometric_bias:
            from .geometric_attention_bias import GeometricAttentionBias
            self.geometric_bias_module = GeometricAttentionBias(
                d_model=self.d_model,
                hidden_dims=self.geometric_bias_hidden_dims,
                feature_types=self.geometric_bias_feature_types,
                num_heads=self.num_heads,
                rot_type=self._rot_type
            )
            self.logger.info(f"Geometric attention bias enabled with features: {self.geometric_bias_feature_types}")
        else:
            self.geometric_bias_module = None
        
        # Initialize time-aware conditioning gate if enabled
        if self.use_t_aware_conditioning:
            from .time_gating import build_time_gate
            self.time_gate = build_time_gate(
                use_t_aware_conditioning=True,
                gate_config=self.t_gate_config,
                time_embed_dim=self.time_embed_dim
            )
            self.logger.info(f"Time-aware conditioning enabled: {self.time_gate.config}")
        else:
            self.time_gate = None
        
        # DiT transformer blocks (reuse from DiT)
        dit_blocks = []
        for layer_idx in range(self.num_layers):
            if self.use_double_stream and layer_idx < self.num_double_blocks:
                block = DiTDoubleStreamBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_head=self.d_head,
                    dropout=self.dropout,
                    cond_dim=cond_dim if self.use_adaln_zero else None,
                    chunk_size=self.attention_chunk_size,
                    use_flash_attention=self.use_flash_attention,
                    attention_dropout=self.attention_dropout,
                )
            else:
                if self.single_block_variant == "fused_parallel":
                    from .dit_single_stream_parallel import ParallelSingleStreamBlock
                    block = ParallelSingleStreamBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        d_head=self.d_head,
                        mlp_ratio=self.mlp_ratio,
                        dropout=self.dropout,
                        cond_dim=cond_dim if self.use_adaln_zero else None,
                        use_flash_attention=self.use_flash_attention,
                    )
                elif self.single_block_variant == "fused_serial":
                    from .dit import DiTSingleSerialNoCrossBlock
                    block = DiTSingleSerialNoCrossBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        d_head=self.d_head,
                        dropout=self.dropout,
                        cond_dim=cond_dim if self.use_adaln_zero else 0,
                        attention_dropout=self.attention_dropout,
                    )
                else:
                    block = DiTBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        d_head=self.d_head,
                        dropout=self.dropout,
                        use_adaptive_norm=self.use_adaptive_norm,
                        time_embed_dim=self.time_embed_dim if self.use_adaptive_norm else None,
                        use_adaln_zero=self.use_adaln_zero,
                        cond_dim=cond_dim if self.use_adaln_zero else None,
                        chunk_size=self.attention_chunk_size,
                        use_flash_attention=self.use_flash_attention,
                        attention_dropout=self.attention_dropout,
                        cross_attention_dropout=self.cross_attention_dropout,
                        use_geometric_bias=self.use_geometric_bias,
                        geometric_bias_module=self.geometric_bias_module,
                        time_gate=self.time_gate
                    )
            dit_blocks.append(block)
        self.dit_blocks = nn.ModuleList(dit_blocks)
        
        # Output projections for different modes
        self.output_projection = OutputProjection(self.d_model, self.d_x)
        
        # Velocity head config
        self.velocity_head_use_layer_norm = getattr(cfg, 'velocity_head_use_layer_norm', False)
        
        # Separate heads for different prediction targets
        if self.pred_mode == 'velocity':
            if self.velocity_head_use_layer_norm:
                self.velocity_head = nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_x)
                )
                self.logger.info("Initialized velocity head with LayerNorm for Flow Matching")
            else:
                self.velocity_head = nn.Linear(self.d_model, self.d_x)
                self.logger.info("Initialized velocity head (Linear) for Flow Matching")
        
        # Conditioning modules (reuse from DiT)
        backbone_cfg = adjust_backbone_config(cfg.backbone, cfg.use_rgb, cfg.use_object_mask)
        self.scene_model = build_backbone(backbone_cfg)
        
        # Scene feature projection to match model dimension
        # Get backbone output dimension (PointNet2=512, PTv3_light=512, PTv3=512)
        backbone_out_dim = getattr(self.scene_model, 'output_dim', 512)
        self.scene_projection = nn.Linear(backbone_out_dim, self.d_model)
        self.logger.info(f"Scene projection: {backbone_out_dim} -> {self.d_model}")
        
        self.text_encoder = None  # Lazily initialized
        self.text_processor = TextConditionProcessor(
            text_dim=512,
            context_dim=self.d_model,
            dropout=self.dropout,
            use_negative_prompts=self.use_negative_prompts,
        )
        
        # Global-Local Scene Conditioning 配置
        self.use_global_local_conditioning = getattr(cfg, 'use_global_local_conditioning', False)
        
        if self.use_global_local_conditioning:
            # 全局池化模块
            global_pool_cfg = getattr(cfg, 'global_pool', {})
            num_latents = global_pool_cfg.get('num_latents', 128)
            pool_num_layers = global_pool_cfg.get('num_layers', 1)
            pool_dropout = global_pool_cfg.get('dropout', 0.0)
            
            self.global_scene_pool = GlobalScenePool(
                d_model=self.d_model,
                num_latents=num_latents,
                num_layers=pool_num_layers,
                num_heads=self.num_heads,
                d_head=self.d_head,
                dropout=pool_dropout,
                use_flash_attention=self.use_flash_attention,
            )
            
            # 局部选择器
            local_selector_cfg = getattr(cfg, 'local_selector', {})
            selector_type = local_selector_cfg.get('type', 'knn')
            
            self.local_selector = build_local_selector(
                selector_type=selector_type,
                k=local_selector_cfg.get('k', 32),
                radius=local_selector_cfg.get('radius', 0.05),
                stochastic=local_selector_cfg.get('stochastic', False),
            )
            
            self.logger.info(
                f"Global-Local conditioning enabled: "
                f"num_latents={num_latents}, selector={selector_type}, "
                f"k={local_selector_cfg.get('k', 32)}"
            )
        else:
            self.global_scene_pool = None
            self.local_selector = None
        
        # Initialize weights
        init_weights(self)

        # Zero-init velocity head for stable FM training (ensures near-zero initial field)
        if self.pred_mode == 'velocity' and hasattr(self, 'velocity_head'):
            # 支持 Linear 或 Sequential(LN+Linear)
            linear_module = self.velocity_head[-1] if isinstance(self.velocity_head, nn.Sequential) else self.velocity_head
            if hasattr(linear_module, 'weight') and linear_module.weight is not None:
                nn.init.zeros_(linear_module.weight)
            if hasattr(linear_module, 'bias') and linear_module.bias is not None:
                nn.init.zeros_(linear_module.bias)
        
        self.logger.info(f"DiTFM initialized in pred_mode: {self.pred_mode}, "
                        f"continuous_time: {self.continuous_time}")
    
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        Forward pass for DiT-FM.
        
        Args:
            x_t: Noisy/interpolated input, shape [B, C] or [B, num_grasps, C]
            ts: Time values
                - For FM: continuous t ∈ [0, 1], shape [B]
                - For DDPM: discrete timesteps, shape [B]
            data: Dictionary with conditioning information
            
        Returns:
            Predicted velocity (FM) or noise (DDPM), same shape as x_t
        """
        # Normalize input shape
        x_t, single_grasp_mode = self._normalize_input_shape(x_t)
        
        # Debug: log input statistics
        if self.debug_log_stats:
            self._log_tensor_stats(x_t, "input x_t")
            self._log_tensor_stats(ts, "timesteps")
        
        # Tokenize grasps
        grasp_tokens = self.grasp_tokenizer(x_t)  # [B, num_grasps, d_model]
        num_grasps = grasp_tokens.shape[1]
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            grasp_tokens = self.pos_embedding(grasp_tokens)
        
        # Ensure ts 在与模型相同的设备上
        model_device = self._get_device()
        try:
            ts = ts.to(model_device)
        except Exception:
            pass

        # Process time embedding based on prediction mode
        if self.pred_mode == 'velocity' and self.continuous_time:
            # Continuous time for Flow Matching
            time_emb_base = self.time_embedding(ts.float())  # [B, d_model]
        else:
            # Discrete timesteps for DDPM
            time_emb_base = self.time_embedding(ts)  # [B, d_model]
        
        projected_time = self.time_proj(time_emb_base) if self.time_proj is not None else time_emb_base
        time_emb = projected_time if self.use_adaptive_norm else None

        # 为 MLP 时间门控准备 time_emb（在未使用 adaptive_norm 时）
        time_emb_for_gate = None
        if self.use_t_aware_conditioning and self.time_gate is not None and time_emb is None:
            # 仅在门控类型为 MLP 时需要 time_emb
            try:
                if getattr(self.time_gate, 'gate_type', 'cos2') == 'mlp' and hasattr(self, 'time_proj_gate') and self.time_proj_gate is not None:
                    time_emb_for_gate = self.time_proj_gate(time_emb_base)
            except Exception:
                time_emb_for_gate = None
        
        # Get conditioning features
        scene_context = data.get('scene_cond')  # [B, N, d_model]
        if scene_context is not None and isinstance(scene_context, torch.Tensor):
            if scene_context.device != model_device:
                scene_context = scene_context.to(model_device)
                data['scene_cond'] = scene_context
        text_context = data.get('text_cond')  # [B, d_model] or None
        if text_context is not None and isinstance(text_context, torch.Tensor):
            if text_context.device != model_device:
                text_context = text_context.to(model_device)
                data['text_cond'] = text_context
        scene_mask = self._normalize_scene_mask(data.get('scene_mask', None), scene_context)
        if scene_mask is not None:
            data['scene_mask'] = scene_mask
        
        if text_context is not None and hasattr(text_context, "dim") and text_context.dim() == 2:
            text_context = text_context.unsqueeze(1)  # [B, 1, d_model]
        
        # 提取 token 级别的文本特征（如果启用且可用）
        text_tokens = None
        text_token_mask = None
        if self.use_text_tokens and self.use_text_condition:
            raw_text_tokens = data.get("text_tokens")
            if raw_text_tokens is not None and isinstance(raw_text_tokens, torch.Tensor):
                text_tokens = raw_text_tokens.to(model_device)
                raw_text_token_mask = data.get("text_token_mask")
                if raw_text_token_mask is not None and isinstance(raw_text_token_mask, torch.Tensor):
                    text_token_mask = raw_text_token_mask.to(model_device)
                if self.debug_log_stats:
                    self._log_tensor_stats(text_tokens, "text_tokens")
        
        # Extract scene_xyz for geometric attention bias
        scene_xyz = None
        if self.use_geometric_bias:
            # 优先使用采样后的 xyz（与 scene_context 的点数匹配）
            if 'scene_xyz_sampled' in data and data['scene_xyz_sampled'] is not None:
                scene_xyz = data['scene_xyz_sampled']
                if not isinstance(scene_xyz, torch.Tensor):
                    scene_xyz = torch.as_tensor(scene_xyz, dtype=torch.float32)
                if scene_xyz.device != model_device:
                    scene_xyz = scene_xyz.to(model_device)
                if self.debug_log_stats:
                    self.logger.debug(f"Using sampled scene_xyz: shape={scene_xyz.shape}")
            else:
                # 回退到原始点云坐标（可能会导致形状不匹配）
                from .geometric_attention_bias import extract_scene_xyz
                scene_xyz = extract_scene_xyz(scene_context, data)
                if scene_xyz is not None:
                    if scene_xyz.device != model_device:
                        scene_xyz = scene_xyz.to(model_device)
                    self.logger.warning(
                        f"Using original scene_xyz (shape={scene_xyz.shape}), "
                        f"may cause shape mismatch with scene_context. "
                        f"Consider using scene_xyz_sampled instead."
                    )
        
        # Compute t_scalar for time-aware conditioning (FM path: ts is already continuous [0,1])
        t_scalar = None
        if self.use_t_aware_conditioning:
            t_scalar = ts.clamp(0.0, 1.0)  # Ensure in [0, 1] range
        
        # 准备条件向量（AdaLN-Zero 模式）
        cond_vector = None
        if self.use_adaln_zero:
            if self.adaln_mode == 'simple':
                if projected_time is None:
                    raise DiTConditioningError("projected_time is required when use_adaln_zero=True")
                cond_vector = projected_time
                if self.debug_log_stats:
                    self._log_tensor_stats(cond_vector, "cond_vector (simple)")
            else:
                if projected_time is None:
                    raise DiTConditioningError("projected_time is required when use_adaln_zero=True and adaln_mode='multi'")
                cond_vector = build_adaln_cond_vector(
                    projected_time,
                    use_scene_pooling=self.use_scene_pooling,
                    scene_to_time=self.scene_to_time,
                    scene_context=scene_context,
                    scene_mask=scene_mask,
                    use_text_condition=self.use_text_condition,
                    text_to_time=self.text_to_time,
                    text_context=text_context,
                    logger=self.logger,
                )
                if self.debug_log_stats:
                    self._log_tensor_stats(cond_vector, "cond_vector")
        
        # Global-Local Scene Conditioning（全局-局部两阶段场景条件）
        latent_global = None
        local_indices = None
        local_mask = None
        if self.use_global_local_conditioning and scene_context is not None:
            # 1. 全局阶段：计算 global latent tokens
            latent_global = self.global_scene_pool(scene_context, scene_mask)
            if self.debug_log_stats:
                self._log_tensor_stats(latent_global, "latent_global")
            
            # 2. 准备局部阶段所需的 scene_xyz 和 grasp_translations
            # 优先使用采样后的 xyz（与 scene_context 匹配）
            local_scene_xyz = None
            if 'scene_xyz_sampled' in data and data['scene_xyz_sampled'] is not None:
                local_scene_xyz = data['scene_xyz_sampled']
                if not isinstance(local_scene_xyz, torch.Tensor):
                    local_scene_xyz = torch.as_tensor(local_scene_xyz, dtype=torch.float32)
                if local_scene_xyz.device != model_device:
                    local_scene_xyz = local_scene_xyz.to(model_device)
            elif 'scene_xyz' in data:
                local_scene_xyz = data['scene_xyz']
                if not isinstance(local_scene_xyz, torch.Tensor):
                    local_scene_xyz = torch.as_tensor(local_scene_xyz, dtype=torch.float32)
                if local_scene_xyz.device != model_device:
                    local_scene_xyz = local_scene_xyz.to(model_device)
            elif 'scene_pc' in data:
                # 从 scene_pc 中提取 xyz（前3维）
                scene_pc = data['scene_pc']
                if isinstance(scene_pc, torch.Tensor):
                    local_scene_xyz = scene_pc[..., :3].to(model_device)
                else:
                    local_scene_xyz = torch.as_tensor(scene_pc, dtype=torch.float32).to(model_device)[..., :3]
            else:
                self.logger.warning(
                    "Global-local conditioning enabled but scene_xyz not found. "
                    "Local stage will be skipped."
                )
            
            # 3. 局部阶段：使用 local_selector 选取每个抓取的局部邻域
            if local_scene_xyz is not None:
                # 对齐检查：local_scene_xyz 点数必须与 scene_context 一致，否则跳过局部阶段
                try:
                    n_xyz = local_scene_xyz.shape[1]
                    n_ctx = scene_context.shape[1]
                    if n_xyz != n_ctx:
                        self.logger.warning(
                            f"Local selector xyz count (N={n_xyz}) != scene_context count (N={n_ctx}); "
                            f"fallback to global-only for local stage."
                        )
                        local_scene_xyz = None
                except Exception as shape_exc:
                    self.logger.warning(f"Failed to verify xyz/context alignment: {shape_exc}; skipping local stage.")
                    local_scene_xyz = None

            if local_scene_xyz is not None:
                # grasp_translations: 从 x_t 中提取平移部分（前3维）
                grasp_translations = x_t[..., :3]  # (B, num_grasps, 3)
                # 使用 local_selector 选取局部邻域索引
                local_indices, local_mask = self.local_selector(
                    grasp_translations=grasp_translations,
                    scene_xyz=local_scene_xyz,
                    scene_mask=scene_mask,
                )
                if self.debug_log_stats:
                    self.logger.debug(
                        f"Local neighborhoods selected: indices shape={local_indices.shape}, "
                        f"mask shape={local_mask.shape}"
                    )
        
        # Apply DiT blocks with dual-stream and fused single-stream support
        time_emb_for_blocks = time_emb if time_emb is not None else time_emb_for_gate
        num_scene_tokens = scene_context.shape[1] if scene_context is not None else 0
        x = grasp_tokens
        current_scene = scene_context
        double_block_count = 0

        for idx, block in enumerate(self.dit_blocks):
            if not getattr(block, "is_double_stream", False):
                break
            if current_scene is None:
                raise DiTConditioningError(
                    "Double-stream block requires scene_context, but it is None."
                )
            x, current_scene = block(
                grasp_tokens=x,
                scene_tokens=current_scene,
                cond_vector=cond_vector,
                scene_mask=scene_mask,
            )
            double_block_count += 1
            if self.debug_check_nan:
                if torch.isnan(x).any():
                    self.logger.error(f"NaN detected after DiT double block {idx}")
                    raise RuntimeError("NaN in DiT forward pass (double stream)")
                if torch.isnan(current_scene).any():
                    self.logger.error(f"NaN detected in scene tokens after double block {idx}")
                    raise RuntimeError("NaN in scene stream during double blocks")

        if double_block_count > 0 and current_scene is not None:
            x = torch.cat([current_scene, x], dim=1)
            if scene_mask is not None:
                grasp_mask = torch.ones(
                    x.shape[0],
                    num_grasps,
                    device=scene_mask.device,
                    dtype=scene_mask.dtype,
                )
                scene_mask_for_single = torch.cat([scene_mask, grasp_mask], dim=1)
            else:
                scene_mask_for_single = None
            scene_context_for_single = None
        else:
            scene_mask_for_single = scene_mask
            scene_context_for_single = current_scene

        for idx in range(double_block_count, len(self.dit_blocks)):
            block = self.dit_blocks[idx]
            if getattr(block, "expects_concatenated_sequence", False):
                x = block(
                    x=x,
                    cond_vector=cond_vector,
                    mask=scene_mask_for_single,
                )
            else:
                x = block(
                    x=x,
                    time_emb=time_emb_for_blocks,
                    scene_context=scene_context_for_single,
                    text_context=text_context,
                    scene_mask=scene_mask_for_single,
                    cond_vector=cond_vector,
                    grasp_poses=x_t,
                    scene_xyz=scene_xyz,
                    latent_global=latent_global,
                    local_indices=local_indices,
                    local_mask=local_mask,
                    text_tokens=text_tokens,
                    text_token_mask=text_token_mask,
                    t_scalar=t_scalar,
                )
            if self.debug_check_nan and torch.isnan(x).any():
                self.logger.error(f"NaN detected after DiT block {idx}")
                raise RuntimeError("NaN in DiT forward pass")

        if double_block_count > 0 and num_scene_tokens > 0:
            x = x[:, num_scene_tokens:, ...]
        
        # Output projection based on prediction mode
        if self.pred_mode == 'velocity':
            # Direct velocity prediction for FM
            x_flat = x.reshape(-1, self.d_model)
            output = self.velocity_head(x_flat)
            output = output.reshape(x.shape[0], x.shape[1], self.d_x)
        else:
            # Use standard output projection for DDPM/pose mode
            output = self.output_projection(x)
        
        # Restore original shape if needed
        if single_grasp_mode:
            output = output.squeeze(1)
        
        # Debug: log output statistics
        if self.debug_log_stats:
            self._log_tensor_stats(output, f"output ({self.pred_mode})")
        
        return output
    
    def condition(self, data: Dict) -> Dict:
        """
        Pre-compute conditioning features (scene, text).
        Reuses logic from original DiT for compatibility.
        """
        condition_dict = {}
        
        # Scene conditioning
        if 'scene_pc' in data and data['scene_pc'] is not None:
            scene_feat = self._prepare_scene_features(data)
            condition_dict['scene_cond'] = scene_feat
        else:
            condition_dict['scene_cond'] = None
        
        # Text conditioning
        if self.use_text_condition and 'positive_prompt' in data:
            text_features = self._prepare_text_features(data, 
                                                       condition_dict.get('scene_cond'))
            condition_dict.update(text_features)
        else:
            condition_dict['text_cond'] = None
            condition_dict['text_mask'] = None
            if self.use_negative_prompts:
                condition_dict['neg_pred'] = None
                condition_dict['neg_text_features'] = None
        
        return condition_dict
    
    # =====================
    # Helper Methods (mostly reused from DiT)
    # =====================
    
    def _normalize_input_shape(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Normalize input to [B, num_grasps, C] format."""
        if x_t.dim() == 2:
            return x_t.unsqueeze(1), True
        elif x_t.dim() == 3:
            return x_t, False
        else:
            raise DiTDimensionError(f"Unsupported input dimension: {x_t.dim()}")
    
    def _log_tensor_stats(self, tensor: torch.Tensor, name: str):
        """Log tensor statistics for debugging."""
        if tensor is None:
            return
        
        stats = {
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'device': tensor.device,
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item() if tensor.numel() > 1 else 0.0,
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item()
        }
        
        self.logger.debug(f"[DiTFM] {name}: {stats}")
    
    def _full_scene_mask(self, scene_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Create an all-ones scene mask aligned with the scene context."""
        if scene_context is None or not isinstance(scene_context, torch.Tensor):
            return None  # Caller handles fallback when context is missing
        B, N, _ = scene_context.shape
        return torch.ones(B, N, device=scene_context.device, dtype=torch.float32)

    def _normalize_scene_mask(
        self,
        scene_mask: Optional[torch.Tensor],
        scene_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Ensure scene_mask exists, lives on the right device, and matches scene tokens."""
        if scene_mask is None:
            if scene_context is None or not isinstance(scene_context, torch.Tensor):
                return None
            return self._full_scene_mask(scene_context)

        if not isinstance(scene_mask, torch.Tensor):
            self.logger.warning(
                f"scene_mask is not a tensor ({type(scene_mask)}); assuming all scene points are valid."
            )
            if scene_context is None or not isinstance(scene_context, torch.Tensor):
                return None
            return self._full_scene_mask(scene_context)

        target_device = (
            scene_context.device if isinstance(scene_context, torch.Tensor) else self._get_device()
        )
        scene_mask = scene_mask.to(target_device)

        if scene_mask.dim() == 3:
            scene_mask = scene_mask.squeeze(1)
        if scene_mask.dtype != torch.float32:
            scene_mask = scene_mask.float()

        if scene_context is not None and isinstance(scene_context, torch.Tensor):
            expected_shape = (scene_context.shape[0], scene_context.shape[1])
            if scene_mask.shape != expected_shape:
                self.logger.warning(
                    f"scene_mask shape {tuple(scene_mask.shape)} mismatches scene_context {expected_shape}; "
                    "falling back to full-valid mask."
                )
                return self._full_scene_mask(scene_context)

        return torch.clamp(scene_mask, 0.0, 1.0)

    
    
    def _prepare_scene_features(self, data: Dict) -> torch.Tensor:
        model_device = self._get_device()
        return prepare_scene_features(
            scene_model=self.scene_model,
            scene_projection=self.scene_projection,
            data=data,
            use_rgb=self.use_rgb,
            use_object_mask=self.use_object_mask,
            device=model_device,
            logger=self.logger,
            strict=True,
        )
    
    def _prepare_text_features(self, data: Dict, scene_feat: torch.Tensor) -> Dict:
        """Process text prompts through text encoder."""
        self._ensure_text_encoder()
        device = self._get_device()
        return prepare_text_features(
            text_encoder=self.text_encoder,
            text_processor=self.text_processor,
            data=data,
            scene_feat=scene_feat,
            use_negative_prompts=self.use_negative_prompts,
            text_dropout_prob=self.text_dropout_prob,
            training=self.training,
            device=device,
            logger=self.logger,
        )
    
    def _ensure_text_encoder(self):
        """Lazily initialize text encoder."""
        if self.text_encoder is None:
            device = self._get_device()
            self.text_encoder = ensure_text_encoder(self.text_encoder, device, logger=self.logger)
    
    def _get_device(self):
        """Get model device."""
        return next(self.parameters()).device
    
    def to(self, *args, **kwargs):
        """Override to handle text encoder device movement."""
        super().to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self
