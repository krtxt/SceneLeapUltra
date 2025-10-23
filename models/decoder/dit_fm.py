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
    DiTBlock, GraspTokenizer, PositionalEmbedding, TimestepEmbedding,
    AdaptiveLayerNorm, OutputProjection, DiTModel,
    DiTValidationError, DiTInputError, DiTDimensionError,
    DiTDeviceError, DiTConditioningError, DiTGracefulFallback
)
from models.backbone import build_backbone
from models.utils.text_encoder import TextConditionProcessor, PosNegTextEncoder


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
        
        # FM-specific config
        self.continuous_time = cfg.get('continuous_time', True)
        self.freq_dim = cfg.get('freq_dim', 256)
        
        # Conditioning config
        self.use_text_condition = cfg.use_text_condition
        self.text_dropout_prob = cfg.text_dropout_prob
        self.use_negative_prompts = getattr(cfg, 'use_negative_prompts', True)
        self.use_object_mask = cfg.use_object_mask
        self.use_rgb = cfg.use_rgb
        
        # Debug config
        self.debug_check_nan = cfg.get('debug', {}).get('check_nan', True)
        self.debug_log_stats = cfg.get('debug', {}).get('log_tensor_stats', False)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
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
        
        # Additional time projection for adaptive norm
        if self.use_adaptive_norm:
            self.time_proj = nn.Sequential(
                nn.Linear(self.d_model, self.time_embed_dim),
                nn.SiLU(),
                nn.Linear(self.time_embed_dim, self.time_embed_dim)
            )
        
        # DiT transformer blocks (reuse from DiT)
        self.dit_blocks = nn.ModuleList([
            DiTBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_head=self.d_head,
                dropout=self.dropout,
                use_adaptive_norm=self.use_adaptive_norm,
                time_embed_dim=self.time_embed_dim if self.use_adaptive_norm else None
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projections for different modes
        self.output_projection = OutputProjection(self.d_model, self.d_x)
        
        # Separate heads for different prediction targets
        if self.pred_mode == 'velocity':
            self.velocity_head = nn.Linear(self.d_model, self.d_x)
            self.logger.info("Initialized velocity head for Flow Matching")
        
        # Conditioning modules (reuse from DiT)
        backbone_cfg = self._adjust_backbone_config(cfg.backbone, cfg.use_rgb, cfg.use_object_mask)
        self.scene_model = build_backbone(backbone_cfg)
        self.scene_projection = nn.Linear(512, self.d_model)
        
        self.text_encoder = None  # Lazily initialized
        self.text_processor = TextConditionProcessor(
            text_dim=512,
            context_dim=self.d_model,
            dropout=self.dropout,
            use_negative_prompts=self.use_negative_prompts,
        )
        
        # Initialize weights
        self._init_weights()
        
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
        
        # Add positional embeddings
        if self.pos_embedding is not None:
            grasp_tokens = self.pos_embedding(grasp_tokens)
        
        # Process time embedding based on prediction mode
        if self.pred_mode == 'velocity' and self.continuous_time:
            # Continuous time for Flow Matching
            time_emb_base = self.time_embedding(ts.float())  # [B, d_model]
        else:
            # Discrete timesteps for DDPM
            time_emb_base = self.time_embedding(ts)  # [B, d_model]
        
        # Project for adaptive norm if needed
        if self.use_adaptive_norm:
            time_emb = self.time_proj(time_emb_base)  # [B, time_embed_dim]
        else:
            time_emb = None
        
        # Get conditioning features
        scene_context = data.get('scene_cond')  # [B, N, d_model]
        text_context = data.get('text_cond')  # [B, d_model] or None
        
        if text_context is not None and text_context.dim() == 2:
            text_context = text_context.unsqueeze(1)  # [B, 1, d_model]
        
        # Apply DiT blocks
        x = grasp_tokens
        for i, block in enumerate(self.dit_blocks):
            x = block(x, time_emb, scene_context, text_context)
            
            if self.debug_check_nan and torch.isnan(x).any():
                self.logger.error(f"NaN detected after DiT block {i}")
                raise RuntimeError("NaN in DiT forward pass")
        
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
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _adjust_backbone_config(self, backbone_cfg, use_rgb, use_object_mask):
        """Adjust backbone config based on input modalities."""
        import copy
        adjusted_cfg = copy.deepcopy(backbone_cfg)
        
        # Calculate input dimensions
        total_input_dim = 3 + (3 if use_rgb else 0) + (1 if use_object_mask else 0)
        feature_input_dim = (3 if use_rgb else 0) + (1 if use_object_mask else 0)
        
        backbone_name = getattr(adjusted_cfg, 'name', '').lower()
        
        if backbone_name == 'pointnet2':
            if hasattr(adjusted_cfg, 'layer1') and hasattr(adjusted_cfg.layer1, 'mlp_list'):
                mlp_list = list(adjusted_cfg.layer1.mlp_list)
                mlp_list[0] = feature_input_dim
                adjusted_cfg.layer1.mlp_list = mlp_list
        elif backbone_name == 'ptv3':
            adjusted_cfg.in_channels = total_input_dim
        
        return adjusted_cfg
    
    def _prepare_scene_features(self, data: Dict) -> torch.Tensor:
        """Process scene point cloud through backbone."""
        scene_pc = data['scene_pc']
        
        if not self.use_rgb:
            scene_pc = scene_pc[..., :3]
        
        if self.use_object_mask and 'object_mask' in data:
            object_mask = data['object_mask']
            if object_mask.dim() == 2:
                object_mask = object_mask.unsqueeze(-1)
            scene_points = torch.cat([scene_pc, object_mask], dim=-1)
        else:
            scene_points = scene_pc
        
        _, scene_feat = self.scene_model(scene_points)
        scene_feat = scene_feat.permute(0, 2, 1).contiguous()
        scene_feat = self.scene_projection(scene_feat)
        
        return scene_feat
    
    def _prepare_text_features(self, data: Dict, scene_feat: torch.Tensor) -> Dict:
        """Process text prompts through text encoder."""
        self._ensure_text_encoder()
        
        batch_size = scene_feat.shape[0] if scene_feat is not None else 1
        positive_prompts = data['positive_prompt']
        
        # Encode positive prompts
        pos_text_features = self.text_encoder.encode_positive(positive_prompts)
        
        # Encode negative prompts if available
        neg_text_features = None
        if self.use_negative_prompts and 'negative_prompts' in data:
            neg_text_features = self.text_encoder.encode_negative(data['negative_prompts'])
        
        # Text dropout mask
        text_mask = torch.ones(batch_size, 1, device=self._get_device())
        if self.training:
            text_mask = torch.bernoulli(
                torch.full((batch_size, 1), 1.0 - self.text_dropout_prob,
                          device=self._get_device())
            )
        
        # Process through text processor
        scene_embedding = torch.mean(scene_feat, dim=1) if scene_feat is not None else None
        pos_text_features_out, neg_pred = self.text_processor(
            pos_text_features, neg_text_features, scene_embedding
        )
        
        result = {
            'text_cond': pos_text_features_out * text_mask,
            'text_mask': text_mask
        }
        
        if self.use_negative_prompts:
            result['neg_pred'] = neg_pred
            result['neg_text_features'] = neg_text_features
        
        return result
    
    def _ensure_text_encoder(self):
        """Lazily initialize text encoder."""
        if self.text_encoder is None:
            device = self._get_device()
            self.text_encoder = PosNegTextEncoder(device=device)
            self.text_encoder.to(device)
            self.logger.info(f"Text encoder lazily initialized on {device}")
    
    def _get_device(self):
        """Get model device."""
        return next(self.parameters()).device
    
    def to(self, *args, **kwargs):
        """Override to handle text encoder device movement."""
        super().to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self
