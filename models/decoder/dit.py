import torch
import torch.nn as nn
import math
import logging
from typing import Dict, Optional, Any
from einops import rearrange
from contextlib import nullcontext

from models.utils.diffusion_utils import timestep_embedding
from models.backbone import build_backbone
from models.utils.text_encoder import TextConditionProcessor, PosNegTextEncoder
from .dit_config_validation import validate_dit_config, get_dit_config_summary
from .dit_validation import (
    validate_dit_inputs, DiTValidationError, DiTInputError, DiTDimensionError,
    DiTDeviceError, DiTConditioningError, DiTGracefulFallback
)
from .dit_memory_optimization import (
    MemoryMonitor, EfficientAttention, GradientCheckpointedDiTBlock,
    BatchProcessor, optimize_memory_usage, get_memory_optimization_config
)


class GraspTokenizer(nn.Module):
    """
    Converts grasp poses to tokens for transformer processing.
    """
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, grasp_poses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grasp_poses: (B, num_grasps, input_dim) or (B, input_dim)
        Returns:
            tokens: (B, num_grasps, d_model) or (B, 1, d_model)
        """
        if grasp_poses.dim() == 2:
            # Single grasp format - add sequence dimension
            grasp_poses = grasp_poses.unsqueeze(1)  # (B, 1, input_dim)
        
        # Project to model dimension
        tokens = self.input_projection(grasp_poses)  # (B, num_grasps, d_model)
        tokens = self.layer_norm(tokens)
        
        return tokens


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for grasp sequence positions.
    """
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            x + positional embeddings: (B, seq_len, d_model)
        """
        seq_len = x.size(1)
        max_len = self.pos_embedding.size(0)
        
        if seq_len <= max_len:
            # Normal case: use exact positional embeddings
            pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)
        else:
            # Sequence longer than max_len: repeat the last position embedding
            pos_emb_base = self.pos_embedding  # (max_len, d_model)
            # Repeat the last embedding for positions beyond max_len
            last_emb = self.pos_embedding[-1:].expand(seq_len - max_len, -1)  # (seq_len - max_len, d_model)
            pos_emb = torch.cat([pos_emb_base, last_emb], dim=0).unsqueeze(0)  # (1, seq_len, d_model)
        
        return x + pos_emb


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding compatible with existing diffusion pipeline.
    """
    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,)
        Returns:
            time_embeddings: (B, time_embed_dim)
        """
        # Use existing timestep_embedding function for compatibility
        t_emb = timestep_embedding(timesteps, self.d_model)
        return self.time_embed(t_emb)


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive layer normalization conditioned on timestep embeddings.
    """
    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift = nn.Linear(time_embed_dim, d_model * 2)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
            time_emb: (B, time_embed_dim)
        Returns:
            normalized_x: (B, seq_len, d_model)
        """
        x = self.layer_norm(x)
        scale, shift = self.scale_shift(time_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # (B, 1, d_model)
        shift = shift.unsqueeze(1)  # (B, 1, d_model)
        return x * (1 + scale) + shift


# MultiHeadAttention is now replaced by EfficientAttention from memory optimization module


class FeedForward(nn.Module):
    """
    Feed-forward network for DiT blocks.
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    DiT transformer block with self-attention, cross-attention, and feed-forward layers.
    Enhanced with memory optimization features.
    """
    def __init__(self, d_model: int, num_heads: int, d_head: int, dropout: float = 0.1,
                 use_adaptive_norm: bool = True, time_embed_dim: Optional[int] = None,
                 chunk_size: int = 512, use_flash_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_adaptive_norm = use_adaptive_norm
        
        # Memory-efficient attention layers
        self.self_attention = EfficientAttention(d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention)
        self.scene_cross_attention = EfficientAttention(d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention)
        self.text_cross_attention = EfficientAttention(d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        
        # Normalization layers
        if use_adaptive_norm and time_embed_dim is not None:
            self.norm1 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm3 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm4 = AdaptiveLayerNorm(d_model, time_embed_dim)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.norm4 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None,
                scene_context: Optional[torch.Tensor] = None, 
                text_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, num_grasps, d_model)
            time_emb: (B, time_embed_dim) or None
            scene_context: (B, N_points, d_model) or None
            text_context: (B, 1, d_model) or None
        Returns:
            output: (B, num_grasps, d_model)
        """
        # Self-attention among grasps
        if self.use_adaptive_norm and time_emb is not None:
            norm_x = self.norm1(x, time_emb)
        else:
            norm_x = self.norm1(x)
        x = x + self.self_attention(norm_x)
        
        # Cross-attention with scene features
        if scene_context is not None:
            if self.use_adaptive_norm and time_emb is not None:
                norm_x = self.norm2(x, time_emb)
            else:
                norm_x = self.norm2(x)
            x = x + self.scene_cross_attention(norm_x, scene_context, scene_context)
        
        # Cross-attention with text features (if available)
        if text_context is not None:
            if self.use_adaptive_norm and time_emb is not None:
                norm_x = self.norm3(x, time_emb)
            else:
                norm_x = self.norm3(x)
            x = x + self.text_cross_attention(norm_x, text_context, text_context)
        
        # Feed-forward network
        if self.use_adaptive_norm and time_emb is not None:
            norm_x = self.norm4(x, time_emb)
        else:
            norm_x = self.norm4(x)
        x = x + self.feed_forward(norm_x)
        
        return x


class OutputProjection(nn.Module):
    """
    Projects DiT output back to grasp pose dimension.
    """
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_grasps, d_model)
        Returns:
            output: (B, num_grasps, output_dim)
        """
        return self.projection(x)


class DiTModel(nn.Module):
    """
    Diffusion Transformer (DiT) model for grasp synthesis.
    
    This model serves as a drop-in replacement for UNet while maintaining
    full compatibility with the existing diffusion pipeline.
    """
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        # Validate configuration
        validate_dit_config(cfg)
        
        # Log configuration summary
        config_summary = get_dit_config_summary(cfg)
        logging.info(f"Initializing DiT model with configuration: {config_summary}")
        
        # Input Dimension Config
        rot_to_dim = {'quat': 23, 'r6d': 25}
        self._rot_type = cfg.rot_type
        self.d_x = rot_to_dim[cfg.rot_type]
        
        # Initialize validation and fallback systems
        self.logger = logging.getLogger(__name__)
        self.fallback = DiTGracefulFallback(self.logger)
        
        # Model Architecture Config
        self.d_model = cfg.d_model
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.d_head = cfg.d_head
        self.dropout = cfg.dropout
        self.max_sequence_length = cfg.max_sequence_length
        self.use_learnable_pos_embedding = cfg.use_learnable_pos_embedding
        
        # Timestep embedding config
        self.time_embed_dim = cfg.time_embed_dim
        self.use_adaptive_norm = cfg.use_adaptive_norm
        
        # Conditioning config
        self.use_text_condition = cfg.use_text_condition
        self.text_dropout_prob = cfg.text_dropout_prob
        self.use_negative_prompts = getattr(cfg, 'use_negative_prompts', True)
        self.use_object_mask = cfg.use_object_mask
        self.use_rgb = cfg.use_rgb
        
        # Memory optimization config
        self.gradient_checkpointing = getattr(cfg, 'gradient_checkpointing', False)
        self.use_flash_attention = getattr(cfg, 'use_flash_attention', False)
        self.attention_chunk_size = getattr(cfg, 'attention_chunk_size', 512)

        # Memory monitoring config
        self.memory_monitoring = getattr(cfg, 'memory_monitoring', True)

        # Initialize memory monitoring utilities
        self.memory_monitor = MemoryMonitor(self.logger)
        self.batch_processor = BatchProcessor(logger=self.logger)

        # Get memory optimization configuration
        memory_config = get_memory_optimization_config({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'max_sequence_length': self.max_sequence_length
        })
        
        # Apply memory optimizations if not explicitly configured
        if not hasattr(cfg, 'gradient_checkpointing'):
            self.gradient_checkpointing = memory_config['gradient_checkpointing']
        if not hasattr(cfg, 'use_flash_attention'):
            self.use_flash_attention = memory_config['use_flash_attention']
        if not hasattr(cfg, 'attention_chunk_size'):
            self.attention_chunk_size = memory_config['attention_chunk_size']
        
        self.logger.info(f"Memory optimizations: checkpointing={self.gradient_checkpointing}, "
                        f"flash_attention={self.use_flash_attention}, chunk_size={self.attention_chunk_size}")
        
        # Core DiT components
        self.grasp_tokenizer = GraspTokenizer(self.d_x, self.d_model)
        
        if self.use_learnable_pos_embedding:
            self.pos_embedding = PositionalEmbedding(self.d_model, self.max_sequence_length)
        else:
            self.pos_embedding = None
            
        self.time_embedding = TimestepEmbedding(self.d_model, self.time_embed_dim)
        
        # DiT transformer blocks with memory optimization
        dit_blocks = []
        for i in range(self.num_layers):
            block = DiTBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_head=self.d_head,
                dropout=self.dropout,
                use_adaptive_norm=self.use_adaptive_norm,
                time_embed_dim=self.time_embed_dim if self.use_adaptive_norm else None,
                chunk_size=self.attention_chunk_size,
                use_flash_attention=self.use_flash_attention
            )
            
            # Wrap with gradient checkpointing if enabled
            if self.gradient_checkpointing:
                block = GradientCheckpointedDiTBlock(block, use_checkpointing=True)
            
            dit_blocks.append(block)
        
        self.dit_blocks = nn.ModuleList(dit_blocks)
        
        # Output projection
        self.output_projection = OutputProjection(self.d_model, self.d_x)
        
        # Conditioning modules (reuse from UNet)
        # Adjust backbone config based on use_rgb and use_object_mask
        backbone_cfg = self._adjust_backbone_config(cfg.backbone, cfg.use_rgb, cfg.use_object_mask)
        self.scene_model = build_backbone(backbone_cfg)
        
        # Scene feature projection to match model dimension
        # The backbone typically outputs 512-dim features, project to d_model
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
        """
        Adjusts the backbone configuration based on use_rgb and use_object_mask settings.
        Supports both PointNet2 and PTv3 backbones.

        Args:
            backbone_cfg: Original backbone configuration
            use_rgb: Whether to use RGB features as input
            use_object_mask: Whether to use object mask as additional input

        Returns:
            Modified backbone configuration with correct input dimensions
        """
        import copy
        adjusted_cfg = copy.deepcopy(backbone_cfg)

        # Calculate total input dimension:
        # - XYZ coordinates: 3 channels (handled automatically by PointNet2, explicitly by PTv3)
        # - RGB features: 3 channels (if use_rgb is True)
        # - Object mask (if enabled): +1 channel
        # Input order: xyz + rgb + object_mask
        total_input_dim = 3 + (3 if use_rgb else 0) + (1 if use_object_mask else 0)
        feature_input_dim = (3 if use_rgb else 0) + (1 if use_object_mask else 0)  # rgb + optional mask (for PointNet2)

        backbone_name = getattr(adjusted_cfg, 'name', '').lower()

        if backbone_name == 'pointnet2':
            # For PointNet2: adjust the first layer's mlp_list first parameter
            # PointNet2 automatically handles xyz coordinates when use_xyz=True
            if (hasattr(adjusted_cfg, 'layer1') and
                hasattr(adjusted_cfg.layer1, 'mlp_list') and
                len(adjusted_cfg.layer1.mlp_list) > 0):
                mlp_list = list(adjusted_cfg.layer1.mlp_list)
                mlp_list[0] = feature_input_dim  # RGB (optional) + optional mask
                adjusted_cfg.layer1.mlp_list = mlp_list

        elif backbone_name == 'ptv3':
            # For PTv3: adjust the in_channels parameter
            # PTv3 expects the total input dimension including xyz coordinates
            adjusted_cfg.in_channels = total_input_dim  # xyz + rgb (optional) + optional mask

        return adjusted_cfg
    
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        Applies the DiT model to a batch of noisy inputs with memory optimization.

        Args:
            x_t: The noisy input grasp pose, shape (B, C) or (B, num_grasps, C).
            ts: The batch of timesteps, shape (B,).
            data: A dictionary containing conditional information.

        Returns:
            The predicted noise or denoised target.
        """
        # Monitor memory usage during forward pass (can be disabled via cfg.memory_monitoring)
        context_mgr = (
            self.memory_monitor.monitor_peak_memory("DiT_forward_pass")
            if self.memory_monitoring else nullcontext()
        )
        with context_mgr:
            try:
                # Check memory pressure and log warnings if needed (optional)
                if self.memory_monitoring:
                    memory_status = self.memory_monitor.check_memory_pressure()
                    if memory_status['under_pressure']:
                        for hint in memory_status['optimization_hints']:
                            self.logger.warning(f"Memory optimization hint: {hint}")

                # Comprehensive input validation with automatic corrections
                model_device = self._get_device()
                x_t, ts, data = validate_dit_inputs(
                    x_t=x_t,
                    ts=ts,
                    data=data,
                    d_x=self.d_x,
                    rot_type=self._rot_type,
                    model_device=model_device,
                    max_sequence_length=self.max_sequence_length,
                    allow_auto_correction=True,
                    logger=self.logger
                )
                
            except DiTValidationError as e:
                self.logger.error(f"Input validation failed: {e}")
                raise DiTInputError(f"Invalid inputs to DiT model: {e}")
        
        # Handle input dimensions
        original_shape = x_t.shape
        if x_t.dim() == 2:
            # Single grasp format - add sequence dimension for processing
            x_t = x_t.unsqueeze(1)  # (B, 1, d_x)
            single_grasp_mode = True
        elif x_t.dim() == 3:
            # Multi-grasp format
            single_grasp_mode = False
        else:
            # This should not happen after validation, but keep as safety check
            raise DiTDimensionError(f"Unsupported input dimension: {x_t.dim()}. Expected 2 or 3.")
        
        B, num_grasps, input_d_x = x_t.shape
        
        try:
            # 1. Tokenize grasp poses
            grasp_tokens = self.grasp_tokenizer(x_t)  # (B, num_grasps, d_model)
            
            # 2. Add positional embeddings
            if self.pos_embedding is not None:
                grasp_tokens = self.pos_embedding(grasp_tokens)
            
            # 3. Get timestep embeddings
            time_emb = self.time_embedding(ts)  # (B, time_embed_dim)
            
            # 4. Get conditioning features with error handling
            scene_context = data.get("scene_cond")  # (B, N_points, d_model)
            text_context = data.get("text_cond")    # (B, d_model) or None
            
            # Validate conditioning features
            if scene_context is not None:
                if not isinstance(scene_context, torch.Tensor):
                    raise DiTConditioningError(f"scene_cond must be torch.Tensor, got {type(scene_context)}")
                if scene_context.device != model_device:
                    scene_context = scene_context.to(model_device)
                    self.logger.debug(f"Moved scene_cond to device {model_device}")
            
            # Prepare text context for cross-attention
            if text_context is not None and self.use_text_condition:
                if not isinstance(text_context, torch.Tensor):
                    self.logger.warning(f"text_cond is not a tensor ({type(text_context)}), disabling text conditioning")
                    text_context = None
                else:
                    if text_context.device != model_device:
                        text_context = text_context.to(model_device)
                        self.logger.debug(f"Moved text_cond to device {model_device}")
                    text_context = text_context.unsqueeze(1)  # (B, 1, d_model)
            else:
                text_context = None
            
            # 5. Process through DiT blocks
            x = grasp_tokens
            for i, block in enumerate(self.dit_blocks):
                try:
                    x = block(
                        x=x,
                        time_emb=time_emb if self.use_adaptive_norm else None,
                        scene_context=scene_context,
                        text_context=text_context
                    )
                except Exception as e:
                    raise DiTConditioningError(f"Error in DiT block {i}: {e}")
            
            # 6. Output projection
            output = self.output_projection(x)  # (B, num_grasps, d_x)
            
            # 7. Handle output format to match input
            if single_grasp_mode:
                output = output.squeeze(1)  # (B, d_x)
            
            # Final validation of output
            if torch.isnan(output).any():
                self.logger.error("DiT model output contains NaN values")
                raise DiTConditioningError("Model output contains NaN values")
            
            if torch.isinf(output).any():
                self.logger.error("DiT model output contains infinite values")
                raise DiTConditioningError("Model output contains infinite values")
            
            return output
            
        except DiTConditioningError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in DiT forward pass: {e}")
            raise DiTConditioningError(f"DiT forward pass failed: {e}")
    
    def condition(self, data: Dict) -> Dict:
        """
        Pre-computes and processes all conditional features (scene, text).
        Reuses UNet's conditioning logic for compatibility with enhanced error handling.

        Args:
            data: The raw data from the dataloader.

        Returns:
            A dictionary with processed "scene_cond" and "text_cond".
        """
        try:
            # Input validation
            if data is None:
                raise DiTConditioningError("Conditioning data is None")
            
            if not isinstance(data, dict):
                raise DiTConditioningError(f"Conditioning data must be dict, got {type(data)}")
            
            # 1. Scene Feature Extraction with error handling
            try:
                if 'scene_pc' not in data or data['scene_pc'] is None:
                    raise DiTConditioningError("Missing scene_pc in conditioning data")
                
                scene_pc = data['scene_pc']
                if not isinstance(scene_pc, torch.Tensor):
                    raise DiTConditioningError(f"scene_pc must be torch.Tensor, got {type(scene_pc)}")
                
                # Ensure proper device placement
                model_device = self._get_device()
                scene_pc = scene_pc.to(model_device, dtype=torch.float32)  # (B, N, 6) - xyz + rgb
                
                # Handle RGB inclusion
                if not self.use_rgb:
                    scene_pc = scene_pc[..., :3]  # Keep only xyz
                
                # Handle object mask inclusion in scene point cloud
                if self.use_object_mask and 'object_mask' in data and data['object_mask'] is not None:
                    object_mask = data['object_mask'].to(model_device, dtype=torch.float32)
                    if object_mask.dim() == 2:
                        object_mask = object_mask.unsqueeze(-1)
                    pos = torch.cat([scene_pc, object_mask], dim=-1)
                else:
                    pos = scene_pc
                
                # Validate scene point cloud dimensions
                if pos.dim() != 3:
                    raise DiTConditioningError(f"Scene point cloud must be 3D tensor, got {pos.dim()}D")
                
                # Log the final point cloud shape and configuration
                # self.logger.info(f"Scene point cloud - Shape: {pos.shape}, use_rgb: {self.use_rgb}, use_object_mask: {self.use_object_mask}")
                
                # Extract scene features
                _, scene_feat = self.scene_model(pos)
                scene_feat = scene_feat.permute(0, 2, 1).contiguous()  # (B, N, C)
                
                # Validate scene features
                if torch.isnan(scene_feat).any():
                    self.logger.warning("Scene features contain NaN values, replacing with zeros")
                    scene_feat = torch.where(torch.isnan(scene_feat), torch.zeros_like(scene_feat), scene_feat)
                
                if torch.isinf(scene_feat).any():
                    self.logger.warning("Scene features contain infinite values, clamping")
                    scene_feat = torch.clamp(scene_feat, -1e6, 1e6)
                
                # Project scene features to model dimension
                scene_feat = self.scene_projection(scene_feat)  # (B, N, d_model)
                
            except Exception as e:
                self.logger.error(f"Scene feature extraction failed: {e}")
                # Create fallback scene features
                batch_size = 1
                if 'scene_pc' in data and data['scene_pc'] is not None:
                    batch_size = data['scene_pc'].shape[0]
                
                model_device = self._get_device()
                # Create fallback scene features with backbone output dimension, then project
                scene_feat_raw = torch.zeros(batch_size, 1024, 512, device=model_device)  # Backbone output dim
                scene_feat = self.scene_projection(scene_feat_raw)  # Project to d_model
                self.logger.warning("Using fallback scene features due to extraction failure")

            condition_dict = {"scene_cond": scene_feat, "text_cond": None, "text_mask": None}
            if self.use_negative_prompts:
                condition_dict.update({"neg_pred": None, "neg_text_features": None})

            # 2. Text Feature Extraction (if enabled) with comprehensive error handling
            if not (self.use_text_condition and 'positive_prompt' in data):
                return condition_dict

            try:
                self._ensure_text_encoder()
                b = scene_feat.shape[0]

                # Validate text prompts
                if not isinstance(data['positive_prompt'], (list, tuple)):
                    raise DiTConditioningError(f"positive_prompt must be list or tuple, got {type(data['positive_prompt'])}")
                
                if len(data['positive_prompt']) != b:
                    raise DiTConditioningError(f"Batch size mismatch: scene features {b}, prompts {len(data['positive_prompt'])}")

                # Encode positive and negative prompts
                pos_text_features = self.text_encoder.encode_positive(data['positive_prompt'])
                neg_text_features = None
                
                if self.use_negative_prompts and 'negative_prompts' in data and data['negative_prompts'] is not None:
                    try:
                        neg_text_features = self.text_encoder.encode_negative(data['negative_prompts'])
                    except Exception as e:
                        self.logger.warning(f"Negative prompt encoding failed: {e}. Continuing without negative prompts.")
                        neg_text_features = None

                # Validate text features
                if torch.isnan(pos_text_features).any():
                    self.logger.warning("Positive text features contain NaN values, replacing with zeros")
                    pos_text_features = torch.where(torch.isnan(pos_text_features), torch.zeros_like(pos_text_features), pos_text_features)

                # Apply text dropout during training
                model_device = self._get_device()
                text_mask = torch.bernoulli(torch.full((b, 1), 1.0 - self.text_dropout_prob, device=model_device)) if self.training else torch.ones(b, 1, device=model_device)
                
                # Process text features
                scene_embedding = torch.mean(scene_feat, dim=1)
                pos_text_features_out, neg_pred = self.text_processor(pos_text_features, neg_text_features, scene_embedding)

                # Validate processed text features
                if torch.isnan(pos_text_features_out).any():
                    self.logger.warning("Processed text features contain NaN values, disabling text conditioning")
                    raise DiTConditioningError("Processed text features contain NaN values")

                condition_dict.update({"text_cond": pos_text_features_out * text_mask, "text_mask": text_mask})
                if self.use_negative_prompts and neg_pred is not None:
                    condition_dict.update({
                        "neg_pred": neg_pred,  # Predicted negative prompt for CFG
                        "neg_text_features": neg_text_features, # Original negative features for loss
                    })

            except Exception as e:
                self.logger.warning(f"Text encoding failed: {e}. Falling back to scene-only conditioning.")
                # Clear all text-related conditioning
                condition_dict.update({
                    "text_cond": None,
                    "text_mask": None,
                    "neg_pred": None,
                    "neg_text_features": None
                })

            return condition_dict
            
        except DiTConditioningError:
            raise
        except Exception as e:
            self.logger.error(f"Conditioning failed with unexpected error: {e}")
            # Use fallback conditioning
            fallback_data = self.fallback.handle_conditioning_failure(data, e)
            return fallback_data
    
    # --- Device Management for Lazy-Loaded Encoder ---

    def _get_device(self):
        """Infers the model's device from its parameters."""
        return next(self.parameters()).device

    def _ensure_text_encoder(self):
        """Initializes the text encoder on the correct device if it doesn't exist."""
        try:
            if self.text_encoder is None:
                device = self._get_device()
                self.text_encoder = PosNegTextEncoder(device=device)
                self.text_encoder.to(device)
                self.logger.info(f"Text encoder lazily initialized on device: {device}")
            else:
                current_device = self._get_device()
                if self.text_encoder.device != current_device:
                    self.text_encoder.to(current_device)
                    self.logger.info(f"Text encoder moved to device: {current_device}")
        except Exception as e:
            self.logger.error(f"Failed to initialize or move text encoder: {e}")
            raise DiTDeviceError(f"Text encoder device management failed: {e}")

    def to(self, *args, **kwargs):
        """Overrides `to()` to ensure the text encoder is also moved with error handling."""
        try:
            super().to(*args, **kwargs)
            if self.text_encoder is not None:
                self.text_encoder.to(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to move model to device: {e}")
            raise DiTDeviceError(f"Device movement failed: {e}")
        return self
    
    def _validate_device_consistency(self, *tensors: torch.Tensor) -> torch.device:
        """
        Validate that all tensors are on the same device as the model.
        
        Args:
            *tensors: Variable number of tensors to check
            
        Returns:
            The model's device
            
        Raises:
            DiTDeviceError: If device inconsistency is detected
        """
        model_device = self._get_device()
        
        for i, tensor in enumerate(tensors):
            if tensor is not None and isinstance(tensor, torch.Tensor):
                if tensor.device != model_device:
                    raise DiTDeviceError(
                        f"Tensor {i} is on device {tensor.device}, but model is on {model_device}. "
                        f"Please ensure all tensors are moved to the model's device before calling forward()."
                    )
        
        return model_device
    
    # --- Memory Optimization Methods ---
    
    def enable_gradient_checkpointing(self, enable: bool = True):
        """Enable or disable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing = enable
        
        # Update existing blocks
        for i, block in enumerate(self.dit_blocks):
            if isinstance(block, GradientCheckpointedDiTBlock):
                block.use_checkpointing = enable
            elif enable:
                # Wrap existing block with checkpointing
                self.dit_blocks[i] = GradientCheckpointedDiTBlock(block, use_checkpointing=True)
        
        self.logger.info(f"Gradient checkpointing {'enabled' if enable else 'disabled'}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        return self.memory_monitor.get_memory_info()
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage."""
        self.memory_monitor.log_memory_usage(stage)
    
    def suggest_batch_configuration(self, target_batch_size: int, sequence_length: int) -> Dict[str, Any]:
        """Suggest optimal batch configuration based on memory constraints."""
        return self.batch_processor.suggest_batch_configuration(
            target_batch_size, sequence_length, self.d_model, self.num_layers
        )
    
    def optimize_for_inference(self):
        """Apply optimizations specifically for inference."""
        # Disable gradient checkpointing during inference (not needed)
        if not self.training:
            self.enable_gradient_checkpointing(False)
        
        # Apply global memory optimizations
        optimize_memory_usage()
        
        self.logger.info("Applied inference-specific memory optimizations")
    
    def optimize_for_training(self):
        """Apply optimizations specifically for training."""
        # Enable gradient checkpointing for large models during training
        if self.training:
            memory_config = get_memory_optimization_config({
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'max_sequence_length': self.max_sequence_length
            })
            self.enable_gradient_checkpointing(memory_config['gradient_checkpointing'])
        
        self.logger.info("Applied training-specific memory optimizations")
    
    def estimate_memory_usage(self, batch_size: int, sequence_length: int) -> float:
        """Estimate memory usage for given batch configuration."""
        return self.batch_processor.estimate_memory_usage(
            batch_size, sequence_length, self.d_model, self.num_layers
        )
    
    def process_variable_length_batch(self, inputs: list, max_memory_gb: float = 8.0) -> list:
        """Process a batch with variable sequence lengths efficiently."""
        def model_forward_fn(batch_inputs):
            # Convert list of inputs to proper batch format
            batch_outputs = []
            for input_tensor in batch_inputs:
                # Assume inputs are already properly formatted
                output = self.forward(input_tensor['x_t'], input_tensor['ts'], input_tensor['data'])
                batch_outputs.append(output)
            return batch_outputs
        
        return self.batch_processor.process_variable_length_batch(
            model_forward_fn, inputs, max_memory_gb
        )