import torch
import torch.nn as nn
import math
import logging
from typing import Dict, Optional, Any, Tuple, Sequence
try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy should be available but guard anyway
    np = None
from einops import rearrange
from contextlib import nullcontext

from models.utils.diffusion_utils import timestep_embedding
from models.backbone import build_backbone
from .dit_utils import adjust_backbone_config, init_weights
from models.utils.text_encoder import TextConditionProcessor, PosNegTextEncoder
from .dit_conditioning import ensure_text_encoder, prepare_text_features
from .dit_config_validation import validate_dit_config, get_dit_config_summary
from .dit_validation import (
    validate_dit_inputs, DiTValidationError, DiTInputError, DiTDimensionError,
    DiTDeviceError, DiTConditioningError, DiTGracefulFallback
)
from .dit_memory_optimization import (
    MemoryMonitor, EfficientAttention, JointAttention, GradientCheckpointedDiTBlock,
    BatchProcessor, optimize_memory_usage, get_memory_optimization_config
)


def _convert_to_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Converts various container types (tensor/list/tuple/numpy array) to a torch.Tensor.
    Ensures the result resides on the requested device and dtype.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise DiTConditioningError(f"{name} list is empty")

        if all(isinstance(elem, torch.Tensor) for elem in value):
            return _pad_and_stack_tensors(value, device, dtype, name)

        if np is not None and all(isinstance(elem, np.ndarray) for elem in value):
            tensors = [torch.as_tensor(elem) for elem in value]
            return _pad_and_stack_tensors(tensors, device, dtype, name)

        try:
            return torch.as_tensor(value, dtype=dtype, device=device)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise DiTConditioningError(
                f"{name} list cannot be converted to tensor: {exc}"
            ) from exc

    if np is not None and isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device, dtype=dtype)

    raise DiTConditioningError(
        f"{name} must be a torch.Tensor, numpy.ndarray, or sequence convertible to tensor. "
        f"Got {type(value)}"
    )


def _pad_and_stack_tensors(
    tensors: Sequence[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """
    Pads variable-length tensors along the first dimension to enable stacking.

    Assumes all tensors share the same rank and feature dimensions (other than the first).
    """
    if len(tensors) == 0:
        raise DiTConditioningError(f"{name} tensor list is empty")

    ref_ndim = tensors[0].ndim
    if ref_ndim == 0:
        raise DiTConditioningError(f"{name} tensors must have at least one dimension")

    feature_shape = tensors[0].shape[1:]
    for idx, tensor in enumerate(tensors):
        if tensor.ndim != ref_ndim:
            raise DiTConditioningError(
                f"{name} tensors must have consistent rank; tensor {idx} has ndim {tensor.ndim}, "
                f"expected {ref_ndim}"
            )
        if tensor.shape[1:] != feature_shape:
            raise DiTConditioningError(
                f"{name} tensors must share feature dimensions; tensor {idx} has shape {tensor.shape}, "
                f"expected (*, {feature_shape})"
            )

    max_length = max(tensor.shape[0] for tensor in tensors)
    batch_size = len(tensors)

    padded = torch.zeros(
        (batch_size, max_length) + feature_shape,
        device=device,
        dtype=dtype
    )

    for idx, tensor in enumerate(tensors):
        tensor = tensor.to(device=device, dtype=dtype)
        length = tensor.shape[0]
        if length == 0:
            continue
        padded[idx, :length] = tensor

    return padded


def _normalize_object_mask(
    scene_points: torch.Tensor,
    object_mask: torch.Tensor,
    name: str = "object_mask",
) -> torch.Tensor:
    """
    Aligns an object mask to match the scene point cloud layout.

    Ensures the mask has shape (B, N, 1) where B is batch size and N is number of points.
    """
    if object_mask.numel() == 0:
        raise DiTConditioningError(f"{name} is empty while use_object_mask=True")

    mask = object_mask
    if mask.dim() == 1:
        if mask.shape[0] != scene_points.shape[1]:
            raise DiTConditioningError(
                f"{name} length {mask.shape[0]} does not match point count {scene_points.shape[1]}"
            )
        mask = mask.unsqueeze(0).expand(scene_points.shape[0], -1)
    elif mask.dim() == 2:
        if mask.shape[0] == scene_points.shape[0] and mask.shape[1] == scene_points.shape[1]:
            pass
        elif mask.shape[0] == scene_points.shape[1] and mask.shape[1] == scene_points.shape[0]:
            mask = mask.t()
        else:
            raise DiTConditioningError(
                f"{name} shape {mask.shape} is incompatible with scene points "
                f"{(scene_points.shape[0], scene_points.shape[1])}"
            )
    elif mask.dim() == 3:
        if (
            mask.shape[0] != scene_points.shape[0]
            or mask.shape[1] != scene_points.shape[1]
            or mask.shape[2] != 1
        ):
            raise DiTConditioningError(
                f"{name} shape {mask.shape} must match (batch, num_points, 1)"
            )
    else:
        raise DiTConditioningError(f"{name} must have 1, 2, or 3 dimensions, got {mask.dim()}")

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    mask = mask.to(device=scene_points.device, dtype=torch.float32)

    if torch.all(mask == 0):
        raise DiTConditioningError(f"{name} contains only zeros while use_object_mask=True")

    mask = torch.clamp(mask, 0.0, 1.0)
    return mask


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
    Enhanced with memory optimization features and mask support.
    Supports optional MMDiT-style joint attention for multi-modal conditioning.
    """
    def __init__(self, d_model: int, num_heads: int, d_head: int, dropout: float = 0.1,
                 use_adaptive_norm: bool = True, time_embed_dim: Optional[int] = None,
                 chunk_size: int = 512, use_flash_attention: bool = False,
                 attention_dropout: float = 0.0, cross_attention_dropout: float = 0.0,
                 mmdit_cfg: Optional[Dict] = None):
        super().__init__()
        self.d_model = d_model
        self.use_adaptive_norm = use_adaptive_norm
        self.logger = logging.getLogger(__name__)

        if mmdit_cfg is not None and hasattr(mmdit_cfg, "items"):
            mmdit_cfg = {k: mmdit_cfg[k] for k in mmdit_cfg}
        elif mmdit_cfg is None:
            mmdit_cfg = {}
        self.mmdit_cfg = mmdit_cfg
        self.mmdit_enabled = bool(self.mmdit_cfg.get('enabled', False))
        self.mmdit_modality_adaln = bool(self.mmdit_cfg.get('modality_specific_adaln', False))
        if self.mmdit_enabled:
            self.logger.info("[DiTBlock] MMDiT mode enabled with config: %s", self.mmdit_cfg)
            if self.mmdit_modality_adaln:
                self.logger.info("[DiTBlock] MMDiT modality-specific AdaLN active")

        # Memory-efficient attention layers with dropout support
        self.self_attention = EfficientAttention(
            d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
            attention_dropout=attention_dropout
        )

        if self.mmdit_enabled:
            # Use JointAttention for combined scene+text conditioning
            self.joint_attention = JointAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_head=d_head,
                dropout=dropout,
                chunk_size=chunk_size,
                use_flash_attention=use_flash_attention,
                attention_dropout=cross_attention_dropout,
                separate_modality_projections=self.mmdit_cfg.get('separate_modality_projections', True),
                qkv_bias=self.mmdit_cfg.get('qkv_bias', False),
                qk_norm=self.mmdit_cfg.get('qk_norm', False),
                logger=self.logger,
            )
            self.scene_cross_attention = None
            self.text_cross_attention = None
            self.logger.info("[DiTBlock] Initialized JointAttention with separate_proj=%s qkv_bias=%s qk_norm=%s",
                             self.mmdit_cfg.get('separate_modality_projections', True),
                             self.mmdit_cfg.get('qkv_bias', False),
                             self.mmdit_cfg.get('qk_norm', False))
        else:
            # Legacy: separate scene and text cross-attention
            self.scene_cross_attention = EfficientAttention(
                d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
                attention_dropout=cross_attention_dropout
            )
            self.text_cross_attention = EfficientAttention(
                d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
                attention_dropout=cross_attention_dropout
            )
            self.joint_attention = None

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
                text_context: Optional[torch.Tensor] = None,
                scene_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, num_grasps, d_model)
            time_emb: (B, time_embed_dim) or None
            scene_context: (B, N_points, d_model) or None
            text_context: (B, 1, d_model) or None
            scene_mask: (B, N_points) or (B, 1, N_points) - mask for scene padding, 1=valid, 0=padding
        Returns:
            output: (B, num_grasps, d_model)
        """
        # Self-attention among grasps
        if self.use_adaptive_norm and time_emb is not None:
            norm_x = self.norm1(x, time_emb)
        else:
            norm_x = self.norm1(x)

        if torch.isnan(norm_x).any():
            logging.error(f"[DiTBlock NaN] NaN after norm1")
            logging.error(f"  Input x stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
            if time_emb is not None:
                logging.error(f"  time_emb stats: min={time_emb.min():.6f}, max={time_emb.max():.6f}, mean={time_emb.mean():.6f}")
            raise RuntimeError("NaN detected in DiTBlock after norm1")

        attn_out = self.self_attention(norm_x)
        if torch.isnan(attn_out).any():
            logging.error(f"[DiTBlock NaN] NaN after self_attention")
            logging.error(f"  norm_x stats: min={norm_x.min():.6f}, max={norm_x.max():.6f}")
            raise RuntimeError("NaN detected in DiTBlock self_attention")

        x = x + attn_out

        # Conditional attention: MMDiT joint attention vs. legacy separate cross-attentions
        if self.mmdit_enabled:
            # MMDiT path: single joint attention over grasp+scene+text
            if self.use_adaptive_norm and time_emb is not None:
                norm_x = self.norm2(x, time_emb)
            else:
                norm_x = self.norm2(x)

            if torch.isnan(norm_x).any():
                logging.error(f"[DiTBlock MMDiT NaN] NaN after norm2 (pre-joint-attn)")
                raise RuntimeError("NaN detected in DiTBlock MMDiT pre-joint-attn")

            # Prepare text context shape: ensure (B, T, d_model)
            text_tokens = None
            if text_context is not None:
                if text_context.dim() == 2:
                    text_tokens = text_context.unsqueeze(1)
                else:
                    text_tokens = text_context

            joint_out = self.joint_attention(
                grasp_tokens=norm_x,
                scene_tokens=scene_context,
                text_tokens=text_tokens,
                scene_mask=scene_mask
            )

            if torch.isnan(joint_out).any():
                logging.error(f"[DiTBlock MMDiT NaN] NaN after joint_attention")
                logging.error(f"  norm_x stats: min={norm_x.min():.6f}, max={norm_x.max():.6f}")
                if scene_context is not None:
                    logging.error(f"  scene_context stats: min={scene_context.min():.6f}, max={scene_context.max():.6f}")
                if text_tokens is not None:
                    logging.error(f"  text_context stats: min={text_tokens.min():.6f}, max={text_tokens.max():.6f}")
                raise RuntimeError("NaN detected in DiTBlock MMDiT joint_attention")

            x = x + joint_out

            if self.mmdit_modality_adaln:
                if self.use_adaptive_norm and time_emb is not None:
                    mod_x = self.norm3(x, time_emb)
                else:
                    mod_x = self.norm3(x)
                if torch.isnan(mod_x).any():
                    logging.error(f"[DiTBlock MMDiT NaN] NaN after modality-specific AdaLN")
                    raise RuntimeError("NaN detected in DiTBlock MMDiT modality-specific AdaLN")
                x = mod_x

        else:
            # Legacy path: separate scene and text cross-attention
            if scene_context is not None:
                if self.use_adaptive_norm and time_emb is not None:
                    norm_x = self.norm2(x, time_emb)
                else:
                    norm_x = self.norm2(x)

                if torch.isnan(norm_x).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm2")
                    raise RuntimeError("NaN detected in DiTBlock after norm2")

                # Pass scene_mask to prevent attention to padding positions
                scene_attn_out = self.scene_cross_attention(norm_x, scene_context, scene_context, mask=scene_mask)
                if torch.isnan(scene_attn_out).any():
                    logging.error(f"[DiTBlock NaN] NaN after scene_cross_attention")
                    logging.error(f"  norm_x stats: min={norm_x.min():.6f}, max={norm_x.max():.6f}")
                    logging.error(f"  scene_context stats: min={scene_context.min():.6f}, max={scene_context.max():.6f}")
                    raise RuntimeError("NaN detected in DiTBlock scene_cross_attention")

                x = x + scene_attn_out

            # Cross-attention with text features (if available)
            if text_context is not None:
                if self.use_adaptive_norm and time_emb is not None:
                    norm_x = self.norm3(x, time_emb)
                else:
                    norm_x = self.norm3(x)

                if torch.isnan(norm_x).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm3")
                    raise RuntimeError("NaN detected in DiTBlock after norm3")

                text_attn_out = self.text_cross_attention(norm_x, text_context, text_context)
                if torch.isnan(text_attn_out).any():
                    logging.error(f"[DiTBlock NaN] NaN after text_cross_attention")
                    logging.error(f"  norm_x stats: min={norm_x.min():.6f}, max={norm_x.max():.6f}")
                    logging.error(f"  text_context stats: min={text_context.min():.6f}, max={text_context.max():.6f}")
                    raise RuntimeError("NaN detected in DiTBlock text_cross_attention")

                x = x + text_attn_out

        # Feed-forward network
        if self.use_adaptive_norm and time_emb is not None:
            norm_x = self.norm4(x, time_emb)
        else:
            norm_x = self.norm4(x)

        if torch.isnan(norm_x).any():
            logging.error(f"[DiTBlock NaN] NaN after norm4")
            raise RuntimeError("NaN detected in DiTBlock after norm4")

        ff_out = self.feed_forward(norm_x)
        if torch.isnan(ff_out).any():
            logging.error(f"[DiTBlock NaN] NaN after feed_forward")
            logging.error(f"  norm_x stats: min={norm_x.min():.6f}, max={norm_x.max():.6f}")
            raise RuntimeError("NaN detected in DiTBlock feed_forward")

        x = x + ff_out

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

        # Attention dropout config (for regularization)
        self.attention_dropout = getattr(cfg, 'attention_dropout', 0.0)
        self.cross_attention_dropout = getattr(cfg, 'cross_attention_dropout', 0.0)
        
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
        
        # MMDiT configuration (optional)
        raw_mmdit_cfg = getattr(cfg, 'mmdit', None)
        if raw_mmdit_cfg is not None and hasattr(raw_mmdit_cfg, 'items'):
            self.mmdit_cfg = {k: raw_mmdit_cfg[k] for k in raw_mmdit_cfg}
        elif raw_mmdit_cfg is None:
            self.mmdit_cfg = {}
        else:
            self.mmdit_cfg = dict(raw_mmdit_cfg)
        self.mmdit_enabled = bool(self.mmdit_cfg.get('enabled', False))
        if self.mmdit_enabled:
            self.logger.info("MMDiT joint attention enabled with config: %s", self.mmdit_cfg)
        else:
            self.logger.info("MMDiT joint attention disabled; using legacy cross-attention path")
        
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
                use_flash_attention=self.use_flash_attention,
                attention_dropout=self.attention_dropout,
                cross_attention_dropout=self.cross_attention_dropout,
                mmdit_cfg=self.mmdit_cfg.copy()
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
        
        # Initialize weights
        init_weights(self)
    
    
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
        context_mgr = (
            self.memory_monitor.monitor_peak_memory("DiT_forward_pass")
            if self.memory_monitoring else nullcontext()
        )
        with context_mgr:
            try:
                if self.memory_monitoring:
                    memory_status = self.memory_monitor.check_memory_pressure()
                    if memory_status['under_pressure']:
                        for hint in memory_status['optimization_hints']:
                            self.logger.warning(f"Memory optimization hint: {hint}")

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

        x_t, single_grasp_mode = self._normalize_input_shape(x_t)
        grasp_tokens = self._tokenize_grasps(x_t)
        time_emb = self._embed_timesteps(ts)
        scene_context, text_context = self._resolve_condition_features(data, model_device)
        
        # Extract scene_mask if available
        scene_mask = data.get("scene_mask", None)
        if scene_mask is not None:
            if not isinstance(scene_mask, torch.Tensor):
                self.logger.warning(f"scene_mask is not a tensor ({type(scene_mask)}), ignoring")
                scene_mask = None
            elif scene_mask.device != model_device:
                scene_mask = scene_mask.to(model_device)

        x = self._run_dit_blocks(
            grasp_tokens,
            time_emb if self.use_adaptive_norm else None,
            scene_context,
            text_context,
            scene_mask=scene_mask
        )

        output = self._finalize_output(x, single_grasp_mode)
        self._assert_finite(output, "DiT output")
        return output

    def condition(self, data: Dict) -> Dict:
        """
        Pre-computes and processes all conditional features (scene, text).
        Reuses UNet's conditioning logic for compatibility with enhanced error handling.
        """
        if data is None:
            raise DiTConditioningError("Conditioning data is None")
        if not isinstance(data, dict):
            raise DiTConditioningError(f"Conditioning data must be dict, got {type(data)}")

        try:
            scene_feat = self._prepare_scene_features(data)
        except Exception as exc:
            self.logger.error(f"Scene feature extraction failed: {exc}")
            scene_feat = self._build_fallback_scene_features(data)

        condition_dict = {"scene_cond": scene_feat, "text_cond": None, "text_mask": None}
        if self.use_negative_prompts:
            condition_dict.update({"neg_pred": None, "neg_text_features": None})

        if not (self.use_text_condition and 'positive_prompt' in data):
            return condition_dict

        try:
            text_features = self._prepare_text_features(data, scene_feat)
            condition_dict.update(text_features)
        except Exception as exc:
            self.logger.warning(f"Text encoding failed: {exc}. Falling back to scene-only conditioning.")
            condition_dict.update({
                "text_cond": None,
                "text_mask": None,
                "neg_pred": None,
                "neg_text_features": None
            })

        return condition_dict

    def _normalize_input_shape(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x_t.dim() == 2:
            return x_t.unsqueeze(1), True
        if x_t.dim() == 3:
            return x_t, False
        raise DiTDimensionError(f"Unsupported input dimension: {x_t.dim()}. Expected 2 or 3.")

    def _tokenize_grasps(self, x_t: torch.Tensor) -> torch.Tensor:
        grasp_tokens = self.grasp_tokenizer(x_t)
        self._assert_finite(grasp_tokens, "grasp_tokens")
        if self.pos_embedding is not None:
            grasp_tokens = self.pos_embedding(grasp_tokens)
            self._assert_finite(grasp_tokens, "grasp_tokens (positional)")
        return grasp_tokens

    def _embed_timesteps(self, ts: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(ts)
        self._assert_finite(time_emb, "time_embedding")
        return time_emb

    def _resolve_condition_features(
        self,
        data: Dict,
        model_device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        scene_context = data.get("scene_cond")
        if scene_context is not None:
            if not isinstance(scene_context, torch.Tensor):
                raise DiTConditioningError(f"scene_cond must be torch.Tensor, got {type(scene_context)}")
            if scene_context.device != model_device:
                scene_context = scene_context.to(model_device)
                self.logger.debug(f"Moved scene_cond to device {model_device}")
            self._assert_finite(scene_context, "scene_context")

        text_context = None
        if self.use_text_condition:
            raw_text_context = data.get("text_cond")
            if raw_text_context is not None:
                if not isinstance(raw_text_context, torch.Tensor):
                    self.logger.warning(
                        f"text_cond is not a tensor ({type(raw_text_context)}), disabling text conditioning"
                    )
                else:
                    if raw_text_context.device != model_device:
                        raw_text_context = raw_text_context.to(model_device)
                        self.logger.debug(f"Moved text_cond to device {model_device}")
                    self._assert_finite(raw_text_context, "text_context")
                    text_context = raw_text_context.unsqueeze(1)
        return scene_context, text_context

    def _run_dit_blocks(
        self,
        tokens: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        scene_context: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor],
        scene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = tokens
        for idx, block in enumerate(self.dit_blocks):
            try:
                self.logger.debug(
                    f"[DiT Block {idx}/{self.num_layers}] "
                    f"Input stats: shape={tuple(x.shape)}, min={x.min():.6f}, max={x.max():.6f}, "
                    f"mean={x.mean():.6f}, std={x.std():.6f}"
                )
                x = block(
                    x=x,
                    time_emb=time_emb,
                    scene_context=scene_context,
                    text_context=text_context,
                    scene_mask=scene_mask
                )
                self._assert_finite(x, f"DiT block {idx} output")
            except DiTConditioningError:
                raise
            except Exception as exc:
                self.logger.error(f"[DiT Block {idx}] Error in block processing: {exc}")
                raise DiTConditioningError(f"Error in DiT block {idx}: {exc}") from exc
        return x

    def _finalize_output(self, x: torch.Tensor, single_grasp_mode: bool) -> torch.Tensor:
        output = self.output_projection(x)
        if single_grasp_mode:
            output = output.squeeze(1)
        return output

    def _prepare_scene_features(self, data: Dict) -> torch.Tensor:
        if 'scene_pc' not in data or data['scene_pc'] is None:
            raise DiTConditioningError("Missing scene_pc in conditioning data")

        model_device = self._get_device()
        scene_pc = _convert_to_tensor(
            data['scene_pc'],
            device=model_device,
            dtype=torch.float32,
            name="scene_pc"
        )
        self.logger.debug(
            f"[Conditioning] scene_pc input: shape={tuple(scene_pc.shape)}, "
            f"min={scene_pc.min():.6f}, max={scene_pc.max():.6f}"
        )

        if not self.use_rgb:
            scene_pc = scene_pc[..., :3]
            self.logger.debug(f"[Conditioning] Removed RGB, scene_pc shape: {tuple(scene_pc.shape)}")

        if self.use_object_mask and 'object_mask' in data and data['object_mask'] is not None:
            object_mask = _convert_to_tensor(
                data['object_mask'],
                device=model_device,
                dtype=torch.float32,
                name="object_mask"
            )
            object_mask = _normalize_object_mask(scene_pc, object_mask)
            scene_points = torch.cat([scene_pc, object_mask], dim=-1)
            self.logger.debug(f"[Conditioning] Added object_mask, pos shape: {tuple(scene_points.shape)}")
        else:
            scene_points = scene_pc

        if scene_points.dim() != 3:
            raise DiTConditioningError(f"Scene point cloud must be 3D tensor, got {scene_points.dim()}D")

        self.logger.debug(
            f"[Conditioning] Final pos before backbone: shape={tuple(scene_points.shape)}, "
            f"min={scene_points.min():.6f}, max={scene_points.max():.6f}, mean={scene_points.mean():.6f}"
        )

        _, scene_feat = self.scene_model(scene_points)
        self._assert_finite(scene_feat, "scene_feat (backbone output)")
        scene_feat = scene_feat.permute(0, 2, 1).contiguous()
        scene_feat = self._replace_non_finite(scene_feat, "scene_feat (permuted)")
        scene_feat = self.scene_projection(scene_feat)
        self._assert_finite(scene_feat, "scene_feat (projected)")
        return scene_feat

    def _build_fallback_scene_features(self, data: Dict) -> torch.Tensor:
        batch_size = 1
        if 'scene_pc' in data:
            scene_pc = data['scene_pc']
            if isinstance(scene_pc, torch.Tensor):
                batch_size = scene_pc.shape[0]
            elif isinstance(scene_pc, (list, tuple)):
                batch_size = len(scene_pc)
            elif np is not None and isinstance(scene_pc, np.ndarray):
                batch_size = scene_pc.shape[0]
        model_device = self._get_device()
        scene_feat_raw = torch.zeros(batch_size, 1024, 512, device=model_device)
        self.logger.warning("Using fallback scene features due to extraction failure")
        return self.scene_projection(scene_feat_raw)

    def _prepare_text_features(self, data: Dict, scene_feat: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self._ensure_text_encoder()
        batch_size = scene_feat.shape[0]
        model_device = self._get_device()

        result = prepare_text_features(
            text_encoder=self.text_encoder,
            text_processor=self.text_processor,
            data=data,
            scene_feat=scene_feat,
            use_negative_prompts=self.use_negative_prompts,
            text_dropout_prob=self.text_dropout_prob,
            training=self.training,
            device=model_device,
            logger=self.logger,
        )
        return result

    def _replace_non_finite(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(tensor).any():
            self.logger.warning(f"{name} contains NaN values, replacing with zeros")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        if torch.isinf(tensor).any():
            self.logger.warning(f"{name} contains infinite values, clamping to [-1e6, 1e6]")
            tensor = torch.clamp(tensor, -1e6, 1e6)
        return tensor

    def _assert_finite(self, tensor: Optional[torch.Tensor], name: str) -> None:
        if tensor is None or tensor.numel() == 0:
            return
        if torch.isnan(tensor).any():
            self._log_tensor_stats(tensor, name)
            raise DiTConditioningError(f"NaN detected in {name}")
        if torch.isinf(tensor).any():
            self._log_tensor_stats(tensor, name)
            raise DiTConditioningError(f"Infinite value detected in {name}")

    def _has_non_finite(self, tensor: torch.Tensor) -> bool:
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    def _log_tensor_stats(self, tensor: torch.Tensor, name: str) -> None:
        finite_tensor = tensor[torch.isfinite(tensor)]
        if finite_tensor.numel() == 0:
            self.logger.error(f"[{name}] tensor is entirely non-finite")
            return
        self.logger.error(
            f"[{name}] stats: shape={tuple(tensor.shape)}, "
            f"min={finite_tensor.min():.6f}, max={finite_tensor.max():.6f}, "
            f"mean={finite_tensor.mean():.6f}"
        )
    
    # --- Device Management for Lazy-Loaded Encoder ---

    def _get_device(self):
        """Infers the model's device from its parameters."""
        return next(self.parameters()).device

    def _ensure_text_encoder(self):
        """Initializes the text encoder on the correct device if it doesn't exist."""
        try:
            device = self._get_device()
            self.text_encoder = ensure_text_encoder(self.text_encoder, device, logger=self.logger)
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
