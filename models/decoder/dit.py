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
from .dit_conditioning import ensure_text_encoder, prepare_text_features, pool_scene_features
from .dit_config_validation import validate_dit_config, get_dit_config_summary
from .dit_validation import (
    validate_dit_inputs, DiTValidationError, DiTInputError, DiTDimensionError,
    DiTDeviceError, DiTConditioningError, DiTGracefulFallback
)
from .dit_memory_optimization import (
    MemoryMonitor, EfficientAttention, GradientCheckpointedDiTBlock,
    BatchProcessor, optimize_memory_usage, get_memory_optimization_config
)
from .scene_pool import GlobalScenePool
from .local_selector import build_local_selector
from .adaln_cond import build_adaln_cond_vector


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


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization and gating (AdaLN-Zero).
    
    按 DiT 论文实现，支持多条件融合（时间步+场景+文本）。
    通过零初始化使模型训练起始时近似恒等映射，提高训练稳定性。
    
    公式：x + gate * ((1 + scale) * LayerNorm(x) + shift)
    其中 scale, shift, gate 均由条件向量 c 通过 MLP 生成。
    """
    def __init__(self, d_model: int, cond_dim: int):
        """
        Args:
            d_model: 特征维度
            cond_dim: 条件向量维度（可以是 time_dim + scene_dim + text_dim 的拼接）
        """
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        
        # LayerNorm without learnable affine parameters
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        
        # 生成 scale, shift, gate 三个参数（每个都是 d_model 维）
        self.modulation = nn.Linear(cond_dim, d_model * 3)
        
        # 零初始化：使模型起始时近似恒等映射，训练稳定
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, seq_len, d_model) - 输入特征
            cond: (B, cond_dim) - 融合的条件向量
        Returns:
            x_modulated: (B, seq_len, d_model) - 用于子层输入的调制特征
            gate: (B, 1, d_model) - 用于对子层输出进行门控的系数
        """
        # 生成调制参数
        modulation_params = self.modulation(cond)  # (B, d_model * 3)
        scale, shift, gate = modulation_params.chunk(3, dim=-1)  # 每个 (B, d_model)

        # 扩展维度以匹配序列长度
        scale = scale.unsqueeze(1)  # (B, 1, d_model)
        shift = shift.unsqueeze(1)  # (B, 1, d_model)
        gate = gate.unsqueeze(1)    # (B, 1, d_model)

        # Pre-LN 调制：x_mod = (1 + scale) * LN(x) + shift
        norm_x = self.layer_norm(x)
        x_modulated = (1 + scale) * norm_x + shift

        # 记录调制统计，便于训练时定位数值问题
        try:
            with torch.no_grad():
                # cond: (B, cond_dim) 可能较大，仅记录基本统计
                c_isfinite = torch.isfinite(cond).all().item() if cond.numel() > 0 else True
                c_min = float(torch.nan_to_num(cond, nan=0.0).min()) if cond.numel() > 0 else 0.0
                c_max = float(torch.nan_to_num(cond, nan=0.0).max()) if cond.numel() > 0 else 0.0
                c_mean = float(torch.nan_to_num(cond, nan=0.0).mean()) if cond.numel() > 0 else 0.0
                c_std = float(torch.nan_to_num(cond, nan=0.0).std()) if cond.numel() > 1 else 0.0
                g_min = float(gate.min()) if gate.numel() > 0 else 0.0
                g_max = float(gate.max()) if gate.numel() > 0 else 0.0
                g_mean = float(gate.mean()) if gate.numel() > 0 else 0.0
                g_std = float(gate.std()) if gate.numel() > 1 else 0.0
                xm_isfinite = torch.isfinite(x_modulated).all().item() if x_modulated.numel() > 0 else True
                xm_mean = float(torch.nan_to_num(x_modulated, nan=0.0).mean()) if x_modulated.numel() > 0 else 0.0
                xm_std = float(torch.nan_to_num(x_modulated, nan=0.0).std()) if x_modulated.numel() > 1 else 0.0
                self._last_stats = {
                    'cond_isfinite': bool(c_isfinite),
                    'cond_min': c_min, 'cond_max': c_max, 'cond_mean': c_mean, 'cond_std': c_std,
                    'gate_min': g_min, 'gate_max': g_max, 'gate_mean': g_mean, 'gate_std': g_std,
                    'xmod_isfinite': bool(xm_isfinite), 'xmod_mean': xm_mean, 'xmod_std': xm_std,
                }
        except Exception:
            # 仅做监控，任何异常都不影响前向
            self._last_stats = getattr(self, '_last_stats', None)

        # 注意：残差与门控在子层输出处应用，而不是此处
        return x_modulated, gate


class _AdaLNZeroContext:
    def __init__(
        self,
        layer_norms: nn.ModuleList,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ):
        self.layer_norms = layer_norms
        self.scale = scale
        self.shift = shift
        self.gate = gate

    def apply(self, idx: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.layer_norms[idx](x)
        scale = self.scale[:, idx].unsqueeze(1)
        shift = self.shift[:, idx].unsqueeze(1)
        gate = self.gate[:, idx].unsqueeze(1)
        x_modulated = (1 + scale) * norm_x + shift
        return x_modulated, gate


class AdaLNZeroStack(nn.Module):
    """
    扩展的 AdaLN-Zero，实现一次性生成多个子层的调制参数，减少冗余权重。
    """

    def __init__(self, d_model: int, cond_dim: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(d_model, elementwise_affine=False) for _ in range(num_layers)
        )
        self.modulation = nn.Linear(cond_dim, d_model * 3 * num_layers)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self._last_stats: Optional[Dict[str, Any]] = None

    def prepare(self, cond: torch.Tensor) -> _AdaLNZeroContext:
        params = self.modulation(cond)  # (B, 3 * num_layers * d_model)
        batch_size = cond.shape[0]
        params = params.view(batch_size, self.num_layers, 3, self.d_model)
        scale = params[:, :, 0, :]
        shift = params[:, :, 1, :]
        gate = params[:, :, 2, :]

        try:
            with torch.no_grad():
                c_isfinite = torch.isfinite(cond).all().item()
                c_min = float(torch.nan_to_num(cond, nan=0.0).min())
                c_max = float(torch.nan_to_num(cond, nan=0.0).max())
                c_mean = float(torch.nan_to_num(cond, nan=0.0).mean())
                c_std = float(torch.nan_to_num(cond, nan=0.0).std())
                g_min = float(gate.min())
                g_max = float(gate.max())
                g_mean = float(gate.mean())
                g_std = float(gate.std())
                self._last_stats = {
                    'cond_isfinite': bool(c_isfinite),
                    'cond_min': c_min,
                    'cond_max': c_max,
                    'cond_mean': c_mean,
                    'cond_std': c_std,
                    'gate_min': g_min,
                    'gate_max': g_max,
                    'gate_mean': g_mean,
                    'gate_std': g_std,
                }
        except Exception:
            self._last_stats = getattr(self, '_last_stats', None)

        return _AdaLNZeroContext(self.layer_norms, scale, shift, gate)


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


class DiTDoubleStreamBlock(nn.Module):
    """
    双流 DiT Block：同时更新抓取 token 与场景 token，通过共享注意力实现信息交互。
    
    参考 Hunyuan3D-DiT 的 DoubleStreamBlock 实现：
    - 分别计算两个流的 QKV
    - 拼接 QKV 后进行联合注意力
    - 分离注意力输出并分别投影

    该模块假设启用了 AdaLN-Zero 调制。
    """

    is_double_stream = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        dropout: float,
        cond_dim: Optional[int],
        chunk_size: int,
        use_flash_attention: bool,
        attention_dropout: float = 0.0,
        debug_check_nan: bool = False,
        debug_log_stats: bool = False,
    ):
        super().__init__()
        if cond_dim is None:
            raise ValueError("DiTDoubleStreamBlock requires cond_dim when use_adaln_zero=True.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.chunk_size = chunk_size
        self.debug_check_nan = debug_check_nan
        self.debug_log_stats = debug_log_stats

        # 每个流独立的调制模块（每个流有2层：attn + mlp）
        self.grasp_mod = AdaLNZeroStack(d_model, cond_dim, num_layers=2)
        self.scene_mod = AdaLNZeroStack(d_model, cond_dim, num_layers=2)

        # 每个流独立的 QKV 投影和输出投影
        self.grasp_qkv = nn.Linear(d_model, d_model * 3)
        self.scene_qkv = nn.Linear(d_model, d_model * 3)
        
        # QK 归一化（参考 Hunyuan3D-DiT）
        from .dit_utils import QKNorm
        self.qk_norm = QKNorm(d_head)
        
        # 输出投影
        self.grasp_proj = nn.Linear(d_model, d_model)
        self.scene_proj = nn.Linear(d_model, d_model)
        
        # MLP
        self.grasp_mlp = FeedForward(d_model, dropout=dropout)
        self.scene_mlp = FeedForward(d_model, dropout=dropout)

    def _log_stats(self, name: str, tensor: torch.Tensor):
        if not self.debug_log_stats or tensor is None:
            return
        try:
            logging.getLogger(__name__).debug(
                f"[DoubleStreamBlock] {name}: shape={tuple(tensor.shape)} min={tensor.min():.6f} max={tensor.max():.6f} mean={tensor.mean():.6f} std={(tensor.std() if tensor.numel()>1 else torch.tensor(0.)).item():.6f}"
            )
        except Exception:
            pass

    def _check_finite(self, name: str, tensor: torch.Tensor):
        if not self.debug_check_nan or tensor is None:
            return
        if not torch.isfinite(tensor).all():
            logging.getLogger(__name__).error(f"[DoubleStreamBlock] Non-finite values at {name}")
            raise RuntimeError(f"NaN/Inf detected in DoubleStreamBlock at {name}")

    def forward(
        self,
        grasp_tokens: torch.Tensor,
        scene_tokens: torch.Tensor,
        cond_vector: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with joint attention mechanism (Hunyuan3D-DiT style).
        
        Args:
            grasp_tokens: (B, L_grasp, D)
            scene_tokens: (B, L_scene, D)
            cond_vector: (B, cond_dim)
            scene_mask: (B, L_scene) - 1 for valid, 0 for padding
        Returns:
            grasp_tokens: (B, L_grasp, D)
            scene_tokens: (B, L_scene, D)
        """
        if cond_vector is None:
            raise DiTConditioningError("cond_vector is required for double-stream blocks with AdaLN-Zero.")
        if scene_tokens is None:
            raise DiTConditioningError("Double-stream blocks require scene tokens.")

        B, num_grasps, _ = grasp_tokens.shape
        _, num_scene, _ = scene_tokens.shape

        # 准备调制上下文
        grasp_ctx = self.grasp_mod.prepare(cond_vector)
        scene_ctx = self.scene_mod.prepare(cond_vector)

        # ============= Attention 部分 =============
        # 1. 调制和归一化
        grasp_attn_in, grasp_gate_attn = grasp_ctx.apply(0, grasp_tokens)
        scene_attn_in, scene_gate_attn = scene_ctx.apply(0, scene_tokens)
        self._check_finite("grasp_attn_in", grasp_attn_in)
        self._check_finite("scene_attn_in", scene_attn_in)
        self._log_stats("grasp_attn_in", grasp_attn_in)
        self._log_stats("scene_attn_in", scene_attn_in)

        # 2. 分别计算 QKV（关键修复点）
        grasp_qkv = self.grasp_qkv(grasp_attn_in)  # (B, L_grasp, 3*D)
        scene_qkv = self.scene_qkv(scene_attn_in)  # (B, L_scene, 3*D)
        
        # 3. Reshape 为 Q, K, V
        # rearrange 输出形状: (K, B, H, L, D)，取第0维得到 q, k, v
        grasp_q, grasp_k, grasp_v = rearrange(
            grasp_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        scene_q, scene_k, scene_v = rearrange(
            scene_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # 每个 q, k, v 的形状: (B, H, L, D)
        
        # 4. QK 归一化
        grasp_q, grasp_k = self.qk_norm(grasp_q, grasp_k, grasp_v)
        scene_q, scene_k = self.qk_norm(scene_q, scene_k, scene_v)
        
        # 5. 拼接 Q, K, V 进行联合注意力（关键：先场景后抓取）
        # 在序列长度维度拼接: dim=2 (B, H, L, D) -> (B, H, L_scene+L_grasp, D)
        q = torch.cat([scene_q, grasp_q], dim=2)
        k = torch.cat([scene_k, grasp_k], dim=2)
        v = torch.cat([scene_v, grasp_v], dim=2)
        
        # 6. 构建 attention mask
        if scene_mask is not None:
            if scene_mask.dim() == 3:
                scene_mask = scene_mask.squeeze(1)
            scene_mask = scene_mask.to(grasp_tokens.device, dtype=torch.float32)
            scene_mask = torch.clamp(scene_mask, 0.0, 1.0)
            # grasp tokens 没有 mask（全部有效）
            grasp_mask = torch.ones(B, num_grasps, device=scene_mask.device, dtype=scene_mask.dtype)
            # 拼接 mask（先场景后抓取）
            combined_mask = torch.cat([scene_mask, grasp_mask], dim=1)  # (B, L_scene + L_grasp)
            # 转换为 attention mask: (B, 1, 1, L) for broadcasting
            attn_mask = combined_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_mask = (1.0 - attn_mask) * -10000.0
        else:
            attn_mask = None
        
        # 7. 联合注意力计算
        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        attn_out = rearrange(attn_out, "B H L D -> B L (H D)")
        
        # 8. 分离注意力输出（先场景后抓取）
        scene_attn_out, grasp_attn_out = attn_out.split([num_scene, num_grasps], dim=1)
        self._check_finite("grasp_attn_out", grasp_attn_out)
        self._check_finite("scene_attn_out", scene_attn_out)
        self._log_stats("grasp_attn_out", grasp_attn_out)
        self._log_stats("scene_attn_out", scene_attn_out)
        
        # 9. 投影并应用门控残差
        grasp_tokens = grasp_tokens + grasp_gate_attn * self.grasp_proj(grasp_attn_out)
        scene_tokens = scene_tokens + scene_gate_attn * self.scene_proj(scene_attn_out)

        # ============= MLP 部分 =============
        # 10. 调制和 MLP
        grasp_ff_in, grasp_gate_ff = grasp_ctx.apply(1, grasp_tokens)
        scene_ff_in, scene_gate_ff = scene_ctx.apply(1, scene_tokens)

        grasp_ff_out = self.grasp_mlp(grasp_ff_in)
        scene_ff_out = self.scene_mlp(scene_ff_in)
        self._check_finite("grasp_ff_out", grasp_ff_out)
        self._check_finite("scene_ff_out", scene_ff_out)
        self._log_stats("grasp_ff_out", grasp_ff_out)
        self._log_stats("scene_ff_out", scene_ff_out)

        # 11. 门控残差
        grasp_tokens = grasp_tokens + grasp_gate_ff * grasp_ff_out
        scene_tokens = scene_tokens + scene_gate_ff * scene_ff_out

        return grasp_tokens, scene_tokens


class DiTSingleParallelBlock(nn.Module):
    """
    单流 DiT Block 并联结构：自注意力与 MLP 并联计算后通过同一残差路径。

    要求使用 AdaLN-Zero 调制。
    """

    is_double_stream = False

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        dropout: float,
        use_adaln_zero: bool,
        cond_dim: Optional[int],
        chunk_size: int,
        use_flash_attention: bool,
        attention_dropout: float,
        cross_attention_dropout: float,
        use_geometric_bias: bool,
        geometric_bias_module: Optional[nn.Module],
        time_gate: Optional[Any],
    ):
        super().__init__()
        if not use_adaln_zero:
            raise ValueError("DiTSingleParallelBlock requires use_adaln_zero=True")
        if cond_dim is None:
            raise ValueError("DiTSingleParallelBlock requires cond_dim when use_adaln_zero=True.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.use_geometric_bias = use_geometric_bias
        self.geometric_bias_module = geometric_bias_module
        self.time_gate = time_gate
        self.attn_dropout = attention_dropout
        self.parallel_dropout = nn.Dropout(dropout)

        self.mod_stack = AdaLNZeroStack(d_model, cond_dim, num_layers=4)

        self.parallel_linear1 = nn.Linear(d_model, d_model * 3 + 4 * d_model)
        self.parallel_linear2 = nn.Linear(d_model + 4 * d_model, d_model)
        self.parallel_activation = nn.GELU()

        self.scene_attention = EfficientAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            dropout=dropout,
            chunk_size=chunk_size,
            use_flash_attention=use_flash_attention,
            attention_dropout=cross_attention_dropout,
        )
        self.text_attention = EfficientAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            dropout=dropout,
            chunk_size=chunk_size,
            use_flash_attention=use_flash_attention,
            attention_dropout=cross_attention_dropout,
        )
        self.final_ff = FeedForward(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        scene_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        cond_vector: Optional[torch.Tensor] = None,
        grasp_poses: Optional[torch.Tensor] = None,
        scene_xyz: Optional[torch.Tensor] = None,
        latent_global: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
        t_scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond_vector is None:
            raise DiTConditioningError("cond_vector is required when use_adaln_zero=True")

        ctx = self.mod_stack.prepare(cond_vector)

        # Stage 1: parallel self-attention + MLP
        x_mod, gate_attn = ctx.apply(0, x)
        qkv_mlp = self.parallel_linear1(x_mod)
        qkv, mlp_in = torch.split(qkv_mlp, [self.d_model * 3, self.d_model * 4], dim=-1)
        B, N, _ = qkv.shape
        qkv = qkv.view(B, N, 3, self.num_heads, self.d_head)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, N, d)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        mlp_out = self.parallel_activation(mlp_in)
        parallel_out = self.parallel_linear2(torch.cat([attn_out, mlp_out], dim=-1))
        parallel_out = self.parallel_dropout(parallel_out)
        x = x + gate_attn * parallel_out

        # Stage 2: scene cross attention
        if scene_context is not None:
            scene_mod, gate_scene = ctx.apply(1, x)
            if scene_mask is not None and scene_mask.dim() == 3:
                scene_mask = scene_mask.squeeze(1)
            attention_bias = None
            if self.use_geometric_bias and self.geometric_bias_module is not None and scene_xyz is not None and grasp_poses is not None:
                try:
                    attention_bias = self.geometric_bias_module(scene_mod, scene_xyz, grasp_poses)
                except Exception as exc:
                    logging.getLogger(__name__).warning(f"Geometric bias failed in parallel block: {exc}")
                    attention_bias = None
            scene_out = self.scene_attention(
                scene_mod,
                scene_context,
                scene_context,
                mask=scene_mask,
                attention_bias=attention_bias,
            )
            if self.time_gate is not None and t_scalar is not None:
                alpha_scene = self.time_gate.get_scene_gate(t=t_scalar, time_emb=time_emb)
                if alpha_scene is not None:
                    scene_out = scene_out * alpha_scene
            x = x + gate_scene * scene_out

        # Stage 3: text cross attention
        if text_context is not None or text_tokens is not None:
            text_mod, gate_text = ctx.apply(2, x)
            if text_tokens is not None:
                text_out = self.text_attention(
                    text_mod,
                    text_tokens,
                    text_tokens,
                    mask=text_token_mask,
                )
            else:
                text_out = self.text_attention(text_mod, text_context, text_context)
            if self.time_gate is not None and t_scalar is not None:
                alpha_text = self.time_gate.get_text_gate(t=t_scalar, time_emb=time_emb)
                if alpha_text is not None:
                    text_out = text_out * alpha_text
            x = x + gate_text * text_out

        # Stage 4: final feed-forward
        ff_in, gate_ff = ctx.apply(3, x)
        ff_out = self.final_ff(ff_in)
        x = x + gate_ff * ff_out

        return x


class DiTBlock(nn.Module):
    """
    DiT transformer block with self-attention, cross-attention, and feed-forward layers.
    Enhanced with memory optimization features and mask support.
    
    支持两种条件注入模式：
    1. AdaLN-Zero 模式：融合时间步+场景+文本的多条件调制（use_adaln_zero=True）
    2. 原有 AdaptiveLayerNorm 模式：仅使用时间步调制（use_adaln_zero=False）
    """
    def __init__(self, d_model: int, num_heads: int, d_head: int, dropout: float = 0.1,
                 use_adaptive_norm: bool = True, time_embed_dim: Optional[int] = None,
                 use_adaln_zero: bool = False, cond_dim: Optional[int] = None,
                 chunk_size: int = 512, use_flash_attention: bool = False,
                 attention_dropout: float = 0.0, cross_attention_dropout: float = 0.0,
                 use_geometric_bias: bool = False, geometric_bias_module: Optional[nn.Module] = None,
                 time_gate: Optional[Any] = None):
        super().__init__()
        self.d_model = d_model
        self.use_adaptive_norm = use_adaptive_norm
        self.use_adaln_zero = use_adaln_zero
        self.use_geometric_bias = use_geometric_bias
        self.geometric_bias_module = geometric_bias_module
        self.time_gate = time_gate
        self.adaln_zero_stack: Optional[AdaLNZeroStack] = None
        
        # Memory-efficient attention layers with dropout support
        self.self_attention = EfficientAttention(
            d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
            attention_dropout=attention_dropout
        )
        self.scene_cross_attention = EfficientAttention(
            d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
            attention_dropout=cross_attention_dropout
        )
        self.text_cross_attention = EfficientAttention(
            d_model, num_heads, d_head, dropout, chunk_size, use_flash_attention,
            attention_dropout=cross_attention_dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, dropout=dropout)
        
        # Normalization layers - 支持三种模式
        if use_adaln_zero and cond_dim is not None and cond_dim > 0:
            # AdaLN-Zero 模式：多条件融合
            self.adaln_zero_stack = AdaLNZeroStack(d_model, cond_dim, num_layers=4)
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None
            self.norm4 = None
        elif use_adaptive_norm and time_embed_dim is not None:
            # 原有 AdaptiveLayerNorm 模式：仅时间步调制
            self.norm1 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm2 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm3 = AdaptiveLayerNorm(d_model, time_embed_dim)
            self.norm4 = AdaptiveLayerNorm(d_model, time_embed_dim)
        else:
            # 普通 LayerNorm 模式
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.norm4 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None,
                scene_context: Optional[torch.Tensor] = None,
                text_context: Optional[torch.Tensor] = None,
                scene_mask: Optional[torch.Tensor] = None,
                cond_vector: Optional[torch.Tensor] = None,
                grasp_poses: Optional[torch.Tensor] = None,
                scene_xyz: Optional[torch.Tensor] = None,
                latent_global: Optional[torch.Tensor] = None,
                local_indices: Optional[torch.Tensor] = None,
                local_mask: Optional[torch.Tensor] = None,
                text_tokens: Optional[torch.Tensor] = None,
                text_token_mask: Optional[torch.Tensor] = None,
                t_scalar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, num_grasps, d_model)
            time_emb: (B, time_embed_dim) or None - 用于原有 AdaptiveLayerNorm 模式
            scene_context: (B, N_points, d_model) or None
            text_context: (B, 1, d_model) or None - pooled 文本特征
            scene_mask: (B, N_points) or (B, 1, N_points) - mask for scene padding, 1=valid, 0=padding
            cond_vector: (B, cond_dim) or None - 用于 AdaLN-Zero 模式的多条件向量
            grasp_poses: (B, num_grasps, d_x) or None - grasp姿态（用于几何偏置计算）
            scene_xyz: (B, N_points, 3) or None - 场景点云坐标（用于几何偏置计算）
            latent_global: (B, K, d_model) or None - 全局 latent tokens（Global-Local conditioning）
            local_indices: (B, G, k) or None - 每个抓取的局部邻域索引
            local_mask: (B, G, k) or None - 局部邻域掩码，1=valid, 0=padding
            text_tokens: (B, L_text, d_model) or None - token 级别文本特征
            text_token_mask: (B, L_text) or None - token attention mask, 1=valid, 0=padding
            t_scalar: (B,) or None - 归一化时间 [0, 1]，用于时间门控
        Returns:
            output: (B, num_grasps, d_model)
        """
        if self.use_adaln_zero and cond_vector is None:
            raise DiTConditioningError("cond_vector 为空，但 use_adaln_zero=True，需要提供条件向量。")

        # Self-attention among grasps（Pre-LN + 条件调制 → 子层 → gate 残差）
        adaln_ctx = None
        if self.use_adaln_zero and cond_vector is not None:
            if self.adaln_zero_stack is None:
                raise DiTConditioningError("AdaLNZeroStack 未初始化，但 use_adaln_zero=True。")
            adaln_ctx = self.adaln_zero_stack.prepare(cond_vector)

        if adaln_ctx is not None:
            x_mod, gate_attn = adaln_ctx.apply(0, x)
            if torch.isnan(x_mod).any():
                logging.error(f"[DiTBlock NaN] NaN after norm1 (AdaLN-Zero)")
                logging.error(f"  Input x stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")
                raise RuntimeError("NaN detected in DiTBlock after norm1 (AdaLN-Zero)")
            attn_out = self.self_attention(x_mod)
        else:
            # 原有 AdaptiveLayerNorm 或普通 LN 路径
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
            raise RuntimeError("NaN detected in DiTBlock self_attention")
        if adaln_ctx is not None:
            x = x + gate_attn * attn_out
        else:
            x = x + attn_out

        # Cross-attention with scene features
        if scene_context is not None:
            if adaln_ctx is not None:
                x_mod2, gate_scene = adaln_ctx.apply(1, x)
                if torch.isnan(x_mod2).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm2 (AdaLN-Zero)")
                    raise RuntimeError("NaN detected in DiTBlock after norm2 (AdaLN-Zero)")
                norm_input = x_mod2
            elif self.use_adaptive_norm and time_emb is not None:
                norm_input = self.norm2(x, time_emb)
                if torch.isnan(norm_input).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm2")
                    raise RuntimeError("NaN detected in DiTBlock after norm2")
            else:
                norm_input = self.norm2(x)
                if torch.isnan(norm_input).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm2")
                    raise RuntimeError("NaN detected in DiTBlock after norm2")
            
            # Global-Local Scene Conditioning: 构建 [latent_global ⊕ local_patch]
            if latent_global is not None and local_indices is not None:
                # 1. 使用 local_indices 从 scene_context 中 gather 局部特征
                # local_indices: (B, G, k), scene_context: (B, N, d_model)
                # 需要将 local_indices 扩展到匹配 d_model 维度
                B, G, k = local_indices.shape
                _, N, d_model = scene_context.shape
                
                # 扩展 local_indices: (B, G, k) -> (B, G, k, d_model)
                local_indices_expanded = local_indices.unsqueeze(-1).expand(B, G, k, d_model)
                
                # Gather 局部特征: (B, N, d_model) -> (B, G, k, d_model)
                # 首先将 scene_context 扩展到 (B, 1, N, d_model) 以便 gather
                scene_context_expanded = scene_context.unsqueeze(1).expand(B, G, N, d_model)
                local_features = torch.gather(scene_context_expanded, 2, local_indices_expanded)
                
                # 2. 展平局部特征: (B, G, k, d_model) -> (B, G*k, d_model)
                local_features_flat = local_features.view(B, G * k, d_model)
                
                # 3. 拼接全局和局部: (B, K, d_model) + (B, G*k, d_model) -> (B, K+G*k, d_model)
                scene_context_combined = torch.cat([latent_global, local_features_flat], dim=1)
                
                # 4. 构建组合掩码: (B, K) + (B, G*k) -> (B, K+G*k)
                # 全局 latent 没有 mask（全部有效）
                B_mask, K = latent_global.shape[:2]
                latent_mask = torch.ones(B_mask, K, dtype=torch.float32, device=latent_global.device)
                
                # 局部掩码: (B, G, k) -> (B, G*k)
                local_mask_flat = local_mask.view(B, G * k)
                
                # 拼接掩码
                scene_mask_combined = torch.cat([latent_mask, local_mask_flat], dim=1)
                
                # 替换 scene_context 和 scene_mask
                scene_context = scene_context_combined
                scene_mask = scene_mask_combined
                
                # 5. 如果使用几何偏置，也需要组合 scene_xyz
                # 为全局 latent 创建虚拟坐标（使用零向量或平均坐标）
                # 为局部特征使用实际的点云坐标
                if self.use_geometric_bias and scene_xyz is not None:
                    # 为全局 latent 创建虚拟坐标（这里使用零向量）
                    # 或者可以使用点云的中心点作为虚拟坐标
                    # latent_xyz = torch.zeros(B, K, 3, device=scene_xyz.device, dtype=scene_xyz.dtype)
                    
                    # 更好的方案：使用点云的均值作为全局 latent 的虚拟坐标
                    if scene_mask is not None and scene_mask.dim() == 2:
                        # 使用原始 scene_mask (在组合之前)
                        # 需要保存原始的 scene_xyz 和 scene_mask
                        # 计算有效点的均值
                        valid_mask = scene_mask.unsqueeze(-1)  # (B, N_original, 1)
                        masked_xyz = scene_xyz * valid_mask
                        sum_xyz = masked_xyz.sum(dim=1, keepdim=True)  # (B, 1, 3)
                        count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
                        mean_xyz = sum_xyz / count  # (B, 1, 3)
                        latent_xyz = mean_xyz.expand(B, K, 3)  # (B, K, 3)
                    else:
                        # 如果没有 mask，直接使用所有点的均值
                        mean_xyz = scene_xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
                        latent_xyz = mean_xyz.expand(B, K, 3)  # (B, K, 3)
                    
                    # Gather 局部点云坐标
                    # local_indices: (B, G, k) -> 扩展为 (B, G, k, 3)
                    local_indices_xyz = local_indices.unsqueeze(-1).expand(B, G, k, 3)
                    scene_xyz_expanded = scene_xyz.unsqueeze(1).expand(B, G, N, 3)
                    local_xyz = torch.gather(scene_xyz_expanded, 2, local_indices_xyz)  # (B, G, k, 3)
                    local_xyz_flat = local_xyz.view(B, G * k, 3)  # (B, G*k, 3)
                    
                    # 拼接坐标: (B, K, 3) + (B, G*k, 3) -> (B, K+G*k, 3)
                    scene_xyz = torch.cat([latent_xyz, local_xyz_flat], dim=1)

            # Compute geometric attention bias if enabled
            geometric_bias = None
            if self.use_geometric_bias and self.geometric_bias_module is not None:
                if grasp_poses is not None and scene_xyz is not None:
                    try:
                        # Note: 当使用局部邻域时，geometric_bias 应该也只计算局部部分的偏置
                        # 这里暂时保持原有逻辑，后续可以优化
                        geometric_bias = self.geometric_bias_module(norm_input, scene_xyz, grasp_poses)
                    except Exception as e:
                        logging.warning(f"Failed to compute geometric bias: {e}, proceeding without it")
                        geometric_bias = None

            # Pass scene_mask and geometric_bias to prevent attention to padding positions
            scene_attn_out = self.scene_cross_attention(
                norm_input, scene_context, scene_context, 
                mask=scene_mask, 
                attention_bias=geometric_bias
            )
            if torch.isnan(scene_attn_out).any():
                logging.error(f"[DiTBlock NaN] NaN after scene_cross_attention")
                logging.error(f"  norm_input stats: min={norm_input.min():.6f}, max={norm_input.max():.6f}")
                logging.error(f"  scene_context stats: min={scene_context.min():.6f}, max={scene_context.max():.6f}")
                raise RuntimeError("NaN detected in DiTBlock scene_cross_attention")

            # 应用场景时间门控
            if self.time_gate is not None and t_scalar is not None:
                alpha_scene = self.time_gate.get_scene_gate(t=t_scalar, time_emb=time_emb)
                if alpha_scene is not None:
                    scene_attn_out = scene_attn_out * alpha_scene

            if adaln_ctx is not None:
                x = x + gate_scene * scene_attn_out
            else:
                x = x + scene_attn_out

        # Cross-attention with text features (if available)
        # 优先使用 text_tokens (token 级别)，回退到 text_context (pooled 特征)
        if text_tokens is not None or text_context is not None:
            if adaln_ctx is not None:
                x_mod3, gate_text = adaln_ctx.apply(2, x)
                if torch.isnan(x_mod3).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm3 (AdaLN-Zero)")
                    raise RuntimeError("NaN detected in DiTBlock after norm3 (AdaLN-Zero)")
                norm_input_text = x_mod3
            elif self.use_adaptive_norm and time_emb is not None:
                norm_input_text = self.norm3(x, time_emb)
                if torch.isnan(norm_input_text).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm3")
                    raise RuntimeError("NaN detected in DiTBlock after norm3")
            else:
                norm_input_text = self.norm3(x)
                if torch.isnan(norm_input_text).any():
                    logging.error(f"[DiTBlock NaN] NaN after norm3")
                    raise RuntimeError("NaN detected in DiTBlock after norm3")

            # 选择使用 token 级别特征或 pooled 特征
            if text_tokens is not None:
                # 使用 token 序列进行更细粒度的 cross-attention
                text_attn_out = self.text_cross_attention(
                    norm_input_text, text_tokens, text_tokens, 
                    mask=text_token_mask
                )
            else:
                # 回退到 pooled 特征
                text_attn_out = self.text_cross_attention(norm_input_text, text_context, text_context)
            
            if torch.isnan(text_attn_out).any():
                logging.error(f"[DiTBlock NaN] NaN after text_cross_attention")
                logging.error(f"  norm_input_text stats: min={norm_input_text.min():.6f}, max={norm_input_text.max():.6f}")
                if text_tokens is not None:
                    logging.error(f"  text_tokens stats: min={text_tokens.min():.6f}, max={text_tokens.max():.6f}")
                else:
                    logging.error(f"  text_context stats: min={text_context.min():.6f}, max={text_context.max():.6f}")
                raise RuntimeError("NaN detected in DiTBlock text_cross_attention")

            # 应用文本时间门控
            if self.time_gate is not None and t_scalar is not None:
                alpha_text = self.time_gate.get_text_gate(t=t_scalar, time_emb=time_emb)
                if alpha_text is not None:
                    text_attn_out = text_attn_out * alpha_text

            if adaln_ctx is not None:
                x = x + gate_text * text_attn_out
            else:
                x = x + text_attn_out

        # Feed-forward network（Pre-LN + 条件调制 → 子层 → gate 残差）
        if adaln_ctx is not None:
            ff_in, gate_ffn = adaln_ctx.apply(3, x)
            if torch.isnan(ff_in).any():
                logging.error(f"[DiTBlock NaN] NaN after norm4 (AdaLN-Zero)")
                raise RuntimeError("NaN detected in DiTBlock after norm4 (AdaLN-Zero)")
            ff_out = self.feed_forward(ff_in)
        else:
            if self.use_adaptive_norm and time_emb is not None:
                ff_in = self.norm4(x, time_emb)
            else:
                ff_in = self.norm4(x)
            if torch.isnan(ff_in).any():
                logging.error(f"[DiTBlock NaN] NaN after norm4")
                raise RuntimeError("NaN detected in DiTBlock after norm4")
            ff_out = self.feed_forward(ff_in)
        if torch.isnan(ff_out).any():
            logging.error(f"[DiTBlock NaN] NaN after feed_forward")
            logging.error(f"  ff_in stats: min={ff_in.min():.6f}, max={ff_in.max():.6f}")
            raise RuntimeError("NaN detected in DiTBlock feed_forward")

        if adaln_ctx is not None:
            x = x + gate_ffn * ff_out
        else:
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


class DiTSingleSerialNoCrossBlock(nn.Module):
    """
    串行单流（无 cross-attn）块：仅自注意力 + FFN，使用 AdaLN-Zero 调制。

    该块用于单流阶段的“融合串行”形态，默认期望处理拼接后的序列。
    """
    expects_concatenated_sequence = True

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        dropout: float,
        cond_dim: int,
        attention_dropout: float,
        debug_check_nan: bool = False,
        debug_log_stats: bool = False,
    ):
        super().__init__()
        if cond_dim is None or cond_dim <= 0:
            raise ValueError("DiTSingleSerialNoCrossBlock requires valid cond_dim with AdaLN-Zero.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.debug_check_nan = debug_check_nan
        self.debug_log_stats = debug_log_stats

        self.adaln_stack = AdaLNZeroStack(d_model, cond_dim, num_layers=2)
        self.self_attention = EfficientAttention(
            d_model, num_heads, d_head, dropout, chunk_size=512,
            use_flash_attention=False, attention_dropout=attention_dropout
        )
        self.ffn = FeedForward(d_model, dropout=dropout)

    def _log_stats(self, name: str, tensor: torch.Tensor):
        if not self.debug_log_stats or tensor is None:
            return
        try:
            logging.getLogger(__name__).debug(
                f"[SerialNoCross] {name}: shape={tuple(tensor.shape)} min={tensor.min():.6f} max={tensor.max():.6f} mean={tensor.mean():.6f}"
            )
        except Exception:
            pass

    def _check_finite(self, name: str, tensor: torch.Tensor):
        if not self.debug_check_nan or tensor is None:
            return
        if not torch.isfinite(tensor).all():
            logging.getLogger(__name__).error(f"[SerialNoCross] Non-finite at {name}")
            raise RuntimeError("NaN/Inf detected in SerialNoCross block")

    def forward(self, x: torch.Tensor, cond_vector: torch.Tensor) -> torch.Tensor:
        # Self-attention
        ctx = self.adaln_stack.prepare(cond_vector)
        x_mod, gate_attn = ctx.apply(0, x)
        self._check_finite("x_mod", x_mod)
        attn_out = self.self_attention(x_mod)
        self._check_finite("attn_out", attn_out)
        x = x + gate_attn * attn_out

        # FFN
        ff_in, gate_ff = ctx.apply(1, x)
        self._check_finite("ff_in", ff_in)
        ff_out = self.ffn(ff_in)
        self._check_finite("ff_out", ff_out)
        x = x + gate_ff * ff_out
        return x


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
        
        # AdaLN-Zero 多条件融合配置
        self.use_adaln_zero = getattr(cfg, 'use_adaln_zero', False)
        self.use_scene_pooling = getattr(cfg, 'use_scene_pooling', True)
        # AdaLN 模式：'multi' 使用多条件(time+scene+text)，'simple' 仅使用 time_emb
        self.adaln_mode = getattr(cfg, 'adaln_mode', 'multi')
        # 双流-单流架构配置
        self.use_double_stream = getattr(cfg, 'use_double_stream', False)
        self.num_double_blocks = int(getattr(cfg, 'num_double_blocks', 0))
        self.single_block_variant = getattr(cfg, 'single_block_variant', 'legacy')
        self._warned_missing_scene_cond = False
        self._warned_missing_text_cond = False
        
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
        
        # Memory optimization config
        self.gradient_checkpointing = getattr(cfg, 'gradient_checkpointing', False)
        self.use_flash_attention = getattr(cfg, 'use_flash_attention', False)
        self.attention_chunk_size = getattr(cfg, 'attention_chunk_size', 512)

        # Attention dropout config (for regularization)
        self.attention_dropout = getattr(cfg, 'attention_dropout', 0.0)
        self.cross_attention_dropout = getattr(cfg, 'cross_attention_dropout', 0.0)
        
        # Geometric attention bias config
        self.use_geometric_bias = getattr(cfg, 'use_geometric_bias', False)
        self.geometric_bias_hidden_dims = getattr(cfg, 'geometric_bias_hidden_dims', [128, 64])
        self.geometric_bias_feature_types = getattr(cfg, 'geometric_bias_feature_types', ['relative_pos', 'distance'])
        
        # Time-aware conditioning config
        self.use_t_aware_conditioning = getattr(cfg, 'use_t_aware_conditioning', False)
        self.t_gate_config = getattr(cfg, 't_gate', None)
        
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

        # Debug/监控开关
        try:
            self.debug_check_nan = bool(getattr(getattr(cfg, 'debug', {}), 'check_nan', False))
        except Exception:
            self.debug_check_nan = False
        try:
            self.debug_log_stats = bool(getattr(getattr(cfg, 'debug', {}), 'log_tensor_stats', False))
        except Exception:
            self.debug_log_stats = False
        
        # Core DiT components
        self.grasp_tokenizer = GraspTokenizer(self.d_x, self.d_model)
        
        if self.use_learnable_pos_embedding:
            self.pos_embedding = PositionalEmbedding(self.d_model, self.max_sequence_length)
        else:
            self.pos_embedding = None
            
        self.time_embedding = TimestepEmbedding(self.d_model, self.time_embed_dim)
        
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
                    "AdaLN-Zero enabled (mode=multi) with projected cond_dim=%d (time=%d + projected scene=%s + "
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
                    f"AdaLN-Zero enabled (mode=simple) with cond_dim={cond_dim} (time={self.time_embed_dim})"
                )
            self.logger.info(
                "AdaLN-Zero activation summary: mode=%s, cond_dim=%d, time_embed_dim=%d",
                self.adaln_mode,
                cond_dim,
                self.time_embed_dim,
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
        
        # MLP ratio for Hunyuan-style parallel single stream blocks
        self.mlp_ratio = getattr(cfg, 'mlp_ratio', 4.0)
        
        # DiT transformer blocks with memory optimization
        dit_blocks = []
        for i in range(self.num_layers):
            if self.use_double_stream and i < self.num_double_blocks:
                # 双流块
                block = DiTDoubleStreamBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_head=self.d_head,
                    dropout=self.dropout,
                    cond_dim=cond_dim if self.use_adaln_zero else None,
                    chunk_size=self.attention_chunk_size,
                    use_flash_attention=self.use_flash_attention,
                    attention_dropout=self.attention_dropout,
                    debug_check_nan=self.debug_check_nan,
                    debug_log_stats=self.debug_log_stats,
                )
                self.logger.info(f"Layer {i}: DiTDoubleStreamBlock")
            else:
                # 单流块：根据 single_block_variant 选择
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
                        debug_check_nan=self.debug_check_nan,
                        debug_log_stats=self.debug_log_stats,
                    )
                    self.logger.info(f"Layer {i}: FusedParallelSingleStream (mlp_ratio={self.mlp_ratio})")
                elif self.single_block_variant == "fused_serial":
                    block = DiTSingleSerialNoCrossBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        d_head=self.d_head,
                        dropout=self.dropout,
                        cond_dim=cond_dim if self.use_adaln_zero else 0,
                        attention_dropout=self.attention_dropout,
                        debug_check_nan=self.debug_check_nan,
                        debug_log_stats=self.debug_log_stats,
                    )
                    self.logger.info(f"Layer {i}: FusedSerialSingleStream (no cross-attn)")
                else:
                    # 传统 DiTBlock
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
                    if self.gradient_checkpointing:
                        block = GradientCheckpointedDiTBlock(block, use_checkpointing=True)
                    self.logger.info(f"Layer {i}: DiTBlock (legacy)")
            dit_blocks.append(block)
        
        self.dit_blocks = nn.ModuleList(dit_blocks)
        
        # 记录整体架构信息
        if self.use_double_stream and self.num_double_blocks > 0:
            self.logger.info(
                f"✓ DiT MM-DiT Architecture: {self.num_double_blocks} double-stream + "
                f"{self.num_layers - self.num_double_blocks} {self.single_block_variant} single-stream blocks"
            )
        
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
        scene_context, text_context, text_tokens, text_token_mask = self._resolve_condition_features(data, model_device)
        
        # Extract scene_mask if available, otherwise create a full-valid mask
        scene_mask = self._normalize_scene_mask(data.get("scene_mask", None), scene_context)
        if scene_mask is not None:
            data["scene_mask"] = scene_mask
        
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
        
        # Extract t_scalar for time-aware conditioning
        t_scalar = None
        if self.use_t_aware_conditioning:
            t_scalar = data.get('t_scalar', None)
            if t_scalar is not None:
                if not isinstance(t_scalar, torch.Tensor):
                    self.logger.warning(f"t_scalar is not a tensor ({type(t_scalar)}), ignoring")
                    t_scalar = None
                elif t_scalar.device != model_device:
                    t_scalar = t_scalar.to(model_device)
        
        # 准备条件向量（AdaLN-Zero 模式）
        cond_vector = None
        if self.use_adaln_zero:
            if self.adaln_mode == 'simple':
                cond_vector = time_emb
                self._assert_finite(cond_vector, "cond_vector (simple)")
            else:
                if time_emb is None:
                    raise DiTConditioningError("time_emb is required when use_adaln_zero=True")
                cond_vector = build_adaln_cond_vector(
                    time_emb,
                    use_scene_pooling=self.use_scene_pooling,
                    scene_to_time=self.scene_to_time,
                    scene_context=scene_context,
                    scene_mask=scene_mask,
                    use_text_condition=self.use_text_condition,
                    text_to_time=self.text_to_time,
                    text_context=text_context,
                    logger=self.logger,
                )
                self._assert_finite(cond_vector, "cond_vector")
                self.logger.debug(f"AdaLN-Zero cond_vector shape: {cond_vector.shape}")
        
        # Global-Local Scene Conditioning（全局-局部两阶段场景条件）
        latent_global = None
        local_indices = None
        local_mask = None
        if self.use_global_local_conditioning and scene_context is not None:
            # 1. 全局阶段：计算 global latent tokens
            latent_global = self.global_scene_pool(scene_context, scene_mask)
            self._assert_finite(latent_global, "latent_global")
            self.logger.debug(
                f"Global latents computed: shape={latent_global.shape}, "
                f"min={latent_global.min():.6f}, max={latent_global.max():.6f}"
            )
            
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
                # grasp_translations: 从 x_t 中提取平移部分（前3维）
                # x_t 形状: (B, num_grasps, d_x)，其中 d_x = 3 (translation) + 16 (joints) + 4/6 (rotation)
                grasp_translations = x_t[..., :3]  # (B, num_grasps, 3)
                
                # 使用 local_selector 选取局部邻域索引
                # local_indices: (B, G, k), local_mask: (B, G, k)
                local_indices, local_mask = self.local_selector(
                    grasp_translations=grasp_translations,
                    scene_xyz=local_scene_xyz,
                    scene_mask=scene_mask,
                )
                self.logger.debug(
                    f"Local neighborhoods selected: indices shape={local_indices.shape}, "
                    f"mask shape={local_mask.shape}"
                )

        x = self._run_dit_blocks(
            grasp_tokens,
            time_emb if self.use_adaptive_norm else None,
            scene_context,
            text_context,
            scene_mask=scene_mask,
            cond_vector=cond_vector,
            grasp_poses=x_t,
            scene_xyz=scene_xyz,
            latent_global=latent_global,
            local_indices=local_indices,
            local_mask=local_mask,
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            t_scalar=t_scalar
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
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        解析条件特征，支持 pooled 和 token 级别的文本特征。
        
        Returns:
            scene_context: (B, N_points, d_model) or None
            text_context: (B, 1, d_model) or None - pooled 文本特征
            text_tokens: (B, L_text, d_model) or None - token 级别文本特征
            text_token_mask: (B, L_text) or None - token attention mask
        """
        scene_context = data.get("scene_cond")
        if scene_context is not None:
            if not isinstance(scene_context, torch.Tensor):
                raise DiTConditioningError(f"scene_cond must be torch.Tensor, got {type(scene_context)}")
            if scene_context.device != model_device:
                scene_context = scene_context.to(model_device)
                self.logger.debug(f"Moved scene_cond to device {model_device}")
            self._assert_finite(scene_context, "scene_context")

        text_context = None
        text_tokens = None
        text_token_mask = None
        
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
            
            # 提取 token 级别的文本特征（如果启用且可用）
            if self.use_text_tokens:
                raw_text_tokens = data.get("text_tokens")
                if raw_text_tokens is not None and isinstance(raw_text_tokens, torch.Tensor):
                    if raw_text_tokens.device != model_device:
                        raw_text_tokens = raw_text_tokens.to(model_device)
                    self._assert_finite(raw_text_tokens, "text_tokens")
                    text_tokens = raw_text_tokens
                    
                    # 提取 token mask
                    raw_text_token_mask = data.get("text_token_mask")
                    if raw_text_token_mask is not None and isinstance(raw_text_token_mask, torch.Tensor):
                        if raw_text_token_mask.device != model_device:
                            raw_text_token_mask = raw_text_token_mask.to(model_device)
                        text_token_mask = raw_text_token_mask
                    
                    self.logger.debug(
                        f"Using text tokens: shape={text_tokens.shape}, "
                        f"mask={'yes' if text_token_mask is not None else 'no'}"
                    )
        
        return scene_context, text_context, text_tokens, text_token_mask

    def _run_dit_blocks(
        self,
        tokens: torch.Tensor,
        time_emb: Optional[torch.Tensor],
        scene_context: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor],
        scene_mask: Optional[torch.Tensor] = None,
        cond_vector: Optional[torch.Tensor] = None,
        grasp_poses: Optional[torch.Tensor] = None,
        scene_xyz: Optional[torch.Tensor] = None,
        latent_global: Optional[torch.Tensor] = None,
        local_indices: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_token_mask: Optional[torch.Tensor] = None,
        t_scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        运行 DiT 块，支持双流+单流混合架构（参考 Hunyuan3D-DiT）。
        
        架构流程：
        1. 双流阶段：分别处理 grasp_tokens 和 scene_tokens
        2. 拼接：将两个流拼接为统一序列 [scene ⊕ grasp]
        3. 单流阶段：处理拼接后的序列
        4. 分离：提取回 grasp 部分
        """
        x = tokens
        current_scene = scene_context
        num_scene_tokens = scene_context.shape[1] if scene_context is not None else 0
        
        # 阶段1: 双流处理
        double_block_count = 0
        for idx, block in enumerate(self.dit_blocks):
            if not getattr(block, "is_double_stream", False):
                break
            
            try:
                if current_scene is None:
                    raise DiTConditioningError(
                        "Double-stream block requires scene_context, but it is None."
                    )
                self.logger.debug(
                    f"[DiT Double Block {idx}/{self.num_layers}] "
                    f"grasp shape={tuple(x.shape)}, scene shape={tuple(current_scene.shape)}"
                )
                x, current_scene = block(
                    grasp_tokens=x,
                    scene_tokens=current_scene,
                    cond_vector=cond_vector,
                    scene_mask=scene_mask,
                )
                self._assert_finite(x, f"DiT double block {idx} grasp output")
                self._assert_finite(current_scene, f"DiT double block {idx} scene output")
                double_block_count += 1
            except DiTConditioningError:
                raise
            except Exception as exc:
                self.logger.error(f"[DiT Double Block {idx}] Error in block processing: {exc}")
                raise DiTConditioningError(f"Error in DiT double block {idx}: {exc}") from exc
        
        # 关键转换：拼接两个流（如果有双流块）
        if double_block_count > 0 and current_scene is not None:
            # 拼接顺序：[scene, grasp]（与 Hunyuan3D-DiT 一致）
            x = torch.cat([current_scene, x], dim=1)  # (B, L_scene + L_grasp, D)
            self.logger.info(
                f"Concatenated scene and grasp tokens after {double_block_count} double blocks: "
                f"shape={tuple(x.shape)} (scene={num_scene_tokens}, grasp={x.shape[1] - num_scene_tokens})"
            )
            
            # 构建组合 mask
            if scene_mask is not None:
                batch_size = x.shape[0]
                num_grasps = tokens.shape[1]
                # grasp tokens 没有 mask（全部有效）
                grasp_mask = torch.ones(
                    batch_size, num_grasps, 
                    device=scene_mask.device, 
                    dtype=scene_mask.dtype
                )
                # 拼接 mask: [scene_mask, grasp_mask]
                combined_mask = torch.cat([scene_mask, grasp_mask], dim=1)
            else:
                combined_mask = None
            
            # 更新变量供单流块使用
            scene_mask_for_single = combined_mask
            # 单流块不再需要 scene_context 作为独立的 cross-attention 输入
            scene_context_for_single = None
        else:
            # 没有双流块，保持原有逻辑
            scene_mask_for_single = scene_mask
            scene_context_for_single = current_scene
        
        # 阶段2: 单流处理
        for idx in range(double_block_count, len(self.dit_blocks)):
            block = self.dit_blocks[idx]
            try:
                if getattr(block, "expects_concatenated_sequence", False):
                    # 并行单流块：只处理拼接后的序列
                    self.logger.debug(
                        f"[DiT Fused Single Block {idx}/{self.num_layers}] "
                        f"Input shape={tuple(x.shape)}"
                    )
                    x = block(
                        x=x,
                        cond_vector=cond_vector,
                        mask=scene_mask_for_single,
                    )
                else:
                    # 传统 DiTBlock（向后兼容）
                    self.logger.debug(
                        f"[DiT Block {idx}/{self.num_layers}] "
                        f"Input stats: shape={tuple(x.shape)}, min={x.min():.6f}, max={x.max():.6f}, "
                        f"mean={x.mean():.6f}, std={x.std():.6f}"
                    )
                    x = block(
                        x=x,
                        time_emb=time_emb,
                        scene_context=scene_context_for_single,
                        text_context=text_context,
                        scene_mask=scene_mask_for_single,
                        cond_vector=cond_vector,
                        grasp_poses=grasp_poses,
                        scene_xyz=scene_xyz,
                        latent_global=latent_global,
                        local_indices=local_indices,
                        local_mask=local_mask,
                        text_tokens=text_tokens,
                        text_token_mask=text_token_mask,
                        t_scalar=t_scalar
                    )
                self._assert_finite(x, f"DiT block {idx} output")
            except DiTConditioningError:
                raise
            except Exception as exc:
                self.logger.error(f"[DiT Block {idx}] Error in block processing: {exc}")
                raise DiTConditioningError(f"Error in DiT block {idx}: {exc}") from exc
        
        # 关键转换：分离回 grasp 部分（如果有双流块）
        if double_block_count > 0 and num_scene_tokens > 0:
            # 提取 grasp 部分：[:, num_scene:, ...]
            x = x[:, num_scene_tokens:, ...]
            self.logger.info(
                f"Extracted grasp tokens after single blocks: shape={tuple(x.shape)}"
            )
        
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

        # PointNet2 返回采样后的 xyz 和特征
        sampled_xyz, scene_feat = self.scene_model(scene_points)
        self._assert_finite(scene_feat, "scene_feat (backbone output)")
        
        # 保存采样后的 xyz 到 data 中，供几何偏置使用
        # sampled_xyz shape: (B, K, 3)，K 是采样后的点数（通常是 128）
        data['scene_xyz_sampled'] = sampled_xyz
        self.logger.debug(
            f"[Conditioning] Sampled xyz saved: shape={tuple(sampled_xyz.shape)}, "
            f"min={sampled_xyz.min():.6f}, max={sampled_xyz.max():.6f}"
        )
        
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

    def _full_scene_mask(self, scene_context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if scene_context is None or not isinstance(scene_context, torch.Tensor):
            return None
        B, N, _ = scene_context.shape
        return torch.ones(B, N, device=scene_context.device, dtype=torch.float32)

    def _normalize_scene_mask(
        self,
        scene_mask: Optional[torch.Tensor],
        scene_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
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

        scene_mask = torch.clamp(scene_mask, 0.0, 1.0)
        self._assert_finite(scene_mask, "scene_mask (normalized)")
        return scene_mask

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
