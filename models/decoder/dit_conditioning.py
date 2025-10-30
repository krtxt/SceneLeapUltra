import logging
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from .dit_validation import DiTConditioningError
from models.utils.text_encoder import PosNegTextEncoder


def convert_to_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Convert various containers to a tensor on device/dtype with validation.

    Mirrors the behavior in DiT to ensure consistency between DDPM and FM.
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
        dtype=dtype,
    )

    for idx, tensor in enumerate(tensors):
        tensor = tensor.to(device=device, dtype=dtype)
        length = tensor.shape[0]
        if length == 0:
            continue
        padded[idx, :length] = tensor

    return padded


def normalize_object_mask(
    scene_points: torch.Tensor,
    object_mask: torch.Tensor,
    name: str = "object_mask",
) -> torch.Tensor:
    """Normalize object mask to shape (B, N, 1) aligned with scene_points."""
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
        elif mask.shape[0] == scene_points.shape[1] and mask.shape[1] == 1:
            mask = mask.unsqueeze(0).expand(scene_points.shape[0], -1, -1).squeeze(-1)
        else:
            raise DiTConditioningError(
                f"{name} shape {tuple(mask.shape)} incompatible with scene points {tuple(scene_points.shape)}"
            )
    elif mask.dim() == 3:
        if not (mask.shape[0] == scene_points.shape[0] and mask.shape[1] == scene_points.shape[1]):
            raise DiTConditioningError(
                f"{name} shape {tuple(mask.shape)} incompatible with scene points {tuple(scene_points.shape)}"
            )
    else:
        raise DiTConditioningError(f"{name} must be 1D/2D/3D, got {mask.dim()}D")

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    return mask


def assert_finite(tensor: Optional[torch.Tensor], name: str) -> None:
    if tensor is None or tensor.numel() == 0:
        return
    if torch.isnan(tensor).any():
        raise DiTConditioningError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise DiTConditioningError(f"Infinite value detected in {name}")


def replace_non_finite(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isnan(tensor).any():
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    if torch.isinf(tensor).any():
        tensor = torch.clamp(tensor, -1e6, 1e6)
    return tensor


def _fallback_scene_features(
    data: Dict,
    device: torch.device,
    projection: nn.Module,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    batch_size = 1
    if 'scene_pc' in data:
        scene_pc = data['scene_pc']
        if isinstance(scene_pc, torch.Tensor):
            batch_size = scene_pc.shape[0]
        elif isinstance(scene_pc, (list, tuple)):
            batch_size = len(scene_pc)
        elif np is not None and isinstance(scene_pc, np.ndarray):
            batch_size = scene_pc.shape[0]
    scene_feat_raw = torch.zeros(batch_size, 1024, 512, device=device)
    if logger is not None:
        logger.warning("Using fallback scene features due to extraction failure")
    return projection(scene_feat_raw)


def prepare_scene_features(
    scene_model: nn.Module,
    scene_projection: nn.Module,
    data: Dict,
    use_rgb: bool,
    use_object_mask: bool,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    strict: bool = True,
) -> torch.Tensor:
    """Process scene point cloud into contextual features with robust checks.

    This function aligns FM with DDPM's robust preprocessing: device/dtype normalization,
    optional mask alignment, finite checks, and safe fallback if strict=False.
    """
    try:
        if 'scene_pc' not in data or data['scene_pc'] is None:
            raise DiTConditioningError("Missing scene_pc in conditioning data")

        scene_pc = convert_to_tensor(
            data['scene_pc'], device=device, dtype=torch.float32, name="scene_pc"
        )
        if logger is not None:
            logger.debug(
                f"[Conditioning] scene_pc input: shape={tuple(scene_pc.shape)}, "
                f"min={scene_pc.min():.6f}, max={scene_pc.max():.6f}"
            )

        if not use_rgb:
            scene_pc = scene_pc[..., :3]
            if logger is not None:
                logger.debug(f"[Conditioning] Removed RGB, scene_pc shape: {tuple(scene_pc.shape)}")

        if use_object_mask and 'object_mask' in data and data['object_mask'] is not None:
            object_mask = convert_to_tensor(
                data['object_mask'], device=device, dtype=torch.float32, name="object_mask"
            )
            object_mask = normalize_object_mask(scene_pc, object_mask)
            scene_points = torch.cat([scene_pc, object_mask], dim=-1)
            if logger is not None:
                logger.debug(f"[Conditioning] Added object_mask, pos shape: {tuple(scene_points.shape)}")
        else:
            scene_points = scene_pc

        if scene_points.dim() != 3:
            raise DiTConditioningError(
                f"Scene point cloud must be 3D tensor, got {scene_points.dim()}D"
            )

        if logger is not None:
            logger.debug(
                f"[Conditioning] Final pos before backbone: shape={tuple(scene_points.shape)}, "
                f"min={scene_points.min():.6f}, max={scene_points.max():.6f}, mean={scene_points.mean():.6f}"
            )

        # Backbone返回采样后的xyz和特征
        sampled_xyz, scene_feat = scene_model(scene_points)
        assert_finite(scene_feat, "scene_feat (backbone output)")
        
        # 保存采样后的xyz到data中，供几何偏置使用
        # sampled_xyz shape: (B, K, 3)，K是采样后的点数
        data['scene_xyz_sampled'] = sampled_xyz
        if logger is not None:
            logger.debug(
                f"[Conditioning] Sampled xyz saved: shape={tuple(sampled_xyz.shape)}, "
                f"min={sampled_xyz.min():.6f}, max={sampled_xyz.max():.6f}"
            )
        
        if logger is not None:
            logger.debug(
                f"[Conditioning] Raw scene_feat shape: {tuple(scene_feat.shape)}, "
                f"tokens_last={getattr(scene_model, 'tokens_last', None)}"
            )

        # Ensure the feature dimension matches the projection input dim
        projection_in_dim = scene_projection.in_features
        if scene_feat.dim() != 3:
            raise DiTConditioningError(
                f"Backbone scene feature must be 3D tensor, got {scene_feat.dim()}D"
            )
        if scene_feat.shape[-1] == projection_in_dim:
            scene_feat = scene_feat.contiguous()
        elif scene_feat.shape[1] == projection_in_dim:
            scene_feat = scene_feat.permute(0, 2, 1).contiguous()
        else:
            raise DiTConditioningError(
                f"Scene feature shape {tuple(scene_feat.shape)} is incompatible with "
                f"projection input dim {projection_in_dim}"
            )

        if logger is not None:
            logger.debug(
                f"[Conditioning] Scene_feat aligned to projection: shape={tuple(scene_feat.shape)}"
            )

        scene_feat = replace_non_finite(scene_feat)
        scene_feat = scene_projection(scene_feat)
        assert_finite(scene_feat, "scene_feat (projected)")
        if logger is not None and logger.isEnabledFor(logging.INFO):
            B, K, C = scene_feat.shape
            logger.info(
                "[Scene Conditioning] Dense scene features ready: B=%d, K=%d, C=%d",
                B,
                K,
                C,
            )
        return scene_feat
    except Exception as exc:
        if strict:
            raise
        if logger is not None:
            logger.warning(f"Scene feature extraction failed: {exc}")
        return _fallback_scene_features(data, device, scene_projection, logger)


def ensure_text_encoder(
    text_encoder: Optional[PosNegTextEncoder],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> PosNegTextEncoder:
    """Lazily create or move text encoder to the target device."""
    if text_encoder is None:
        te = PosNegTextEncoder(device=device)
        te.to(device)
        if logger is not None:
            logger.info(f"Text encoder lazily initialized on device: {device}")
        return te
    # Ensure device is correct
    text_encoder.to(device)
    return text_encoder


def prepare_text_features(
    text_encoder: PosNegTextEncoder,
    text_processor: nn.Module,
    data: Dict,
    scene_feat: torch.Tensor,
    use_negative_prompts: bool,
    text_dropout_prob: float,
    training: bool,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Process text prompts with robust checks, returning conditioning dict.

    Aligns FM with DDPM behavior: validate inputs, dropout mask, optional negatives,
    and processing through TextConditionProcessor.
    """
    if scene_feat is None:
        batch_size = 1
    else:
        batch_size = scene_feat.shape[0]

    if 'positive_prompt' not in data:
        raise DiTConditioningError("Missing positive_prompt in conditioning data")

    positive_prompts = data['positive_prompt']
    if not isinstance(positive_prompts, (list, tuple)):
        raise DiTConditioningError(f"positive_prompt must be list or tuple, got {type(positive_prompts)}")
    if len(positive_prompts) != batch_size:
        raise DiTConditioningError(
            f"Batch size mismatch: scene features {batch_size}, prompts {len(positive_prompts)}"
        )

    # Encode positive prompts
    pos_text_outputs = text_encoder.encode_positive(positive_prompts)
    pos_text_sequence = pos_text_outputs['sequence']  # (B, L_text, d_model)
    pos_text_pooled = pos_text_outputs['pooled']      # (B, d_model)
    token_attention_mask = pos_text_outputs['attention_mask']  # (B, L_text)

    # Encode negative prompts if available
    neg_text_features: Optional[torch.Tensor] = None
    if use_negative_prompts and data.get('negative_prompts') is not None:
        try:
            neg_outputs = text_encoder.encode_negative(data['negative_prompts'])
            neg_text_features = neg_outputs['pooled']  # (B, num_neg, d_model)
            if torch.isnan(neg_text_features).any() or torch.isinf(neg_text_features).any():
                if logger is not None:
                    logger.warning("Negative text features contain non-finite values, disabling negatives")
                neg_text_features = None
        except Exception as exc:
            if logger is not None:
                logger.warning(f"Negative prompt encoding failed: {exc}. Continuing without negative prompts.")
            neg_text_features = None

    # Text dropout mask
    if training:
        text_mask = torch.bernoulli(
            torch.full((batch_size, 1), 1.0 - text_dropout_prob, device=device)
        )
    else:
        text_mask = torch.ones(batch_size, 1, device=device)

    # Process through text processor
    scene_embedding = torch.mean(scene_feat, dim=1) if scene_feat is not None else None
    pos_text_features_out, neg_pred = text_processor(pos_text_pooled, neg_text_features, scene_embedding)

    # Apply dropout mask to pooled/text tokens
    expanded_dropout_mask_seq = text_mask.unsqueeze(-1)  # (B, 1, 1)
    text_tokens = pos_text_sequence * expanded_dropout_mask_seq
    text_token_mask = token_attention_mask * text_mask  # broadcast (B, 1) -> (B, L)

    payload: Dict[str, Optional[torch.Tensor]] = {
        'text_cond': pos_text_features_out * text_mask,
        'text_mask': text_mask,
        'text_tokens': text_tokens,
        'text_token_mask': text_token_mask,
        'text_pooled': pos_text_pooled,
    }
    if use_negative_prompts:
        payload.update({
            'neg_pred': neg_pred,
            'neg_text_features': neg_text_features,
        })
    return payload


def pool_scene_features(
    scene_features: torch.Tensor, 
    scene_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    对场景特征进行带 mask 的均值池化，用于 AdaLN-Zero 多条件融合。
    
    Args:
        scene_features: (B, N_points, d_model) - 场景点云特征
        scene_mask: (B, N_points) or (B, 1, N_points) - mask for scene padding
                    1=valid, 0=padding (可选)
    
    Returns:
        pooled: (B, d_model) - 池化后的全局场景特征
    
    Examples:
        >>> scene_feat = torch.randn(4, 1024, 512)  # (B=4, N=1024, D=512)
        >>> mask = torch.ones(4, 1024)  # 全部有效
        >>> pooled = pool_scene_features(scene_feat, mask)
        >>> pooled.shape
        torch.Size([4, 512])
    """
    if scene_mask is not None:
        # 标准化 mask 的形状
        if scene_mask.dim() == 3:
            # (B, 1, N_points) -> (B, N_points)
            scene_mask = scene_mask.squeeze(1)
        
        # 扩展 mask 到特征维度
        mask_expanded = scene_mask.unsqueeze(-1)  # (B, N_points, 1)
        
        # 带 mask 的求和
        masked_sum = (scene_features * mask_expanded).sum(dim=1)  # (B, d_model)
        
        # 有效点数（至少为 1 以避免除零）
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        
        # 均值池化
        pooled = masked_sum / valid_counts  # (B, d_model)
    else:
        # 无 mask 时的简单均值池化
        pooled = scene_features.mean(dim=1)  # (B, d_model)
    
    return pooled
