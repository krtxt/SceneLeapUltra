import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Initialize weights consistent with existing DiT initialization.

    - Linear: Xavier uniform for weights, zeros for bias
    - LayerNorm: ones for weight, zeros for bias
    """
    for sub_module in module.modules():
        if isinstance(sub_module, nn.Linear):
            nn.init.xavier_uniform_(sub_module.weight)
            if sub_module.bias is not None:
                nn.init.zeros_(sub_module.bias)
        elif isinstance(sub_module, nn.LayerNorm):
            if sub_module.weight is not None:
                nn.init.ones_(sub_module.weight)
            if sub_module.bias is not None:
                nn.init.zeros_(sub_module.bias)


def adjust_backbone_config(backbone_cfg, use_rgb: bool, use_object_mask: bool):
    """Adjust backbone config based on modalities.

    This mirrors the logic from DiT/DiTFM to avoid duplication.
    - For PointNet2: update layer1.mlp_list[0] to feature_input_dim (rgb + mask)
    - For PTv3: store input_feature_dim for later use
    """
    from omegaconf import OmegaConf

    adjusted_cfg = copy.deepcopy(backbone_cfg)

    # rgb + optional object mask; xyz is handled by backbone itself
    feature_input_dim = (3 if use_rgb else 0) + (1 if use_object_mask else 0)

    backbone_name = getattr(adjusted_cfg, 'name', '').lower()

    if backbone_name == 'pointnet2':
        if (hasattr(adjusted_cfg, 'layer1') and
            hasattr(adjusted_cfg.layer1, 'mlp_list') and
            len(adjusted_cfg.layer1.mlp_list) > 0):
            mlp_list = list(adjusted_cfg.layer1.mlp_list)
            mlp_list[0] = feature_input_dim
            adjusted_cfg.layer1.mlp_list = mlp_list
    elif backbone_name == 'ptv3':
        OmegaConf.set_struct(adjusted_cfg, False)
        adjusted_cfg.input_feature_dim = feature_input_dim
        OmegaConf.set_struct(adjusted_cfg, True)

    return adjusted_cfg


def build_attn_mask_all(
    scene_mask: Optional[torch.Tensor],
    text_mask: Optional[torch.Tensor],
    len_x: int,
    len_scene: int,
    len_text: int,
    *,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Construct unified attention masks for MM-DiT self-attention.

    Returns a tuple ``(combined_mask, attn_mask)`` where:
        * combined_mask: (B, total_len) mask used to zero invalid tokens (1=valid, 0=masked)
        * attn_mask: (B, total_len, total_len) mask for attention weights or None if no masking is needed
    """
    if len_x < 0 or len_scene < 0 or len_text < 0:
        raise ValueError("Sequence lengths must be non-negative")

    total_len = len_x + len_scene + len_text
    if total_len == 0:
        return None, None

    if batch_size is None:
        if scene_mask is not None:
            batch_size = scene_mask.shape[0]
        elif text_mask is not None:
            batch_size = text_mask.shape[0]
        else:
            raise ValueError("batch_size must be provided when no masks are supplied")

    if device is None:
        if scene_mask is not None:
            device = scene_mask.device
        elif text_mask is not None:
            device = text_mask.device
        else:
            device = torch.device("cpu")

    combined_mask = torch.ones(batch_size, total_len, device=device, dtype=dtype)

    offset = len_x
    if len_scene > 0:
        if scene_mask is not None:
            combined_mask[:, offset:offset + len_scene] = scene_mask.to(device=device, dtype=dtype)
        else:
            combined_mask[:, offset:offset + len_scene] = 1.0
        offset += len_scene

    if len_text > 0:
        if text_mask is not None:
            combined_mask[:, offset:offset + len_text] = text_mask.to(device=device, dtype=dtype)
        else:
            combined_mask[:, offset:offset + len_text] = 1.0

    # Grasp tokens always valid
    combined_mask[:, :len_x] = 1.0
    combined_mask = torch.clamp(combined_mask, 0.0, 1.0)

    has_context_mask = (scene_mask is not None and len_scene > 0) or (text_mask is not None and len_text > 0)
    if not has_context_mask and len_scene == 0 and len_text == 0:
        attn_mask = None
    else:
        attn_mask = combined_mask.unsqueeze(1) * combined_mask.unsqueeze(2)

    return combined_mask, attn_mask
