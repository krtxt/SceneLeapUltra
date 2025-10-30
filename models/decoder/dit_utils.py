import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization (参考 Hunyuan3D-DiT)"""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """
    Query-Key Normalization (参考 Hunyuan3D-DiT)
    
    对 Q 和 K 分别进行 RMS 归一化，提高训练稳定性。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


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
