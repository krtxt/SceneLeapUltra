import logging
from typing import Optional

import torch

from .dit_conditioning import pool_scene_features


def build_adaln_cond_vector(
    time_emb: torch.Tensor,
    *,
    use_scene_pooling: bool,
    scene_to_time: Optional[torch.nn.Module],
    scene_context: Optional[torch.Tensor],
    scene_mask: Optional[torch.Tensor],
    use_text_condition: bool,
    text_to_time: Optional[torch.nn.Module],
    text_context: Optional[torch.Tensor],
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """
    构建 AdaLN‑Zero 条件向量（统一在 DiT 与 DiTFM 间复用）。

    Args:
        time_emb: (B, C) 经过投影后的时间嵌入（即与 cond_dim 一致）。
        use_scene_pooling: 是否融合场景池化特征。
        scene_to_time: 将场景 pooled 特征映射到 cond_dim 的线性层（零初始化）。
        scene_context: (B, N, D) 场景特征序列或 None。
        scene_mask: (B, N) 场景 mask 或 None。
        use_text_condition: 是否融合文本条件。
        text_to_time: 文本 pooled → cond 的线性层（零初始化）。
        text_context: (B, 1, D) 或 (B, D) 的 pooled 文本或 None（函数内部统一 squeeze）。
        logger: 可选 logger。

    Returns:
        cond_vector: (B, C)
    """
    log = logger or logging.getLogger(__name__)

    cond_sum = time_emb
    device = time_emb.device
    dtype = time_emb.dtype
    batch_size = time_emb.shape[0]

    # 场景条件
    if use_scene_pooling and scene_to_time is not None:
        if scene_context is not None:
            scene_pooled = pool_scene_features(scene_context, scene_mask)
            scene_pooled = scene_pooled.to(device=device, dtype=dtype)
        else:
            log.warning(
                "AdaLN-Zero: use_scene_pooling=True 但缺少 scene_context；使用零向量填充。"
            )
            # 使用零向量保持与零初始化投影一致的“无偏置”行为
            scene_pooled = torch.zeros(batch_size, scene_to_time.in_features, device=device, dtype=dtype)
        scene_delta = scene_to_time(scene_pooled)
        cond_sum = cond_sum + scene_delta
        try:
            if log.isEnabledFor(logging.INFO):
                delta_norm = float(scene_delta.norm(dim=-1).mean())
                scene_norm = float(scene_pooled.norm(dim=-1).mean())
                log.info(
                    "AdaLN-Zero: fused scene pooling (||scene||=%.4f, ||Δ||=%.4f)",
                    scene_norm,
                    delta_norm,
                )
        except Exception:
            pass

    # 文本条件
    if use_text_condition and text_to_time is not None:
        if text_context is not None:
            if text_context.dim() == 3 and text_context.shape[1] == 1:
                text_pooled = text_context.squeeze(1)
            else:
                text_pooled = text_context
            text_pooled = text_pooled.to(device=device, dtype=dtype)
        else:
            log.warning(
                "AdaLN-Zero: use_text_condition=True 但缺少 text_context；使用零向量填充。"
            )
            text_pooled = torch.zeros(batch_size, text_to_time.in_features, device=device, dtype=dtype)
        text_delta = text_to_time(text_pooled)
        cond_sum = cond_sum + text_delta
        try:
            if log.isEnabledFor(logging.INFO):
                delta_norm = float(text_delta.norm(dim=-1).mean())
                text_norm = float(text_pooled.norm(dim=-1).mean())
                log.info(
                    "AdaLN-Zero: fused text pooling (||text||=%.4f, ||Δ||=%.4f)",
                    text_norm,
                    delta_norm,
                )
        except Exception:
            pass

    return cond_sum


