"""
Global Scene Pooling module for two-stage scene conditioning.

Implements Perceiver-IO style latent cross-attention to compress scene features
into a fixed number of global latent tokens.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

from models.decoder.dit_memory_optimization import EfficientAttention


class GlobalScenePool(nn.Module):
    """
    全局场景池化模块（Perceiver-IO 思路）
    
    将 scene_context (B, N, d_model) 通过 latent cross-attention 压缩到固定数量的
    全局 latent tokens (B, K, d_model)，作为每层 DiT block 的全局记忆。
    
    Args:
        d_model: 特征维度
        num_latents: 全局 latent 数量 K（默认 128）
        num_layers: latent cross-attn 层数（默认 1，轻量级）
        num_heads: 注意力头数
        d_head: 每个头的维度
        dropout: Dropout 比例
        use_flash_attention: 是否使用 Flash Attention
    """
    
    def __init__(
        self,
        d_model: int,
        num_latents: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
        d_head: int = 64,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_layers = num_layers
        
        # 可学习的 latent queries
        self.latent_queries = nn.Parameter(torch.randn(num_latents, d_model) * 0.02)
        
        # Latent cross-attention layers
        self.cross_attns = nn.ModuleList([
            EfficientAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_head=d_head,
                dropout=dropout,
                chunk_size=512,
                use_flash_attention=use_flash_attention,
                attention_dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each cross-attention layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward network for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"GlobalScenePool initialized: K={num_latents}, layers={num_layers}, "
            f"d_model={d_model}, heads={num_heads}, d_head={d_head}"
        )
    
    def forward(
        self,
        scene_context: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            scene_context: (B, N, d_model) - 场景特征
            scene_mask: (B, N) or None - 场景掩码，1=valid, 0=padding
            
        Returns:
            latent_global: (B, K, d_model) - 全局 latent tokens
        """
        B, N, _ = scene_context.shape
        
        # 扩展 latent queries 到 batch
        latents = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
        
        # 多层 latent cross-attention
        for layer_idx in range(self.num_layers):
            # Cross-attention: latents attend to scene_context
            latents_normed = self.layer_norms[layer_idx](latents)
            
            # EfficientAttention(query, key, value, mask)
            # query: latents, key/value: scene_context
            attn_out = self.cross_attns[layer_idx](
                query=latents_normed,
                key=scene_context,
                value=scene_context,
                mask=scene_mask,  # (B, N) - mask for scene padding
            )
            
            latents = latents + attn_out
            
            # Feed-forward
            latents_normed = self.ffn_norms[layer_idx](latents)
            ffn_out = self.ffns[layer_idx](latents_normed)
            latents = latents + ffn_out
        
        return latents
    
    def __repr__(self):
        return (
            f"GlobalScenePool(d_model={self.d_model}, num_latents={self.num_latents}, "
            f"num_layers={self.num_layers})"
        )

