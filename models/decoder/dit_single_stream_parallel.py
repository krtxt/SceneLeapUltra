"""
并行单流块实现（参考 Hunyuan3D-DiT 的 SingleStreamBlock）
"""
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
import logging

try:
    from .dit_utils import QKNorm
except ImportError:
    # 如果导入失败，定义简单的归一化
    class QKNorm(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        def forward(self, q, k, v):
            return self.norm(q).to(v), self.norm(k).to(v)


class GELU(nn.Module):
    """GELU activation with approximate='tanh' (Hunyuan3D-DiT style)"""
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)


class ParallelSingleStreamBlock(nn.Module):
    """
    并行单流DiT Block，参考 Hunyuan3D-DiT 的 SingleStreamBlock。
    
    核心思想（MM-DiT 论文）：
    1. 并行计算 attention 和 MLP（而不是串行）
    2. 使用单个 linear1 生成 QKV + MLP输入（节省参数）
    3. 使用单个 linear2 融合 attention + MLP 输出
    4. 整体效率比传统串行 DiTBlock 高约 1.3x
    
    参考：
    - Hunyuan3D-DiT SingleStreamBlock (220-268行)
    - FLUX DiT: https://arxiv.org/abs/2302.05442
    """
    
    # 表示该块期望输入为拼接后的 [scene ⊕ grasp] 序列
    expects_concatenated_sequence = True
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        cond_dim: Optional[int] = None,
        use_flash_attention: bool = False,
        debug_check_nan: bool = False,
        debug_log_stats: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.mlp_hidden_dim = int(d_model * mlp_ratio)
        self.logger = logging.getLogger(__name__)
        self.debug_check_nan = debug_check_nan
        self.debug_log_stats = debug_log_stats
        
        # 预归一化（无 affine 参数，通过 modulation 控制）
        self.pre_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        
        # 调制模块（如果使用 AdaLN-Zero）
        if cond_dim is not None and cond_dim > 0:
            # Modulation: scale, shift, gate
            self.modulation = nn.Linear(cond_dim, d_model * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)
        else:
            self.modulation = None
        
        # 并行线性层1: 同时生成 QKV + MLP输入
        # QKV: d_model * 3, MLP: mlp_hidden_dim
        self.linear1 = nn.Linear(d_model, d_model * 3 + self.mlp_hidden_dim)
        
        # QK归一化（参考 Hunyuan3D-DiT）
        self.qk_norm = QKNorm(d_head)
        
        # MLP激活（使用 GELU with tanh approximation）
        self.mlp_act = GELU(approximate="tanh")
        
        # 并行线性层2: 融合 attention + MLP 输出
        self.linear2 = nn.Linear(d_model + self.mlp_hidden_dim, d_model)
        
        self.logger.info(
            f"ParallelSingleStreamBlock initialized: d_model={d_model}, "
            f"num_heads={num_heads}, mlp_hidden_dim={self.mlp_hidden_dim}, "
            f"cond_dim={cond_dim}"
        )

    def _log_stats(self, name: str, tensor: torch.Tensor):
        if not self.debug_log_stats or tensor is None:
            return
        try:
            self.logger.debug(
                f"[ParallelSingleStream] {name}: shape={tuple(tensor.shape)} min={tensor.min():.6f} max={tensor.max():.6f} mean={tensor.mean():.6f} std={(tensor.std() if tensor.numel()>1 else torch.tensor(0.)).item():.6f}"
            )
        except Exception:
            pass

    def _check_finite(self, name: str, tensor: torch.Tensor):
        if not self.debug_check_nan or tensor is None:
            return
        if not torch.isfinite(tensor).all():
            self.logger.error(f"[ParallelSingleStream] Non-finite values at {name}")
            raise RuntimeError(f"NaN/Inf detected in ParallelSingleStreamBlock at {name}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        cond_vector: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with parallel attention and MLP.
        
        Args:
            x: (B, L, d_model) - 拼接后的序列 [scene_tokens ⊕ grasp_tokens]
            cond_vector: (B, cond_dim) - 调制向量（time + scene + text融合）
            mask: (B, L) - attention mask，1=valid, 0=padding
        Returns:
            (B, L, d_model)
        """
        # 1. 调制和归一化
        if self.modulation is not None and cond_vector is not None:
            # 计算 modulation 参数
            mod_out = nn.functional.silu(cond_vector)  # SiLU激活
            mod_params = self.modulation(mod_out)  # (B, d_model * 3)
            scale, shift, gate = mod_params.chunk(3, dim=-1)
            
            # 扩展维度以匹配序列长度
            scale = scale.unsqueeze(1)  # (B, 1, d_model)
            shift = shift.unsqueeze(1)  # (B, 1, d_model)
            gate = gate.unsqueeze(1)    # (B, 1, d_model)
            
            # 调制：(1 + scale) * LN(x) + shift
            x_mod = (1 + scale) * self.pre_norm(x) + shift
        else:
            x_mod = self.pre_norm(x)
            gate = 1.0
        self._check_finite("x_mod", x_mod)
        self._log_stats("x_mod", x_mod)
        
        # 2. 并行生成 QKV + MLP输入（关键优化）
        qkv_mlp = self.linear1(x_mod)  # (B, L, d_model*3 + mlp_hidden_dim)
        qkv, mlp_in = torch.split(
            qkv_mlp, 
            [self.d_model * 3, self.mlp_hidden_dim], 
            dim=-1
        )
        
        # 3. 计算 attention
        q, k, v = rearrange(
            qkv, 
            "B L (K H D) -> K B H L D", 
            K=3, 
            H=self.num_heads
        )
        
        # QK归一化（提高训练稳定性）
        q, k = self.qk_norm(q, k, v)
        
        # Scaled dot-product attention
        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.to(dtype=q.dtype)
            # 将 0（padding）转换为大负数
            attn_mask = (1.0 - attn_mask) * -10000.0
        else:
            attn_mask = None
        
        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        attn_out = rearrange(attn_out, "B H L D -> B L (H D)")
        self._check_finite("attn_out", attn_out)
        self._log_stats("attn_out", attn_out)
        
        # 4. 计算MLP（并行，不等待 attention）
        mlp_out = self.mlp_act(mlp_in)
        self._check_finite("mlp_out", mlp_out)
        self._log_stats("mlp_out", mlp_out)
        
        # 5. 融合 attention + MLP（通过 concat + linear）
        combined = torch.cat([attn_out, mlp_out], dim=-1)
        output = self.linear2(combined)
        self._check_finite("output", output)
        self._log_stats("output", output)
        
        # 6. 门控残差
        return x + gate * output


def build_parallel_single_stream_block(
    d_model: int,
    num_heads: int,
    d_head: int,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    cond_dim: Optional[int] = None,
    use_flash_attention: bool = False,
) -> ParallelSingleStreamBlock:
    """
    构建并行单流块的工厂函数
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        d_head: 每个头的维度
        mlp_ratio: MLP 隐藏层维度倍数（默认4.0）
        dropout: Dropout 概率（暂未使用）
        cond_dim: 条件向量维度（AdaLN-Zero）
        use_flash_attention: 是否使用 flash attention（由 SDPA 自动处理）
    """
    return ParallelSingleStreamBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        cond_dim=cond_dim,
        use_flash_attention=use_flash_attention,
    )

