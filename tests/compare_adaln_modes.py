"""
对比不同归一化模式的行为

这个脚本对比了三种归一化模式：
1. 普通 LayerNorm
2. AdaptiveLayerNorm (仅时间步)
3. AdaLN-Zero (时间步 + 场景 + 文本)
"""

import torch
import torch.nn as nn


class PlainLayerNorm(nn.Module):
    """普通 LayerNorm"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.norm(x)


class AdaptiveLayerNorm(nn.Module):
    """AdaptiveLayerNorm (仅时间步调制)"""
    def __init__(self, d_model: int, time_embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift = nn.Linear(time_embed_dim, d_model * 2)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        x = self.layer_norm(x)
        scale, shift = self.scale_shift(time_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        return x * (1 + scale) + shift


class AdaLNZero(nn.Module):
    """AdaLN-Zero (多条件融合 + 门控 + 零初始化)"""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, d_model * 3)
        # 零初始化
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        modulation_params = self.modulation(cond)
        scale, shift, gate = modulation_params.chunk(3, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        norm_x = self.layer_norm(x)
        modulated = (1 + scale) * norm_x + shift
        return x + gate * modulated


def compare_modes():
    """对比三种归一化模式"""
    print("=" * 80)
    print("归一化模式对比")
    print("=" * 80)
    print()
    
    # 配置
    batch_size = 4
    seq_len = 8
    d_model = 512
    time_embed_dim = 1024
    cond_dim = 2048
    
    # 输入数据
    x = torch.randn(batch_size, seq_len, d_model)
    time_emb = torch.randn(batch_size, time_embed_dim)
    scene_pooled = torch.randn(batch_size, d_model)
    text_pooled = torch.randn(batch_size, d_model)
    cond_vector = torch.cat([time_emb, scene_pooled, text_pooled], dim=-1)
    
    print(f"输入数据:")
    print(f"  - x.shape = {x.shape}")
    print(f"  - time_emb.shape = {time_emb.shape}")
    print(f"  - scene_pooled.shape = {scene_pooled.shape}")
    print(f"  - text_pooled.shape = {text_pooled.shape}")
    print(f"  - cond_vector.shape = {cond_vector.shape}")
    print()
    
    # ========================================================================
    # 模式 1: 普通 LayerNorm
    # ========================================================================
    print("-" * 80)
    print("模式 1: 普通 LayerNorm")
    print("-" * 80)
    norm1 = PlainLayerNorm(d_model)
    out1 = norm1(x)
    
    print(f"使用条件: 无")
    print(f"输出: {out1.shape}")
    print(f"参数量: {sum(p.numel() for p in norm1.parameters()):,}")
    print(f"公式: output = LayerNorm(x)")
    print(f"特点:")
    print(f"  - 最简单的归一化")
    print(f"  - 无条件依赖")
    print(f"  - 适用于不需要时间或上下文调制的场景")
    print()
    
    # ========================================================================
    # 模式 2: AdaptiveLayerNorm (原版 DiT 使用)
    # ========================================================================
    print("-" * 80)
    print("模式 2: AdaptiveLayerNorm (原版 DiT)")
    print("-" * 80)
    norm2 = AdaptiveLayerNorm(d_model, time_embed_dim)
    out2 = norm2(x, time_emb)
    
    print(f"使用条件: time_emb [B, {time_embed_dim}]")
    print(f"输出: {out2.shape}")
    print(f"参数量: {sum(p.numel() for p in norm2.parameters()):,}")
    print(f"公式: output = (1 + scale(time)) * LayerNorm(x) + shift(time)")
    print(f"特点:")
    print(f"  - 仅使用时间步进行调制")
    print(f"  - scale 和 shift 由时间嵌入生成")
    print(f"  - 无门控机制")
    print(f"  - 无零初始化，训练初期可能不稳定")
    print()
    
    # ========================================================================
    # 模式 3: AdaLN-Zero (增强版)
    # ========================================================================
    print("-" * 80)
    print("模式 3: AdaLN-Zero (增强版)")
    print("-" * 80)
    norm3 = AdaLNZero(d_model, cond_dim)
    out3 = norm3(x, cond_vector)
    
    print(f"使用条件: cond_vector [B, {cond_dim}]")
    print(f"  = concat([time_emb, scene_pooled, text_pooled])")
    print(f"  = concat([B, {time_embed_dim}], [B, {d_model}], [B, {d_model}])")
    print(f"输出: {out3.shape}")
    print(f"参数量: {sum(p.numel() for p in norm3.parameters()):,}")
    print(f"公式: output = x + gate * ((1 + scale) * LayerNorm(x) + shift)")
    print(f"      其中 [scale, shift, gate] = MLP(cond_vector)")
    print(f"特点:")
    print(f"  - 融合多个条件源（时间步 + 场景 + 文本）")
    print(f"  - 增加门控机制 (gate)，允许动态调整调制强度")
    print(f"  - 零初始化权重，训练初期 output ≈ x (恒等映射)")
    print(f"  - 更稳定的训练过程")
    print()
    
    # ========================================================================
    # 零初始化验证
    # ========================================================================
    print("-" * 80)
    print("零初始化验证 (AdaLN-Zero)")
    print("-" * 80)
    
    # 检查权重和偏置
    weight_norm = norm3.modulation.weight.norm().item()
    bias_norm = norm3.modulation.bias.norm().item()
    print(f"modulation.weight.norm() = {weight_norm:.8f}")
    print(f"modulation.bias.norm() = {bias_norm:.8f}")
    print()
    
    # 验证初始化时的输出
    with torch.no_grad():
        diff = (out3 - x).norm()
        relative_diff = diff / x.norm()
    print(f"||output - input||_2 = {diff.item():.8f}")
    print(f"相对差异 = {relative_diff.item():.8f}")
    print(f"说明: 由于零初始化，训练初期 output ≈ x")
    print()
    
    # ========================================================================
    # 参数量对比
    # ========================================================================
    print("-" * 80)
    print("参数量对比")
    print("-" * 80)
    
    params1 = sum(p.numel() for p in norm1.parameters())
    params2 = sum(p.numel() for p in norm2.parameters())
    params3 = sum(p.numel() for p in norm3.parameters())
    
    print(f"{'模式':<30} {'参数量':>15} {'相对比例':>15}")
    print(f"{'-' * 60}")
    print(f"{'1. 普通 LayerNorm':<30} {params1:>15,} {params1/params1:>14.2f}x")
    print(f"{'2. AdaptiveLayerNorm':<30} {params2:>15,} {params2/params1:>14.2f}x")
    print(f"{'3. AdaLN-Zero':<30} {params3:>15,} {params3/params1:>14.2f}x")
    print()
    
    print(f"说明:")
    print(f"  - AdaLN-Zero 参数量最大，因为 cond_dim 更大")
    print(f"  - 但在整个模型中占比很小（相比注意力层）")
    print(f"  - 额外的参数换来更强的条件建模能力")
    print()
    
    # ========================================================================
    # 使用建议
    # ========================================================================
    print("=" * 80)
    print("使用建议")
    print("=" * 80)
    print()
    
    print(f"【模式 1: 普通 LayerNorm】")
    print(f"  适用场景:")
    print(f"    - 不需要时间或上下文信息")
    print(f"    - 追求最小参数量")
    print(f"  配置: use_adaptive_norm=False, use_adaln_zero=False")
    print()
    
    print(f"【模式 2: AdaptiveLayerNorm】")
    print(f"  适用场景:")
    print(f"    - 需要时间步条件")
    print(f"    - 场景/文本信息通过 cross-attention 足够")
    print(f"    - 追求中等参数量")
    print(f"  配置: use_adaptive_norm=True, use_adaln_zero=False")
    print()
    
    print(f"【模式 3: AdaLN-Zero】 ⭐ 推荐")
    print(f"  适用场景:")
    print(f"    - 场景全局信息很重要（需要池化特征）")
    print(f"    - 需要强大的多模态条件融合")
    print(f"    - 对训练稳定性有高要求")
    print(f"  配置: use_adaptive_norm=True, use_adaln_zero=True, use_scene_pooling=True")
    print()
    
    print(f"性能影响:")
    print(f"  - 计算开销：AdaLN-Zero 略高于 AdaptiveLayerNorm")
    print(f"  - 内存开销：主要来自 cond_vector 的存储和传播")
    print(f"  - 训练速度：影响很小（< 5%），因为归一化不是瓶颈")
    print(f"  - 推理速度：影响可忽略")
    print()


if __name__ == "__main__":
    compare_modes()

