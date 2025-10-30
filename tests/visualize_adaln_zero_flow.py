"""
可视化 AdaLN-Zero + Scene Pooling 的数据流

这个脚本展示了当启用 use_adaln_zero=True 和 use_scene_pooling=True 时，
数据在 DiT-FM 模型中的流动过程。
"""

import torch
import torch.nn as nn


class DemoAdaLNZero(nn.Module):
    """演示用的 AdaLN-Zero 实现"""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, d_model * 3)
        # 零初始化
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [B, seq_len, d_model]
        cond: [B, cond_dim]
        """
        # 生成调制参数
        modulation_params = self.modulation(cond)  # [B, d_model * 3]
        scale, shift, gate = modulation_params.chunk(3, dim=-1)
        
        # 扩展维度
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        
        # AdaLN-Zero 公式
        norm_x = self.layer_norm(x)
        modulated = (1 + scale) * norm_x + shift
        return x + gate * modulated


def pool_scene_features(scene_features: torch.Tensor, scene_mask: torch.Tensor = None):
    """场景特征池化"""
    if scene_mask is not None:
        if scene_mask.dim() == 3:
            scene_mask = scene_mask.squeeze(1)
        mask_expanded = scene_mask.unsqueeze(-1)
        masked_sum = (scene_features * mask_expanded).sum(dim=1)
        valid_counts = mask_expanded.sum(dim=1).clamp(min=1)
        pooled = masked_sum / valid_counts
    else:
        pooled = scene_features.mean(dim=1)
    return pooled


def visualize_data_flow():
    """可视化数据流"""
    print("=" * 80)
    print("AdaLN-Zero + Scene Pooling 数据流可视化")
    print("=" * 80)
    print()
    
    # 配置参数
    batch_size = 4
    num_grasps = 8
    num_points = 1024
    d_model = 512
    time_embed_dim = 1024
    
    print(f"配置参数:")
    print(f"  - batch_size: {batch_size}")
    print(f"  - num_grasps: {num_grasps}")
    print(f"  - num_points: {num_points}")
    print(f"  - d_model: {d_model}")
    print(f"  - time_embed_dim: {time_embed_dim}")
    print()
    
    # 步骤 1: 时间嵌入
    print("-" * 80)
    print("步骤 1: 时间嵌入")
    print("-" * 80)
    ts = torch.rand(batch_size)
    time_emb = torch.randn(batch_size, time_embed_dim)
    print(f"输入: ts.shape = {ts.shape}")
    print(f"输出: time_emb.shape = {time_emb.shape}")
    print(f"说明: 连续时间 t ∈ [0,1] 通过 Fourier 特征 + MLP 嵌入到高维空间")
    print()
    
    # 步骤 2: 场景特征提取
    print("-" * 80)
    print("步骤 2: 场景特征提取")
    print("-" * 80)
    scene_pc = torch.randn(batch_size, num_points, 3)  # 点云坐标
    scene_context = torch.randn(batch_size, num_points, d_model)  # 提取后的特征
    scene_mask = torch.ones(batch_size, num_points)  # mask
    print(f"输入: scene_pc.shape = {scene_pc.shape}")
    print(f"输出: scene_context.shape = {scene_context.shape}")
    print(f"     scene_mask.shape = {scene_mask.shape}")
    print(f"说明: 通过 PointNet2/PTv3 提取场景点云特征")
    print()
    
    # 步骤 3: 场景特征池化 (关键步骤)
    print("-" * 80)
    print("步骤 3: 场景特征池化 (use_scene_pooling=True)")
    print("-" * 80)
    scene_pooled = pool_scene_features(scene_context, scene_mask)
    print(f"输入: scene_context.shape = {scene_context.shape}")
    print(f"     scene_mask.shape = {scene_mask.shape}")
    print(f"输出: scene_pooled.shape = {scene_pooled.shape}")
    print(f"说明: 通过带 mask 的均值池化将整个场景压缩为全局特征向量")
    print(f"      公式: pooled[i] = Σ_j (features[i,j,:] * mask[i,j]) / Σ_j mask[i,j]")
    print()
    
    # 步骤 4: 文本特征提取
    print("-" * 80)
    print("步骤 4: 文本特征提取")
    print("-" * 80)
    text_context = torch.randn(batch_size, 1, d_model)
    text_pooled = text_context.squeeze(1)
    print(f"输入: positive_prompt (字符串列表)")
    print(f"输出: text_context.shape = {text_context.shape}")
    print(f"     text_pooled.shape = {text_pooled.shape}")
    print(f"说明: 通过 CLIP/T5 编码文本提示")
    print()
    
    # 步骤 5: 多条件向量融合 (关键步骤)
    print("-" * 80)
    print("步骤 5: 多条件向量融合 (use_adaln_zero=True)")
    print("-" * 80)
    cond_vector = torch.cat([time_emb, scene_pooled, text_pooled], dim=-1)
    cond_dim = cond_vector.shape[-1]
    print(f"拼接条件:")
    print(f"  1. time_emb.shape = {time_emb.shape}  (时间步)")
    print(f"  2. scene_pooled.shape = {scene_pooled.shape}  (场景全局特征)")
    print(f"  3. text_pooled.shape = {text_pooled.shape}  (文本特征)")
    print(f"输出: cond_vector.shape = {cond_vector.shape}")
    print(f"     cond_dim = {cond_dim}")
    print(f"说明: 将三种条件融合为统一的向量，作为 AdaLN-Zero 的输入")
    print()
    
    # 步骤 6: 抓取姿态 Tokenization
    print("-" * 80)
    print("步骤 6: 抓取姿态 Tokenization")
    print("-" * 80)
    x_t = torch.randn(batch_size, num_grasps, 23)  # 噪声抓取姿态 (quat 模式)
    grasp_tokens = torch.randn(batch_size, num_grasps, d_model)
    print(f"输入: x_t.shape = {x_t.shape}  (噪声抓取姿态)")
    print(f"输出: grasp_tokens.shape = {grasp_tokens.shape}")
    print(f"说明: 通过线性层 + LayerNorm 将抓取姿态投影到模型维度")
    print()
    
    # 步骤 7: AdaLN-Zero 调制
    print("-" * 80)
    print("步骤 7: AdaLN-Zero 调制 (每个 DiT Block 中)")
    print("-" * 80)
    adaln_zero = DemoAdaLNZero(d_model, cond_dim)
    x = grasp_tokens
    
    print(f"输入:")
    print(f"  - x.shape = {x.shape}  (特征)")
    print(f"  - cond_vector.shape = {cond_vector.shape}  (条件)")
    print()
    
    # 模拟调制过程
    modulation_params = adaln_zero.modulation(cond_vector)
    scale, shift, gate = modulation_params.chunk(3, dim=-1)
    print(f"调制参数生成:")
    print(f"  modulation_params.shape = {modulation_params.shape}")
    print(f"  → scale.shape = {scale.shape}")
    print(f"  → shift.shape = {shift.shape}")
    print(f"  → gate.shape = {gate.shape}")
    print()
    
    scale = scale.unsqueeze(1)
    shift = shift.unsqueeze(1)
    gate = gate.unsqueeze(1)
    print(f"扩展维度以匹配序列长度:")
    print(f"  scale.shape = {scale.shape}")
    print(f"  shift.shape = {shift.shape}")
    print(f"  gate.shape = {gate.shape}")
    print()
    
    norm_x = adaln_zero.layer_norm(x)
    modulated = (1 + scale) * norm_x + shift
    output = x + gate * modulated
    print(f"AdaLN-Zero 公式应用:")
    print(f"  1. norm_x = LayerNorm(x)")
    print(f"     norm_x.shape = {norm_x.shape}")
    print(f"  2. modulated = (1 + scale) * norm_x + shift")
    print(f"     modulated.shape = {modulated.shape}")
    print(f"  3. output = x + gate * modulated")
    print(f"     output.shape = {output.shape}")
    print()
    
    # 检查零初始化效果
    print(f"零初始化验证:")
    print(f"  modulation.weight.norm() = {adaln_zero.modulation.weight.norm().item():.6f}")
    print(f"  modulation.bias.norm() = {adaln_zero.modulation.bias.norm().item():.6f}")
    print(f"  → 在训练初期，output ≈ x (恒等映射)")
    print()
    
    # 步骤 8: 完整的 DiT Block
    print("-" * 80)
    print("步骤 8: 完整的 DiT Block 流程")
    print("-" * 80)
    print(f"每个 DiT Block 包含 4 个处理步骤，每步都使用 AdaLN-Zero:")
    print()
    print(f"  1. Self-Attention:")
    print(f"     norm_x = norm1(x, cond_vector)")
    print(f"     attn_out = self_attention(norm_x)")
    print(f"     x = x + attn_out")
    print()
    print(f"  2. Scene Cross-Attention:")
    print(f"     norm_x = norm2(x, cond_vector)")
    print(f"     scene_attn = scene_cross_attention(norm_x, scene_context)")
    print(f"     x = x + scene_attn")
    print()
    print(f"  3. Text Cross-Attention:")
    print(f"     norm_x = norm3(x, cond_vector)")
    print(f"     text_attn = text_cross_attention(norm_x, text_context)")
    print(f"     x = x + text_attn")
    print()
    print(f"  4. Feed-Forward:")
    print(f"     norm_x = norm4(x, cond_vector)")
    print(f"     ff_out = feed_forward(norm_x)")
    print(f"     x = x + ff_out")
    print()
    print(f"说明: 与原版 AdaptiveLayerNorm 的区别在于：")
    print(f"      - 原版: norm(x, time_emb) → 仅使用时间步")
    print(f"      - AdaLN-Zero: norm(x, cond_vector) → 融合时间步+场景+文本")
    print()
    
    # 总结
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print(f"关键尺寸变化:")
    print(f"  1. 场景: [B, N, D] → pool → [B, D]")
    print(f"  2. 条件融合: [B, T] + [B, D] + [B, D] → concat → [B, T+2D]")
    print(f"               (时间)  (场景)  (文本)              (条件向量)")
    print(f"  3. 调制参数: [B, T+2D] → linear → [B, 3D]")
    print(f"                                     (scale, shift, gate)")
    print()
    print(f"优势:")
    print(f"  ✓ 更丰富的条件信息（时间步 + 场景全局 + 文本语义）")
    print(f"  ✓ 零初始化确保训练稳定性")
    print(f"  ✓ 门控机制允许动态调整调制强度")
    print()
    print(f"适用场景:")
    print(f"  - 场景全局信息对抓取很重要（如物体分布、场景复杂度）")
    print(f"  - 需要强大的多模态条件融合能力")
    print(f"  - 对训练稳定性有较高要求")
    print()


if __name__ == "__main__":
    visualize_data_flow()

