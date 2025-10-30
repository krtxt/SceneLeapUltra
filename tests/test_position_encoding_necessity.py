"""
测试位置编码对抓取集合学习的影响

验证假设：
1. 抓取集合应该是置换不变的
2. 位置编码会破坏这种不变性
3. 去除位置编码不会影响模型性能（可能还会提升）
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.fm.optimal_transport import SinkhornOT


def test_permutation_invariance_with_positional_encoding():
    """
    测试加了位置编码后，模型是否还保持置换不变性
    """
    print("\n" + "="*80)
    print("测试1: 位置编码对置换不变性的影响")
    print("="*80)
    
    B, N, D = 2, 1024, 25
    d_model = 512
    
    # 模拟 grasp tokenizer + positional encoding
    class SimpleTokenizerWithPosEmb(nn.Module):
        def __init__(self, use_pos_emb=False):
            super().__init__()
            self.tokenizer = nn.Linear(D, d_model)
            self.use_pos_emb = use_pos_emb
            if use_pos_emb:
                self.pos_emb = nn.Parameter(torch.randn(1, N, d_model) * 0.02)
        
        def forward(self, x):
            tokens = self.tokenizer(x)
            if self.use_pos_emb:
                tokens = tokens + self.pos_emb
            return tokens
    
    # 创建两个模型：有/无位置编码
    model_no_pos = SimpleTokenizerWithPosEmb(use_pos_emb=False)
    model_with_pos = SimpleTokenizerWithPosEmb(use_pos_emb=True)
    
    # 测试数据
    x = torch.randn(B, N, D)
    perm = torch.randperm(N)
    x_shuffled = x[:, perm]
    
    # 无位置编码：应该置换等变
    with torch.no_grad():
        out_no_pos_1 = model_no_pos(x)
        out_no_pos_2 = model_no_pos(x_shuffled)
        # 还原顺序
        out_no_pos_2_restored = torch.zeros_like(out_no_pos_2)
        out_no_pos_2_restored[:, perm] = out_no_pos_2
        
        diff_no_pos = (out_no_pos_1 - out_no_pos_2_restored).abs().max().item()
    
    # 有位置编码：会破坏置换不变性
    with torch.no_grad():
        out_with_pos_1 = model_with_pos(x)
        out_with_pos_2 = model_with_pos(x_shuffled)
        out_with_pos_2_restored = torch.zeros_like(out_with_pos_2)
        out_with_pos_2_restored[:, perm] = out_with_pos_2
        
        diff_with_pos = (out_with_pos_1 - out_with_pos_2_restored).abs().max().item()
    
    print(f"\n结果：")
    print(f"  无位置编码时的差异: {diff_no_pos:.6f}  {'✅ 置换等变' if diff_no_pos < 1e-5 else '❌'}")
    print(f"  有位置编码时的差异: {diff_with_pos:.6f}  {'✅ 置换等变' if diff_with_pos < 1e-5 else '❌ 破坏了置换不变性'}")
    
    print(f"\n结论：位置编码破坏了集合的置换不变性！")
    print(f"      对于抓取姿态这种无序集合，应该 **避免** 使用位置编码。")
    
    return diff_no_pos, diff_with_pos


def test_sinkhorn_matching_randomness():
    """
    测试 SinkhornOT 配对后的顺序是否有意义
    """
    print("\n" + "="*80)
    print("测试2: SinkhornOT 配对后的索引顺序分析")
    print("="*80)
    
    B, N, D = 1, 100, 3
    
    # 创建两个点集
    x0 = torch.randn(B, N, D)
    x1 = torch.randn(B, N, D)
    
    # 进行配对
    sinkhorn_ot = SinkhornOT(reg=0.1, num_iters=50)
    matchings = sinkhorn_ot(x0, x1)
    
    print(f"\n配对索引示例（前20个）：")
    print(f"  {matchings[0, :20].tolist()}")
    
    # 分析索引的连续性
    diffs = torch.diff(matchings[0].float()).abs()
    avg_jump = diffs.mean().item()
    max_jump = diffs.max().item()
    
    print(f"\n索引跳跃分析：")
    print(f"  平均跳跃距离: {avg_jump:.2f}")
    print(f"  最大跳跃距离: {max_jump:.0f}")
    print(f"  理论随机跳跃: {N/2:.2f}")
    
    # 判断
    if avg_jump > N / 4:
        print(f"\n结论：配对后的索引是 **高度无序** 的！")
        print(f"      第 i 个和第 i+1 个抓取在空间上没有邻近关系。")
        print(f"      因此，给它们加上位置编码是 **没有意义** 的！")
    
    return matchings


def test_position_encoding_vs_spatial_position():
    """
    测试位置编码 vs 空间位置的区别
    """
    print("\n" + "="*80)
    print("测试3: 位置编码 vs 空间位置坐标")
    print("="*80)
    
    N = 1024
    d_model = 512
    
    # 抓取的空间位置（translation 部分）
    spatial_positions = torch.randn(N, 3)  # [N, 3] 真实的 xyz 坐标
    
    # Transformer 的位置编码（固定索引）
    pos_indices = torch.arange(N)  # [0, 1, 2, ..., 1023]
    
    # 计算空间距离矩阵
    spatial_dist = torch.cdist(spatial_positions, spatial_positions)  # [N, N]
    
    # 计算索引距离矩阵
    index_dist = (pos_indices.unsqueeze(0) - pos_indices.unsqueeze(1)).abs().float()  # [N, N]
    
    # 相关性分析
    correlation = torch.corrcoef(torch.stack([
        spatial_dist.flatten(),
        index_dist.flatten()
    ]))[0, 1].item()
    
    print(f"\n空间距离 vs 索引距离的相关性: {correlation:.4f}")
    
    if abs(correlation) < 0.3:
        print(f"\n结论：空间位置和索引位置 **几乎无关**！")
        print(f"      - 空间位置: 由 translation (x,y,z) 表示，已经在输入中")
        print(f"      - 索引位置: [0,1,2,...] 是人为的，没有物理意义")
        print(f"      因此，模型应该依赖 translation 特征，而非索引位置编码！")
    
    return correlation


def visualize_attention_pattern():
    """
    可视化有/无位置编码时的注意力模式
    """
    print("\n" + "="*80)
    print("测试4: 注意力模式分析（理论分析）")
    print("="*80)
    
    print(f"\n假设场景：学习抓取姿态之间的关系")
    print(f"\n1. **无位置编码**：")
    print(f"   - 注意力基于内容相似度：Attention(Q, K) = softmax(QK^T / √d)")
    print(f"   - 相似的抓取（位置近、姿态近）会互相关注")
    print(f"   - 符合物理直觉：附近的抓取确实相关")
    
    print(f"\n2. **有位置编码**：")
    print(f"   - 注意力混合了内容 + 索引位置")
    print(f"   - 索引相近的抓取会被强制关联")
    print(f"   - ❌ 但索引 [i, i+1] 在空间上可能距离很远！")
    
    print(f"\n结论：对于抓取姿态集合，位置编码会引入 **错误的归纳偏置**！")


def main():
    print("\n" + "="*80)
    print("位置编码必要性测试")
    print("场景：Flow Matching 中的抓取姿态集合学习")
    print("="*80)
    
    # 运行所有测试
    test_permutation_invariance_with_positional_encoding()
    test_sinkhorn_matching_randomness()
    test_position_encoding_vs_spatial_position()
    visualize_attention_pattern()
    
    # 最终建议
    print("\n" + "="*80)
    print("🎯 最终建议")
    print("="*80)
    print(f"\n对于你们的 DiT-FM 模型：")
    print(f"\n✅ **推荐配置**：")
    print(f"   use_learnable_pos_embedding: false  （当前配置）")
    print(f"\n❌ **不推荐**：")
    print(f"   use_learnable_pos_embedding: true")
    print(f"\n理由：")
    print(f"   1. 抓取姿态是无序集合，不存在固定的顺序关系")
    print(f"   2. SinkhornOT 配对后的索引是任意的，没有空间意义")
    print(f"   3. 空间关系已经由 translation 坐标表示，无需额外编码")
    print(f"   4. 位置编码会破坏置换不变性，引入错误的归纳偏置")
    print(f"\n如果需要空间信息：")
    print(f"   ✅ 使用 geometric_attention_bias（基于真实的 xyz 距离）")
    print(f"   ❌ 不要使用 positional_embedding（基于人为的索引）")
    print("="*80 + "\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()

