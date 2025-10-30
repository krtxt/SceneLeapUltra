"""
测试全局-局部场景条件模块

验证 GlobalScenePool 和 LocalSelector 的基本功能。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.scene_pool import GlobalScenePool
from models.decoder.local_selector import KNNSelector, BallQuerySelector


def test_global_scene_pool():
    """测试 GlobalScenePool 基本功能"""
    print("\n=== 测试 GlobalScenePool ===")
    
    # 参数
    B, N, d_model = 2, 1024, 512
    K = 128
    
    # 创建模块
    pool = GlobalScenePool(
        d_model=d_model,
        num_latents=K,
        num_layers=1,
        num_heads=8,
        d_head=64,
    )
    
    # 输入
    scene_context = torch.randn(B, N, d_model)
    scene_mask = torch.ones(B, N)
    
    # 前向传播
    latent_global = pool(scene_context, scene_mask)
    
    # 验证
    assert latent_global.shape == (B, K, d_model), f"期望形状 {(B, K, d_model)}, 实际 {latent_global.shape}"
    assert not torch.isnan(latent_global).any(), "输出包含 NaN"
    assert not torch.isinf(latent_global).any(), "输出包含 Inf"
    
    print(f"✓ 输出形状: {latent_global.shape}")
    print(f"✓ 输出范围: [{latent_global.min():.4f}, {latent_global.max():.4f}]")
    print("✓ GlobalScenePool 测试通过")


def test_knn_selector():
    """测试 KNNSelector 基本功能"""
    print("\n=== 测试 KNNSelector ===")
    
    # 参数
    B, G, N, k = 2, 50, 1024, 32
    
    # 创建模块
    selector = KNNSelector(k=k)
    
    # 输入
    grasp_translations = torch.randn(B, G, 3)
    scene_xyz = torch.randn(B, N, 3)
    scene_mask = torch.ones(B, N)
    
    # 前向传播
    local_indices, local_mask = selector(grasp_translations, scene_xyz, scene_mask)
    
    # 验证
    assert local_indices.shape == (B, G, k), f"期望形状 {(B, G, k)}, 实际 {local_indices.shape}"
    assert local_mask.shape == (B, G, k), f"期望形状 {(B, G, k)}, 实际 {local_mask.shape}"
    assert (local_indices >= 0).all() and (local_indices < N).all(), "索引超出范围"
    assert ((local_mask == 0) | (local_mask == 1)).all(), "mask 应该只包含 0 或 1"
    
    print(f"✓ 索引形状: {local_indices.shape}")
    print(f"✓ 索引范围: [{local_indices.min()}, {local_indices.max()}]")
    print(f"✓ 有效邻居比例: {local_mask.mean():.2%}")
    print("✓ KNNSelector 测试通过")


def test_ball_query_selector():
    """测试 BallQuerySelector 基本功能"""
    print("\n=== 测试 BallQuerySelector ===")
    
    # 参数
    B, G, N = 2, 50, 1024
    radius = 0.1
    max_samples = 32
    
    # 创建模块
    selector = BallQuerySelector(radius=radius, max_samples=max_samples)
    
    # 输入：将抓取点放在场景点附近，以确保能找到邻居
    scene_xyz = torch.randn(B, N, 3)
    # 从场景点中随机选择 G 个作为抓取点的基础
    grasp_base_indices = torch.randint(0, N, (B, G))
    grasp_translations = torch.stack([
        scene_xyz[b, grasp_base_indices[b]] + torch.randn(G, 3) * 0.05
        for b in range(B)
    ])
    scene_mask = torch.ones(B, N)
    
    # 前向传播
    local_indices, local_mask = selector(grasp_translations, scene_xyz, scene_mask)
    
    # 验证
    assert local_indices.shape == (B, G, max_samples), \
        f"期望形状 {(B, G, max_samples)}, 实际 {local_indices.shape}"
    assert local_mask.shape == (B, G, max_samples), \
        f"期望形状 {(B, G, max_samples)}, 实际 {local_mask.shape}"
    assert (local_indices >= 0).all() and (local_indices < N).all(), "索引超出范围"
    assert ((local_mask == 0) | (local_mask == 1)).all(), "mask 应该只包含 0 或 1"
    
    print(f"✓ 索引形状: {local_indices.shape}")
    print(f"✓ 平均邻居数: {local_mask.sum(dim=-1).float().mean():.1f} / {max_samples}")
    print(f"✓ 有效邻居比例: {local_mask.mean():.2%}")
    print("✓ BallQuerySelector 测试通过")


def test_integration():
    """测试全局-局部条件的集成流程"""
    print("\n=== 测试集成流程 ===")
    
    # 参数
    B, G, N, d_model = 2, 50, 1024, 512
    K, k = 128, 32
    
    # 模拟场景特征
    scene_context = torch.randn(B, N, d_model)
    scene_xyz = torch.randn(B, N, 3)
    scene_mask = torch.ones(B, N)
    
    # 模拟抓取姿态（只使用前 3 维的平移）
    grasp_translations = torch.randn(B, G, 3)
    
    # 1. 全局阶段
    global_pool = GlobalScenePool(
        d_model=d_model,
        num_latents=K,
        num_layers=1,
        num_heads=8,
        d_head=64,
    )
    latent_global = global_pool(scene_context, scene_mask)
    
    # 2. 局部阶段
    local_selector = KNNSelector(k=k)
    local_indices, local_mask = local_selector(grasp_translations, scene_xyz, scene_mask)
    
    # 3. 构建全局-局部上下文（模拟 DiTBlock 中的操作）
    # Gather 局部特征
    local_indices_expanded = local_indices.unsqueeze(-1).expand(B, G, k, d_model)
    scene_context_expanded = scene_context.unsqueeze(1).expand(B, G, N, d_model)
    local_features = torch.gather(scene_context_expanded, 2, local_indices_expanded)
    
    # 展平并拼接
    local_features_flat = local_features.view(B, G * k, d_model)
    scene_context_combined = torch.cat([latent_global, local_features_flat], dim=1)
    
    # 验证
    expected_length = K + G * k
    assert scene_context_combined.shape == (B, expected_length, d_model), \
        f"期望形状 {(B, expected_length, d_model)}, 实际 {scene_context_combined.shape}"
    
    print(f"✓ 全局特征形状: {latent_global.shape}")
    print(f"✓ 局部特征形状: {local_features_flat.shape}")
    print(f"✓ 组合特征形状: {scene_context_combined.shape}")
    print(f"✓ 压缩比: {N} -> {expected_length} ({expected_length/N:.1%})")
    print("✓ 集成流程测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("全局-局部场景条件模块测试")
    print("=" * 60)
    
    try:
        test_global_scene_pool()
        test_knn_selector()
        test_ball_query_selector()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

