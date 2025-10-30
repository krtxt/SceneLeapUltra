"""
测试几何注意力偏置模块的正确性

测试内容:
1. GeometricAttentionBias模块的前向传播
2. 旋转表示转换的正确性（quat和r6d）
3. 几何特征计算的正确性
4. 与DiT模型的集成测试
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.geometric_attention_bias import (
    GeometricAttentionBias, 
    extract_scene_xyz
)


def test_quaternion_to_matrix():
    """测试四元数转旋转矩阵的正确性"""
    print("\n=== 测试四元数转旋转矩阵 ===")
    
    # 创建模块
    bias_module = GeometricAttentionBias(
        d_model=512,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance'],
        num_heads=8,
        rot_type='quat'
    )
    
    # 测试单位四元数（无旋转）
    identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # (x, y, z, w)
    R = bias_module._quaternion_to_matrix(identity_quat)
    
    expected = torch.eye(3).unsqueeze(0)
    assert torch.allclose(R, expected, atol=1e-5), f"Identity quaternion failed: {R}"
    print("✓ 单位四元数测试通过")
    
    # 测试绕Z轴旋转90度
    # quat = [0, 0, sin(45°), cos(45°)] for 90° rotation around Z
    quat_90z = torch.tensor([[0.0, 0.0, 0.7071068, 0.7071068]])
    R_90z = bias_module._quaternion_to_matrix(quat_90z)
    
    # 预期的旋转矩阵（绕Z轴逆时针90度）
    expected_90z = torch.tensor([[[0., -1., 0.],
                                   [1., 0., 0.],
                                   [0., 0., 1.]]])
    assert torch.allclose(R_90z, expected_90z, atol=1e-4), f"90° Z-rotation failed: {R_90z}"
    print("✓ 90度旋转测试通过")
    
    # 测试批量处理
    batch_quats = torch.randn(4, 5, 4)  # (B=4, N=5, 4)
    batch_quats = batch_quats / (batch_quats.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化
    R_batch = bias_module._quaternion_to_matrix(batch_quats)
    
    assert R_batch.shape == (4, 5, 3, 3), f"Batch shape incorrect: {R_batch.shape}"
    
    # 验证是正交矩阵（R^T R = I）
    R_transpose = R_batch.transpose(-2, -1)
    should_be_identity = torch.matmul(R_transpose, R_batch)
    identity_batch = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(4, 5, 3, 3)
    assert torch.allclose(should_be_identity, identity_batch, atol=1e-4), "Not orthogonal matrices"
    print("✓ 批量处理和正交性测试通过")


def test_r6d_to_matrix():
    """测试6D旋转表示转旋转矩阵的正确性"""
    print("\n=== 测试6D旋转表示转旋转矩阵 ===")
    
    bias_module = GeometricAttentionBias(
        d_model=512,
        hidden_dims=[128, 64],
        feature_types=['relative_pos'],
        num_heads=8,
        rot_type='r6d'
    )
    
    # 测试单位矩阵对应的6D表示
    identity_r6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    R = bias_module._r6d_to_matrix(identity_r6d)
    
    expected = torch.eye(3).unsqueeze(0)
    assert torch.allclose(R, expected, atol=1e-5), f"Identity r6d failed: {R}"
    print("✓ 单位矩阵6D表示测试通过")
    
    # 测试随机正交向量
    batch_size = 4
    num_grasps = 5
    r6d_batch = torch.randn(batch_size, num_grasps, 6)
    R_batch = bias_module._r6d_to_matrix(r6d_batch)
    
    assert R_batch.shape == (batch_size, num_grasps, 3, 3)
    
    # 验证正交性
    R_transpose = R_batch.transpose(-2, -1)
    should_be_identity = torch.matmul(R_transpose, R_batch)
    identity_batch = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(batch_size, num_grasps, 3, 3)
    assert torch.allclose(should_be_identity, identity_batch, atol=1e-4), "R6D matrices not orthogonal"
    print("✓ 6D批量处理和正交性测试通过")


def test_geometric_features():
    """测试几何特征计算的正确性"""
    print("\n=== 测试几何特征计算 ===")
    
    bias_module = GeometricAttentionBias(
        d_model=512,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance', 'direction'],
        num_heads=8,
        rot_type='quat'
    )
    
    # 简单场景：grasp在原点，场景点在(1, 0, 0)
    B, N_grasps, N_points = 2, 3, 4
    
    translations = torch.zeros(B, N_grasps, 3)
    rotation_matrices = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N_grasps, 3, 3)
    scene_points = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.],
                                  [1., 1., 1.]]]).expand(B, -1, -1)
    
    features = bias_module._compute_geometric_features(
        translations, rotation_matrices, scene_points
    )
    
    # 检查形状
    expected_dim = 3 + 1 + 3  # relative_pos(3) + distance(1) + direction(3)
    assert features.shape == (B, N_grasps, N_points, expected_dim)
    print(f"✓ 几何特征形状正确: {features.shape}")
    
    # 检查第一个点的相对位置（应该是[1, 0, 0]）
    assert torch.allclose(features[0, 0, 0, :3], torch.tensor([1., 0., 0.]), atol=1e-5)
    
    # 检查第一个点的距离（应该是1.0）
    assert torch.allclose(features[0, 0, 0, 3], torch.tensor(1.0), atol=1e-5)
    
    # 检查方向向量（应该是归一化的[1, 0, 0]）
    assert torch.allclose(features[0, 0, 0, 4:7], torch.tensor([1., 0., 0.]), atol=1e-5)
    print("✓ 几何特征数值正确")


def test_geometric_bias_forward():
    """测试GeometricAttentionBias的前向传播"""
    print("\n=== 测试GeometricAttentionBias前向传播 ===")
    
    B, N_grasps, N_points = 2, 4, 10
    d_model = 512
    num_heads = 8
    d_x_quat = 23  # 3 (translation) + 4 (quaternion) + 其他
    
    # 创建模块
    bias_module = GeometricAttentionBias(
        d_model=d_model,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance'],
        num_heads=num_heads,
        rot_type='quat'
    )
    
    # 创建输入
    grasp_tokens = torch.randn(B, N_grasps, d_model)
    scene_points = torch.randn(B, N_points, 3)
    
    # grasp_poses: 前3维是平移，接下来4维是四元数
    grasp_poses = torch.randn(B, N_grasps, d_x_quat)
    grasp_poses[:, :, :3] = torch.randn(B, N_grasps, 3)  # 平移
    grasp_poses[:, :, 3:7] = torch.randn(B, N_grasps, 4)  # 四元数（会被归一化）
    
    # 前向传播
    bias = bias_module(grasp_tokens, scene_points, grasp_poses)
    
    # 检查输出形状
    expected_shape = (B, num_heads, N_grasps, N_points)
    assert bias.shape == expected_shape, f"Output shape mismatch: {bias.shape} vs {expected_shape}"
    print(f"✓ 输出形状正确: {bias.shape}")
    
    # 检查数值范围（偏置不应过大）
    assert not torch.isnan(bias).any(), "Output contains NaN"
    assert not torch.isinf(bias).any(), "Output contains Inf"
    print(f"✓ 输出数值有效（无NaN/Inf）")
    print(f"  偏置统计: min={bias.min():.4f}, max={bias.max():.4f}, mean={bias.mean():.4f}")


def test_extract_scene_xyz():
    """测试extract_scene_xyz函数"""
    print("\n=== 测试extract_scene_xyz ===")
    
    B, N_points = 2, 100
    d_model = 512
    
    # 创建scene_context和data
    scene_context = torch.randn(B, N_points, d_model)
    scene_pc = torch.randn(B, N_points, 6)  # xyz + rgb
    data = {'scene_pc': scene_pc}
    
    # 提取xyz
    scene_xyz = extract_scene_xyz(scene_context, data)
    
    assert scene_xyz is not None
    assert scene_xyz.shape == (B, N_points, 3)
    assert torch.allclose(scene_xyz, scene_pc[:, :, :3])
    print("✓ extract_scene_xyz工作正常")
    
    # 测试缺失scene_pc的情况
    data_empty = {}
    scene_xyz_none = extract_scene_xyz(scene_context, data_empty)
    assert scene_xyz_none is None
    print("✓ 缺失scene_pc时返回None")


def test_integration_with_attention():
    """测试与attention机制的集成"""
    print("\n=== 测试与attention机制的集成 ===")
    
    B, N_grasps, N_points = 2, 4, 10
    d_model = 512
    num_heads = 8
    d_head = 64
    
    # 创建几何偏置模块
    bias_module = GeometricAttentionBias(
        d_model=d_model,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance'],
        num_heads=num_heads,
        rot_type='quat'
    )
    
    # 创建简化的attention计算
    grasp_tokens = torch.randn(B, N_grasps, d_model)
    scene_context = torch.randn(B, N_points, d_model)
    scene_points = torch.randn(B, N_points, 3)
    grasp_poses = torch.randn(B, N_grasps, 23)
    
    # 计算几何偏置
    bias = bias_module(grasp_tokens, scene_points, grasp_poses)
    
    # 简化的attention计算（仅计算scores）
    Q = torch.randn(B, num_heads, N_grasps, d_head)
    K = torch.randn(B, num_heads, N_points, d_head)
    
    # 计算attention scores
    scale = d_head ** -0.5
    scores = torch.einsum('bhid,bhjd->bhij', Q, K) * scale  # (B, num_heads, N_grasps, N_points)
    
    # 添加几何偏置
    scores_with_bias = scores + bias
    
    # 检查形状匹配
    assert scores_with_bias.shape == (B, num_heads, N_grasps, N_points)
    print("✓ 几何偏置可以正确添加到attention scores")
    
    # 计算attention weights
    attn_weights = torch.softmax(scores_with_bias, dim=-1)
    assert not torch.isnan(attn_weights).any()
    print("✓ 添加偏置后的softmax计算正常")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试几何注意力偏置模块")
    print("="*60)
    
    try:
        test_quaternion_to_matrix()
        test_r6d_to_matrix()
        test_geometric_features()
        test_geometric_bias_forward()
        test_extract_scene_xyz()
        test_integration_with_attention()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60 + "\n")
        return True
    
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

