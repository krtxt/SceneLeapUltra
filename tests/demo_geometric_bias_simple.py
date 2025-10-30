"""
几何注意力偏置功能简化演示脚本

展示核心几何偏置模块的功能，不依赖完整的DiT模型配置。
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.geometric_attention_bias import GeometricAttentionBias


def demo_geometric_bias_module():
    """演示GeometricAttentionBias模块的基本功能"""
    print("\n" + "="*60)
    print("演示几何注意力偏置模块")
    print("="*60 + "\n")
    
    # 参数设置
    B, N_grasps, N_points = 2, 4, 100
    d_model = 512
    num_heads = 8
    d_x_quat = 23
    
    print("参数设置:")
    print(f"  Batch size: {B}")
    print(f"  Number of grasps: {N_grasps}")
    print(f"  Number of scene points: {N_points}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of attention heads: {num_heads}\n")
    
    # 创建几何偏置模块（使用quat旋转表示）
    print("创建几何偏置模块（quat旋转表示）...")
    bias_module_quat = GeometricAttentionBias(
        d_model=d_model,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance'],
        num_heads=num_heads,
        rot_type='quat'
    )
    print("✓ 模块创建成功\n")
    
    # 创建输入数据
    print("创建测试数据...")
    grasp_tokens = torch.randn(B, N_grasps, d_model)
    scene_points = torch.randn(B, N_points, 3)
    
    # grasp_poses: [translation(3) + quaternion(4) + others(16)]
    grasp_poses = torch.randn(B, N_grasps, d_x_quat)
    print("✓ 测试数据创建成功\n")
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        bias = bias_module_quat(grasp_tokens, scene_points, grasp_poses)
    
    print(f"✓ 前向传播成功")
    print(f"  输入 grasp_tokens shape: {grasp_tokens.shape}")
    print(f"  输入 scene_points shape: {scene_points.shape}")
    print(f"  输入 grasp_poses shape: {grasp_poses.shape}")
    print(f"  输出 bias shape: {bias.shape}")
    print(f"  输出统计: min={bias.min():.4f}, max={bias.max():.4f}, mean={bias.mean():.4f}\n")
    
    # 创建几何偏置模块（使用r6d旋转表示）
    print("创建几何偏置模块（r6d旋转表示）...")
    bias_module_r6d = GeometricAttentionBias(
        d_model=d_model,
        hidden_dims=[128, 64],
        feature_types=['relative_pos', 'distance', 'direction'],
        num_heads=num_heads,
        rot_type='r6d'
    )
    print("✓ 模块创建成功\n")
    
    # 使用r6d格式的grasp poses
    d_x_r6d = 25  # 3 (translation) + 6 (r6d) + 16 (others)
    grasp_poses_r6d = torch.randn(B, N_grasps, d_x_r6d)
    
    print("执行前向传播（r6d）...")
    with torch.no_grad():
        bias_r6d = bias_module_r6d(grasp_tokens, scene_points, grasp_poses_r6d)
    
    print(f"✓ 前向传播成功")
    print(f"  输出 bias shape: {bias_r6d.shape}")
    print(f"  输出统计: min={bias_r6d.min():.4f}, max={bias_r6d.max():.4f}, mean={bias_r6d.mean():.4f}\n")


def demo_attention_integration():
    """演示与attention机制的集成"""
    print("="*60)
    print("演示与Attention机制的集成")
    print("="*60 + "\n")
    
    B, N_grasps, N_points = 2, 4, 100
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
    
    # 创建数据
    grasp_tokens = torch.randn(B, N_grasps, d_model)
    scene_points = torch.randn(B, N_points, 3)
    grasp_poses = torch.randn(B, N_grasps, 23)
    
    # 计算几何偏置
    print("计算几何偏置...")
    with torch.no_grad():
        bias = bias_module(grasp_tokens, scene_points, grasp_poses)
    print(f"✓ 几何偏置 shape: {bias.shape}\n")
    
    # 模拟attention计算
    print("模拟Attention计算...")
    Q = torch.randn(B, num_heads, N_grasps, d_head)
    K = torch.randn(B, num_heads, N_points, d_head)
    V = torch.randn(B, num_heads, N_points, d_head)
    
    scale = d_head ** -0.5
    
    # 计算标准attention scores
    scores = torch.einsum('bhid,bhjd->bhij', Q, K) * scale
    print(f"✓ 标准 attention scores shape: {scores.shape}")
    print(f"  统计: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}\n")
    
    # 添加几何偏置
    scores_with_bias = scores + bias
    print(f"✓ 添加几何偏置后的 scores shape: {scores_with_bias.shape}")
    print(f"  统计: min={scores_with_bias.min():.4f}, max={scores_with_bias.max():.4f}, mean={scores_with_bias.mean():.4f}\n")
    
    # 计算attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights_with_bias = torch.softmax(scores_with_bias, dim=-1)
    
    print(f"✓ 标准 attention weights 熵: {-(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean():.4f}")
    print(f"✓ 带偏置 attention weights 熵: {-(attn_weights_with_bias * torch.log(attn_weights_with_bias + 1e-10)).sum(dim=-1).mean():.4f}")
    print("  (熵值变化反映attention分布的变化)\n")
    
    # 应用attention
    output = torch.einsum('bhij,bhjd->bhid', attn_weights, V)
    output_with_bias = torch.einsum('bhij,bhjd->bhid', attn_weights_with_bias, V)
    
    print(f"✓ 标准输出 shape: {output.shape}")
    print(f"✓ 带偏置输出 shape: {output_with_bias.shape}")
    print(f"✓ 输出差异 L2 norm: {torch.norm(output - output_with_bias):.4f}\n")


def print_usage_summary():
    """打印使用总结"""
    print("="*60)
    print("使用总结")
    print("="*60 + "\n")
    
    print("1. 在配置文件中启用几何偏置:")
    print("   config/model/diffuser/diffuser.yaml 或")
    print("   config/model/flow_matching/decoder/dit_fm.yaml\n")
    print("   ```yaml")
    print("   use_geometric_bias: true")
    print("   geometric_bias_hidden_dims: [128, 64]")
    print("   geometric_bias_feature_types: ['relative_pos', 'distance']")
    print("   ```\n")
    
    print("2. 进行对比实验:")
    print("   # Baseline（无几何偏置）")
    print("   python train_lightning.py model.decoder.use_geometric_bias=false \\")
    print("       save_root=./experiments/baseline\n")
    print("   # 增强版（有几何偏置）")
    print("   python train_lightning.py model.decoder.use_geometric_bias=true \\")
    print("       save_root=./experiments/with_geo_bias\n")
    
    print("3. 特征类型选项:")
    print("   - relative_pos: 相对位置坐标 (3D)")
    print("   - distance: 欧氏距离 (1D)")
    print("   - direction: 归一化方向向量 (3D)")
    print("   - distance_log: log(distance + eps) (1D)\n")
    
    print("4. 注意事项:")
    print("   - 几何偏置会禁用Flash Attention和SDPA优化")
    print("   - 对于大规模点云，可能会增加内存消耗")
    print("   - 建议先在小数据集上验证效果\n")
    
    print("="*60)
    print("✅ 几何注意力偏置功能演示完成！")
    print("="*60)


if __name__ == "__main__":
    try:
        demo_geometric_bias_module()
        demo_attention_integration()
        print_usage_summary()
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

