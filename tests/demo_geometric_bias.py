"""
几何注意力偏置功能演示脚本

展示如何在训练和测试中启用/禁用几何注意力偏置，进行对比实验。
"""

import torch
import sys
import os
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.dit import DiTModel
from models.decoder.dit_fm import DiTFM


def demo_dit_with_geometric_bias():
    """演示DiT模型使用几何偏置"""
    print("\n" + "="*60)
    print("演示 DiT 模型使用几何注意力偏置")
    print("="*60 + "\n")
    
    # 创建配置（启用几何偏置）
    cfg = OmegaConf.create({
        'name': 'dit',
        'rot_type': 'quat',
        'd_model': 256,
        'num_layers': 4,
        'num_heads': 4,
        'd_head': 64,
        'dropout': 0.1,
        'max_sequence_length': 100,
        'use_learnable_pos_embedding': False,
        'time_embed_dim': 512,
        'use_adaptive_norm': True,
        'use_text_condition': False,
        'text_dropout_prob': 0.1,
        'use_negative_prompts': False,
        'use_object_mask': False,
        'use_rgb': False,
        'attention_dropout': 0.0,
        'cross_attention_dropout': 0.0,
        'time_embed_mult': 4,
        'ff_mult': 4,
        'ff_dropout': 0.1,
        'gradient_checkpointing': False,
        'use_flash_attention': False,
        'attention_chunk_size': 512,
        'memory_monitoring': False,
        'use_adaln_zero': False,
        'use_scene_pooling': True,
        # 几何偏置配置
        'use_geometric_bias': True,
        'geometric_bias_hidden_dims': [128, 64],
        'geometric_bias_feature_types': ['relative_pos', 'distance'],
        # Backbone配置
        'backbone': {
            'name': 'pointnet2',
            'c_in': 3,
            'num_points': 1024,
        }
    })
    
    # 创建模型
    print("创建DiT模型（启用几何偏置）...")
    model = DiTModel(cfg).eval()
    print(f"✓ 模型创建成功，几何偏置: {'启用' if model.use_geometric_bias else '禁用'}")
    
    # 创建测试数据
    B, N_grasps, N_points = 2, 4, 1024
    x_t = torch.randn(B, N_grasps, 23)  # (B, N_grasps, d_x)
    ts = torch.randint(0, 1000, (B,))
    scene_pc = torch.randn(B, N_points, 3)
    
    data = {
        'scene_pc': scene_pc,
        'scene_mask': torch.ones(B, N_points)
    }
    
    # 预处理条件特征
    print("\n预处理条件特征...")
    data = model.condition(data)
    
    # 前向传播
    print("执行前向传播...")
    with torch.no_grad():
        output = model(x_t, ts, data)
    
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x_t.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出统计: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    # 对比：禁用几何偏置
    print("\n\n创建DiT模型（禁用几何偏置）...")
    cfg.use_geometric_bias = False
    model_no_bias = DiTModel(cfg).eval()
    print(f"✓ 模型创建成功，几何偏置: {'启用' if model_no_bias.use_geometric_bias else '禁用'}")
    
    # 使用相同的数据
    data_no_bias = model_no_bias.condition({'scene_pc': scene_pc, 'scene_mask': torch.ones(B, N_points)})
    
    print("执行前向传播...")
    with torch.no_grad():
        output_no_bias = model_no_bias(x_t, ts, data_no_bias)
    
    print(f"✓ 前向传播成功")
    print(f"  输出形状: {output_no_bias.shape}")
    print(f"  输出统计: min={output_no_bias.min():.4f}, max={output_no_bias.max():.4f}, mean={output_no_bias.mean():.4f}")


def demo_dit_fm_with_geometric_bias():
    """演示DiT-FM模型使用几何偏置"""
    print("\n\n" + "="*60)
    print("演示 DiT-FM 模型使用几何注意力偏置")
    print("="*60 + "\n")
    
    # 创建配置（启用几何偏置）
    cfg = OmegaConf.create({
        'name': 'dit_fm',
        'rot_type': 'quat',
        'pred_mode': 'velocity',
        'd_model': 256,
        'num_layers': 4,
        'num_heads': 4,
        'd_head': 64,
        'dropout': 0.1,
        'max_sequence_length': 100,
        'use_learnable_pos_embedding': False,
        'time_embed_dim': 512,
        'use_adaptive_norm': True,
        'continuous_time': True,
        'freq_dim': 256,
        'use_text_condition': False,
        'text_dropout_prob': 0.1,
        'use_negative_prompts': False,
        'use_object_mask': False,
        'use_rgb': False,
        'attention_dropout': 0.0,
        'cross_attention_dropout': 0.0,
        'time_embed_mult': 4,
        'ff_mult': 4,
        'ff_dropout': 0.1,
        'gradient_checkpointing': False,
        'use_flash_attention': False,
        'attention_chunk_size': 512,
        'memory_monitoring': False,
        'use_adaln_zero': False,
        'use_scene_pooling': True,
        'debug': {'check_nan': True, 'log_tensor_stats': False},
        # 几何偏置配置
        'use_geometric_bias': True,
        'geometric_bias_hidden_dims': [128, 64],
        'geometric_bias_feature_types': ['relative_pos', 'distance', 'direction'],
        # Backbone配置
        'backbone': {
            'name': 'pointnet2',
            'c_in': 3,
            'num_points': 1024,
        }
    })
    
    # 创建模型
    print("创建DiT-FM模型（启用几何偏置）...")
    model = DiTFM(cfg).eval()
    print(f"✓ 模型创建成功，几何偏置: {'启用' if model.use_geometric_bias else '禁用'}")
    
    # 创建测试数据
    B, N_grasps, N_points = 2, 4, 1024
    x_t = torch.randn(B, N_grasps, 23)
    ts = torch.rand(B)  # Flow Matching使用[0,1]的连续时间
    scene_pc = torch.randn(B, N_points, 3)
    
    # 创建scene context（需要先通过backbone处理）
    from models.backbone import build_backbone
    backbone = build_backbone(cfg.backbone)
    with torch.no_grad():
        scene_features = backbone(scene_pc.transpose(1, 2).contiguous())
        scene_cond = scene_features.transpose(1, 2)  # (B, N, C)
    
    data = {
        'scene_pc': scene_pc,
        'scene_cond': scene_cond,
        'scene_mask': torch.ones(B, N_points)
    }
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        output = model(x_t, ts, data)
    
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {x_t.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出统计: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    print("\n" + "="*60)
    print("✅ 几何注意力偏置演示完成！")
    print("="*60)


def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "="*60)
    print("几何注意力偏置使用说明")
    print("="*60 + "\n")
    
    print("1. 在配置文件中启用几何偏置:")
    print("   - config/model/diffuser/diffuser.yaml")
    print("   - config/model/flow_matching/decoder/dit_fm.yaml")
    print("\n   设置以下参数:")
    print("   ```yaml")
    print("   use_geometric_bias: true  # 默认false")
    print("   geometric_bias_hidden_dims: [128, 64]  # MLP隐藏层")
    print("   geometric_bias_feature_types: ['relative_pos', 'distance']  # 特征类型")
    print("   ```\n")
    
    print("2. 特征类型选项:")
    print("   - 'relative_pos': 相对位置坐标 (3D)")
    print("   - 'distance': 欧氏距离 (1D)")
    print("   - 'direction': 归一化方向向量 (3D)")
    print("   - 'distance_log': log(distance + eps) (1D)")
    print("\n")
    
    print("3. 进行对比实验:")
    print("   a) 训练baseline（禁用几何偏置）:")
    print("      python train_lightning.py model.decoder.use_geometric_bias=false")
    print("\n   b) 训练增强版（启用几何偏置）:")
    print("      python train_lightning.py model.decoder.use_geometric_bias=true")
    print("\n   c) 比较两个模型的收敛速度和最终性能")
    print("\n")
    
    print("4. 注意事项:")
    print("   - 几何偏置会禁用Flash Attention和SDPA优化")
    print("   - 对于大规模点云，可能会增加内存消耗")
    print("   - 建议在较小的学习率下微调已有模型")


if __name__ == "__main__":
    print_usage_instructions()
    
    try:
        demo_dit_with_geometric_bias()
        demo_dit_fm_with_geometric_bias()
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

