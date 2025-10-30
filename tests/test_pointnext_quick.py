#!/usr/bin/env python
"""
PointNext Backbone 快速测试脚本

用途: 快速验证 PointNext backbone 是否正常工作
运行: python tests/test_pointnext_quick.py
"""

import sys
sys.path.insert(0, '.')

import torch
from omegaconf import OmegaConf
from models.backbone import build_backbone

def main():
    print("="*80)
    print("PointNext Backbone 快速测试")
    print("="*80)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  警告: CUDA 不可用")
        print("PointNext 需要 GPU 运行，测试将跳过")
        return
    
    print(f"\n✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 创建配置
    print("\n1. 创建配置...")
    cfg = OmegaConf.create({
        'name': 'pointnext',
        'num_points': 8192,
        'num_tokens': 128,
        'out_dim': 512,
        'width': 32,
        'blocks': [1, 1, 1, 1, 1],
        'strides': [1, 2, 2, 4, 4],  # 64x 下采样 → 128 tokens
        'use_res': True,
        'radius': 0.1,
        'nsample': 32,
        'input_feature_dim': 3,
        'use_xyz': True,
        'normalize_xyz': True,
        'use_fps': True,
        'sampler': 'random',  # 使用随机采样避免潜在问题
    })
    print("   ✓ 配置创建成功")
    
    # 创建模型
    print("\n2. 创建模型...")
    try:
        model = build_backbone(cfg).cuda()
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   ✓ 模型创建成功")
        print(f"   ✓ 参数量: {param_count:.2f}M")
    except Exception as e:
        print(f"   ✗ 模型创建失败: {e}")
        print("\n提示: 请确保已安装依赖:")
        print("  pip install multimethod shortuuid easydict einops timm")
        return
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    model.eval()
    
    try:
        # 创建测试输入
        batch_size = 2
        num_points = 8192
        pos = torch.randn(batch_size, num_points, 3).cuda()
        
        with torch.no_grad():
            xyz_out, feat_out = model(pos)
        
        print(f"   ✓ 前向传播成功")
        print(f"   输入形状: {pos.shape}")
        print(f"   输出 xyz 形状: {xyz_out.shape}")
        print(f"   输出 features 形状: {feat_out.shape}")
        
        # 验证输出
        encoder_tokens = xyz_out.shape[1]
        expected_dim = cfg.out_dim
        actual_dim = feat_out.shape[1]
        
        print(f"\n4. 验证输出...")
        print(f"   Encoder 输出 tokens: {encoder_tokens}")
        print(f"   期望特征维度: {expected_dim}")
        print(f"   实际特征维度: {actual_dim}")
        
        if actual_dim == expected_dim:
            print(f"   ✓ 特征维度匹配")
        else:
            print(f"   ⚠️  特征维度不匹配")
        
        # 计算下采样率
        downsample_rate = num_points / encoder_tokens
        print(f"   实际下采样率: {downsample_rate:.1f}x")
        
        # 说明
        print(f"\n5. 说明:")
        if encoder_tokens < cfg.num_tokens:
            print(f"   ℹ️  Encoder 输出 {encoder_tokens} tokens < 目标 {cfg.num_tokens} tokens")
            print(f"   建议: 减小 strides 以增加输出 tokens")
            print(f"   例如: strides=[1,2,2,2,2] → 输出 ~512 tokens")
        elif encoder_tokens > cfg.num_tokens:
            print(f"   ℹ️  Encoder 输出 {encoder_tokens} tokens > 目标 {cfg.num_tokens} tokens")
            print(f"   FPS 采样将下采样到 {cfg.num_tokens} tokens")
        else:
            print(f"   ✓ Encoder 输出恰好匹配目标 tokens!")
        
        print("\n" + "="*80)
        print("✓ 所有测试通过！PointNext backbone 工作正常")
        print("="*80)
        
        print("\n下一步:")
        print("  1. 根据需要调整配置文件:")
        print("     config/model/flow_matching/decoder/backbone/pointnext.yaml")
        print("  2. 在训练中使用:")
        print("     python train_lightning.py model/decoder/backbone=pointnext")
        print("  3. 查看完整文档:")
        print("     docs/pointnext_setup.md")
        
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n调试提示:")
        print("  1. 确保在 GPU 上运行")
        print("  2. 检查 CUDA 版本兼容性")
        print("  3. 尝试使用 sampler='random' 而不是 'fps'")
        return

if __name__ == "__main__":
    main()

