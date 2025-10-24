"""
Point Transformer Backbone 集成烟雾测试

测试 Point Transformer 是否能够正确集成到项目中，
并与 PointNet2 和 PTv3 保持接口一致性。
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from omegaconf import OmegaConf

from models.backbone import build_backbone


def test_point_transformer_instantiation():
    """测试 Point Transformer 能否正确实例化"""
    print("测试 1: Point Transformer 实例化")
    
    # 创建配置
    cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 6,
        'num_points': 32768,
        'out_dim': 512,
        'use_xyz': True,
        'normalize_xyz': True
    })
    
    # 实例化模型
    model = build_backbone(cfg)
    
    # 验证属性
    assert hasattr(model, 'output_dim'), "模型缺少 output_dim 属性"
    assert model.output_dim == 512, f"output_dim 应为 512，实际为 {model.output_dim}"
    
    print("✓ Point Transformer 实例化成功")
    return model


def test_forward_pass_xyz_only():
    """测试仅有 xyz 坐标的前向传播"""
    print("测试 2: 仅 xyz 坐标的前向传播")
    
    cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 3,
        'num_points': 8192,
        'out_dim': 512
    })
    
    model = build_backbone(cfg).cuda()
    
    # 创建测试数据 (B=2, N=8192, C=3)
    B, N, C = 2, 8192, 3
    pos = torch.randn(B, N, C).cuda()
    
    # 前向传播
    xyz, features = model(pos)
    
    # 验证输出形状
    assert xyz.dim() == 3, f"xyz 应为 3 维张量，实际为 {xyz.dim()}"
    assert features.dim() == 3, f"features 应为 3 维张量，实际为 {features.dim()}"
    assert xyz.shape[0] == B, f"batch size 不匹配：期望 {B}，实际 {xyz.shape[0]}"
    assert xyz.shape[2] == 3, f"xyz 最后一维应为 3，实际为 {xyz.shape[2]}"
    assert features.shape[0] == B, f"features batch size 不匹配"
    assert features.shape[1] == 512, f"features 通道数应为 512，实际为 {features.shape[1]}"
    
    print(f"✓ 前向传播成功 - xyz: {xyz.shape}, features: {features.shape}")


def test_forward_pass_xyz_rgb():
    """测试 xyz + rgb 的前向传播"""
    print("测试 3: xyz + rgb 的前向传播")
    
    cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 6,
        'num_points': 8192,
        'out_dim': 512
    })
    
    model = build_backbone(cfg).cuda()
    
    # 创建测试数据 (B=2, N=8192, C=6)
    B, N, C = 2, 8192, 6
    pos = torch.randn(B, N, C).cuda()
    
    # 前向传播
    xyz, features = model(pos)
    
    # 验证输出形状
    assert xyz.shape == (B, xyz.shape[1], 3), f"xyz 形状不符合预期"
    assert features.shape[0] == B, f"features batch size 不匹配"
    assert features.shape[1] == 512, f"features 通道数应为 512"
    
    print(f"✓ xyz+rgb 前向传播成功 - xyz: {xyz.shape}, features: {features.shape}")


def test_forward_pass_xyz_rgb_mask():
    """测试 xyz + rgb + mask 的前向传播"""
    print("测试 4: xyz + rgb + mask 的前向传播")
    
    cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 7,
        'num_points': 8192,
        'out_dim': 512
    })
    
    model = build_backbone(cfg).cuda()
    
    # 创建测试数据 (B=2, N=8192, C=7)
    B, N, C = 2, 8192, 7
    pos = torch.randn(B, N, C).cuda()
    
    # 前向传播
    xyz, features = model(pos)
    
    # 验证输出形状
    assert xyz.shape[0] == B, f"batch size 不匹配"
    assert features.shape[1] == 512, f"features 通道数应为 512"
    
    print(f"✓ xyz+rgb+mask 前向传播成功 - xyz: {xyz.shape}, features: {features.shape}")


def test_interface_compatibility():
    """测试与其他 backbone 的接口兼容性"""
    print("测试 5: 与 PointNet2 接口兼容性")
    
    # Point Transformer 配置
    pt_cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 6,
        'num_points': 8192,
        'out_dim': 512
    })
    
    # PointNet2 配置
    pn2_cfg = OmegaConf.create({
        'name': 'pointnet2',
        'use_pooling': False,
        'use_xyz': True,
        'normalize_xyz': True,
        'layer1': {
            'npoint': 2048,
            'radius_list': [0.04],
            'nsample_list': [64],
            'mlp_list': [3, 64, 64, 128]
        },
        'layer2': {
            'npoint': 1024,
            'radius_list': [0.1],
            'nsample_list': [32],
            'mlp_list': [128, 128, 128, 256]
        },
        'layer3': {
            'npoint': 512,
            'radius_list': [0.2],
            'nsample_list': [32],
            'mlp_list': [256, 128, 128, 256]
        },
        'layer4': {
            'npoint': 128,
            'radius_list': [0.3],
            'nsample_list': [16],
            'mlp_list': [256, 512, 512]
        }
    })
    
    # 实例化两个模型
    pt_model = build_backbone(pt_cfg).cuda()
    pn2_model = build_backbone(pn2_cfg).cuda()
    
    # 创建测试数据
    B, N, C = 2, 8192, 6
    pos = torch.randn(B, N, C).cuda()
    
    # 前向传播
    pt_xyz, pt_feat = pt_model(pos)
    pn2_xyz, pn2_feat = pn2_model(pos)
    
    # 验证输出格式一致
    assert pt_xyz.dim() == pn2_xyz.dim(), "xyz 维度不一致"
    assert pt_feat.dim() == pn2_feat.dim(), "features 维度不一致"
    assert pt_feat.shape[1] == pn2_feat.shape[1], "features 通道数不一致"
    
    print("✓ 接口兼容性测试通过")
    print(f"  Point Transformer 输出: xyz={pt_xyz.shape}, feat={pt_feat.shape}")
    print(f"  PointNet2 输出: xyz={pn2_xyz.shape}, feat={pn2_feat.shape}")


def test_batch_consistency():
    """测试不同 batch size 的一致性"""
    print("测试 6: 批次大小一致性")
    
    cfg = OmegaConf.create({
        'name': 'point_transformer',
        'c': 6,
        'num_points': 8192,
        'out_dim': 512
    })
    
    model = build_backbone(cfg).cuda()
    
    # 测试不同的 batch size
    for batch_size in [1, 2, 4]:
        N, C = 8192, 6
        pos = torch.randn(batch_size, N, C).cuda()
        xyz, features = model(pos)
        
        assert xyz.shape[0] == batch_size, f"batch_size={batch_size} 时输出不匹配"
        assert features.shape[0] == batch_size, f"batch_size={batch_size} 时输出不匹配"
        print(f"  batch_size={batch_size}: xyz={xyz.shape}, feat={features.shape}")
    
    print("✓ 批次大小一致性测试通过")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Point Transformer Backbone 集成烟雾测试")
    print("=" * 60)
    
    try:
        # 检查 CUDA 是否可用
        if not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，测试将跳过")
            return
        
        # 运行测试
        test_point_transformer_instantiation()
        test_forward_pass_xyz_only()
        test_forward_pass_xyz_rgb()
        test_forward_pass_xyz_rgb_mask()
        test_interface_compatibility()
        test_batch_consistency()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

