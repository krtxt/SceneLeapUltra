"""
测试 PointNext Backbone 的功能和兼容性

测试内容:
1. 基本功能：输入输出形状验证
2. 不同输入格式：xyz, xyz+rgb, xyz+features
3. 与其他 backbone 的接口兼容性（PointNet2, PTv3）
4. 配置文件加载
5. 性能基准测试
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from omegaconf import OmegaConf

# 导入 backbone
from models.backbone import build_backbone
from models.backbone.pointnext_backbone import PointNextBackbone


class TestPointNextBackbone:
    """PointNext Backbone 测试套件"""
    
    @classmethod
    def setup_class(cls):
        """测试前的初始化"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*80}")
        print(f"测试设备: {cls.device}")
        print(f"{'='*80}\n")
        
        # 基础配置
        cls.batch_size = 2
        cls.num_points = 8192
        cls.num_tokens = 128
        cls.out_dim = 512
    
    def _create_config(self, **overrides):
        """创建测试配置"""
        cfg_dict = {
            'name': 'pointnext',
            'num_points': self.num_points,
            'num_tokens': self.num_tokens,
            'out_dim': self.out_dim,
            'width': 32,
            'blocks': [1, 1, 1, 1, 1],
            'strides': [1, 4, 4, 4, 4],
            'use_res': True,
            'radius': 0.1,
            'nsample': 32,
            'input_feature_dim': 3,
            'use_xyz': True,
            'normalize_xyz': True,
            'use_fps': True,
        }
        cfg_dict.update(overrides)
        return OmegaConf.create(cfg_dict)
    
    def test_01_basic_xyz_input(self):
        """测试1: 基本 xyz 输入 (B, N, 3)"""
        print("\n[测试1] 基本 xyz 输入 (B, N, 3)")
        print("-" * 80)
        
        cfg = self._create_config()
        model = PointNextBackbone(cfg).to(self.device)
        model.eval()
        
        # 创建测试输入
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        
        with torch.no_grad():
            xyz_out, feat_out = model(pos)
        
        # 验证输出形状
        expected_xyz_shape = (self.batch_size, self.num_tokens, 3)
        expected_feat_shape = (self.batch_size, self.out_dim, self.num_tokens)
        
        print(f"输入形状: {pos.shape}")
        print(f"输出 xyz 形状: {xyz_out.shape} (期望: {expected_xyz_shape})")
        print(f"输出特征形状: {feat_out.shape} (期望: {expected_feat_shape})")
        
        assert xyz_out.shape == expected_xyz_shape, \
            f"xyz 形状错误: {xyz_out.shape} != {expected_xyz_shape}"
        assert feat_out.shape == expected_feat_shape, \
            f"特征形状错误: {feat_out.shape} != {expected_feat_shape}"
        
        print("✓ 测试通过")
    
    def test_02_xyzrgb_input(self):
        """测试2: xyz+rgb 输入 (B, N, 6)"""
        print("\n[测试2] xyz+rgb 输入 (B, N, 6)")
        print("-" * 80)
        
        cfg = self._create_config(input_feature_dim=6)
        model = PointNextBackbone(cfg).to(self.device)
        model.eval()
        
        # 创建测试输入
        pos = torch.randn(self.batch_size, self.num_points, 6).to(self.device)
        
        with torch.no_grad():
            xyz_out, feat_out = model(pos)
        
        # 验证输出形状
        expected_xyz_shape = (self.batch_size, self.num_tokens, 3)
        expected_feat_shape = (self.batch_size, self.out_dim, self.num_tokens)
        
        print(f"输入形状: {pos.shape}")
        print(f"输出 xyz 形状: {xyz_out.shape} (期望: {expected_xyz_shape})")
        print(f"输出特征形状: {feat_out.shape} (期望: {expected_feat_shape})")
        
        assert xyz_out.shape == expected_xyz_shape
        assert feat_out.shape == expected_feat_shape
        
        print("✓ 测试通过")
    
    def test_03_different_output_dims(self):
        """测试3: 不同的输出维度"""
        print("\n[测试3] 不同的输出维度")
        print("-" * 80)
        
        test_configs = [
            {'out_dim': 256, 'num_tokens': 256},
            {'out_dim': 512, 'num_tokens': 128},
            {'out_dim': 768, 'num_tokens': 64},
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n配置 {i+1}: out_dim={config['out_dim']}, num_tokens={config['num_tokens']}")
            
            cfg = self._create_config(**config)
            model = PointNextBackbone(cfg).to(self.device)
            model.eval()
            
            pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
            
            with torch.no_grad():
                xyz_out, feat_out = model(pos)
            
            expected_xyz_shape = (self.batch_size, config['num_tokens'], 3)
            expected_feat_shape = (self.batch_size, config['out_dim'], config['num_tokens'])
            
            print(f"  输出 xyz 形状: {xyz_out.shape} (期望: {expected_xyz_shape})")
            print(f"  输出特征形状: {feat_out.shape} (期望: {expected_feat_shape})")
            
            assert xyz_out.shape == expected_xyz_shape
            assert feat_out.shape == expected_feat_shape
            print(f"  ✓ 配置 {i+1} 通过")
        
        print("\n✓ 所有配置测试通过")
    
    def test_04_build_backbone_interface(self):
        """测试4: 通过 build_backbone 接口创建"""
        print("\n[测试4] 通过 build_backbone 接口创建")
        print("-" * 80)
        
        cfg = self._create_config()
        model = build_backbone(cfg).to(self.device)
        model.eval()
        
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        
        with torch.no_grad():
            xyz_out, feat_out = model(pos)
        
        expected_xyz_shape = (self.batch_size, self.num_tokens, 3)
        expected_feat_shape = (self.batch_size, self.out_dim, self.num_tokens)
        
        print(f"模型类型: {type(model).__name__}")
        print(f"输出 xyz 形状: {xyz_out.shape} (期望: {expected_xyz_shape})")
        print(f"输出特征形状: {feat_out.shape} (期望: {expected_feat_shape})")
        
        assert isinstance(model, PointNextBackbone)
        assert xyz_out.shape == expected_xyz_shape
        assert feat_out.shape == expected_feat_shape
        
        print("✓ 测试通过")
    
    def test_05_config_file_loading(self):
        """测试5: 从配置文件加载"""
        print("\n[测试5] 从配置文件加载")
        print("-" * 80)
        
        config_path = project_root / "config/model/flow_matching/decoder/backbone/pointnext.yaml"
        
        if not config_path.exists():
            print(f"⚠ 配置文件不存在: {config_path}")
            pytest.skip("配置文件不存在")
            return
        
        cfg = OmegaConf.load(config_path)
        print(f"配置文件路径: {config_path}")
        print(f"配置内容: {OmegaConf.to_yaml(cfg)[:200]}...")
        
        model = build_backbone(cfg).to(self.device)
        model.eval()
        
        # 使用配置文件中的参数
        num_points = cfg.get('num_points', 8192)
        num_tokens = cfg.get('num_tokens', 128)
        out_dim = cfg.get('out_dim', 512)
        
        pos = torch.randn(self.batch_size, num_points, 3).to(self.device)
        
        with torch.no_grad():
            xyz_out, feat_out = model(pos)
        
        expected_xyz_shape = (self.batch_size, num_tokens, 3)
        expected_feat_shape = (self.batch_size, out_dim, num_tokens)
        
        print(f"输出 xyz 形状: {xyz_out.shape} (期望: {expected_xyz_shape})")
        print(f"输出特征形状: {feat_out.shape} (期望: {expected_feat_shape})")
        
        assert xyz_out.shape == expected_xyz_shape
        assert feat_out.shape == expected_feat_shape
        
        print("✓ 测试通过")
    
    def test_06_fps_vs_stride_sampling(self):
        """测试6: FPS 采样 vs 步长采样"""
        print("\n[测试6] FPS 采样 vs 步长采样")
        print("-" * 80)
        
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        
        # FPS 采样
        cfg_fps = self._create_config(use_fps=True)
        model_fps = PointNextBackbone(cfg_fps).to(self.device)
        model_fps.eval()
        
        with torch.no_grad():
            xyz_fps, feat_fps = model_fps(pos)
        
        print(f"FPS 采样:")
        print(f"  输出 xyz 形状: {xyz_fps.shape}")
        print(f"  输出特征形状: {feat_fps.shape}")
        
        # 步长采样
        cfg_stride = self._create_config(use_fps=False)
        model_stride = PointNextBackbone(cfg_stride).to(self.device)
        model_stride.eval()
        
        with torch.no_grad():
            xyz_stride, feat_stride = model_stride(pos)
        
        print(f"步长采样:")
        print(f"  输出 xyz 形状: {xyz_stride.shape}")
        print(f"  输出特征形状: {feat_stride.shape}")
        
        # 两种采样方法应该输出相同形状
        assert xyz_fps.shape == xyz_stride.shape
        assert feat_fps.shape == feat_stride.shape
        
        print("✓ 测试通过")
    
    def test_07_normalization_effect(self):
        """测试7: 归一化的影响"""
        print("\n[测试7] 归一化的影响")
        print("-" * 80)
        
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        # 添加一些偏移和缩放
        pos = pos * 10.0 + 5.0
        
        print(f"输入点云统计:")
        print(f"  最小值: {pos.min().item():.4f}")
        print(f"  最大值: {pos.max().item():.4f}")
        print(f"  均值: {pos.mean().item():.4f}")
        print(f"  标准差: {pos.std().item():.4f}")
        
        # 有归一化
        cfg_norm = self._create_config(normalize_xyz=True)
        model_norm = PointNextBackbone(cfg_norm).to(self.device)
        model_norm.eval()
        
        with torch.no_grad():
            xyz_norm, feat_norm = model_norm(pos)
        
        print(f"\n有归一化:")
        print(f"  输出形状: {xyz_norm.shape}, {feat_norm.shape}")
        print(f"  输出有效性: 无 NaN={not torch.isnan(feat_norm).any()}, 无 Inf={not torch.isinf(feat_norm).any()}")
        
        # 无归一化
        cfg_no_norm = self._create_config(normalize_xyz=False)
        model_no_norm = PointNextBackbone(cfg_no_norm).to(self.device)
        model_no_norm.eval()
        
        with torch.no_grad():
            xyz_no_norm, feat_no_norm = model_no_norm(pos)
        
        print(f"\n无归一化:")
        print(f"  输出形状: {xyz_no_norm.shape}, {feat_no_norm.shape}")
        print(f"  输出有效性: 无 NaN={not torch.isnan(feat_no_norm).any()}, 无 Inf={not torch.isinf(feat_no_norm).any()}")
        
        # 检查输出是否有效（无 NaN 或 Inf）
        assert not torch.isnan(feat_norm).any()
        assert not torch.isinf(feat_norm).any()
        assert not torch.isnan(feat_no_norm).any()
        assert not torch.isinf(feat_no_norm).any()
        
        print("\n✓ 测试通过")
    
    def test_08_performance_benchmark(self):
        """测试8: 性能基准测试"""
        print("\n[测试8] 性能基准测试")
        print("-" * 80)
        
        cfg = self._create_config()
        model = PointNextBackbone(cfg).to(self.device)
        model.eval()
        
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(pos)
        
        # 计时
        num_runs = 20
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                xyz_out, feat_out = model(pos)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        
        # 计算显存使用（如果在 GPU 上）
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                xyz_out, feat_out = model(pos)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            print(f"平均推理时间: {avg_time:.2f} ms/sample")
            print(f"峰值显存使用: {peak_memory:.2f} MB")
        else:
            print(f"平均推理时间: {avg_time:.2f} ms/sample")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params/1e6:.2f}M")
        print(f"可训练参数量: {trainable_params/1e6:.2f}M")
        
        print("✓ 测试通过")
    
    def test_09_interface_compatibility(self):
        """测试9: 与其他 backbone 的接口兼容性"""
        print("\n[测试9] 与其他 backbone 的接口兼容性")
        print("-" * 80)
        
        pos = torch.randn(self.batch_size, self.num_points, 3).to(self.device)
        
        # PointNext
        cfg_pointnext = self._create_config()
        model_pointnext = build_backbone(cfg_pointnext).to(self.device)
        model_pointnext.eval()
        
        with torch.no_grad():
            xyz_pn, feat_pn = model_pointnext(pos)
        
        print(f"PointNext 输出: xyz={xyz_pn.shape}, feat={feat_pn.shape}")
        
        # 验证输出格式一致性
        # 所有 backbone 应该返回:
        # - xyz: (B, K, 3)
        # - features: (B, D, K)
        assert len(xyz_pn.shape) == 3 and xyz_pn.shape[-1] == 3
        assert len(feat_pn.shape) == 3 and feat_pn.shape[1] == self.out_dim
        
        print("✓ 接口兼容性测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("PointNext Backbone 完整测试套件")
    print("="*80)
    
    # 创建测试类实例
    test_suite = TestPointNextBackbone()
    test_suite.setup_class()
    
    # 运行所有测试
    tests = [
        test_suite.test_01_basic_xyz_input,
        test_suite.test_02_xyzrgb_input,
        test_suite.test_03_different_output_dims,
        test_suite.test_04_build_backbone_interface,
        test_suite.test_05_config_file_loading,
        test_suite.test_06_fps_vs_stride_sampling,
        test_suite.test_07_normalization_effect,
        test_suite.test_08_performance_benchmark,
        test_suite.test_09_interface_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ 测试失败: {test_func.__name__}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    
    if failed == 0:
        print("\n✓ 所有测试通过!")
    else:
        print(f"\n✗ {failed} 个测试失败")
    
    print("="*80)


if __name__ == "__main__":
    # 检查是否有 pytest
    try:
        import pytest
        print("使用 pytest 运行测试...")
        pytest.main([__file__, "-v", "-s"])
    except ImportError:
        print("pytest 未安装，使用自定义测试运行器...")
        run_all_tests()

