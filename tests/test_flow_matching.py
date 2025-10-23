"""
Flow Matching 功能测试脚本

测试FM模型、路径、求解器和引导模块的基本功能
"""

import torch
import sys
sys.path.insert(0, '/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra')

def test_imports():
    """测试所有FM模块是否能正确导入"""
    print("测试1: 模块导入")
    try:
        from models.decoder.dit_fm import DiTFM, ContinuousTimeEmbedding
        from models.fm_lightning import FlowMatchingLightning
        from models.fm.paths import linear_ot_path, diffusion_path_vp
        from models.fm.solvers import heun_solver, rk4_solver, integrate_ode
        from models.fm.guidance import apply_cfg, apply_cfg_clipped
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_continuous_time_embedding():
    """测试连续时间嵌入"""
    print("\n测试2: ContinuousTimeEmbedding")
    try:
        from models.decoder.dit_fm import ContinuousTimeEmbedding
        
        B = 4
        dim = 512
        freq_dim = 256
        
        embed = ContinuousTimeEmbedding(dim, freq_dim)
        t = torch.rand(B)  # [0, 1]
        
        emb = embed(t)
        
        assert emb.shape == (B, dim), f"输出形状错误: {emb.shape}"
        assert not torch.isnan(emb).any(), "输出包含NaN"
        assert not torch.isinf(emb).any(), "输出包含Inf"
        
        print(f"✅ 连续时间嵌入测试通过")
        print(f"   输入: t={t.shape}, 输出: emb={emb.shape}")
        print(f"   范围: [{emb.min():.3f}, {emb.max():.3f}]")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linear_ot_path():
    """测试线性OT路径"""
    print("\n测试3: Linear OT Path")
    try:
        from models.fm.paths import linear_ot_path
        
        B, num_grasps, D = 2, 4, 25
        x0 = torch.randn(B, num_grasps, D)
        x1 = torch.randn(B, num_grasps, D)
        t = torch.rand(B)
        
        x_t, v_star = linear_ot_path(x0, x1, t)
        
        assert x_t.shape == (B, num_grasps, D), f"x_t形状错误: {x_t.shape}"
        assert v_star.shape == (B, num_grasps, D), f"v_star形状错误: {v_star.shape}"
        
        # 验证速度是常量
        v_expected = x1 - x0
        assert torch.allclose(v_star, v_expected, atol=1e-6), "速度计算错误"
        
        # 验证插值
        t_exp = t.view(-1, 1, 1)
        x_t_expected = (1 - t_exp) * x0 + t_exp * x1
        assert torch.allclose(x_t, x_t_expected, atol=1e-6), "插值计算错误"
        
        print(f"✅ Linear OT Path测试通过")
        print(f"   x0: {x0.shape}, x1: {x1.shape}, t: {t.shape}")
        print(f"   x_t: {x_t.shape}, v_star: {v_star.shape}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rk4_solver():
    """测试RK4求解器"""
    print("\n测试4: RK4 Solver")
    try:
        from models.fm.solvers import rk4_solver, ODESolverStats
        
        B, num_grasps, D = 2, 4, 25
        
        # 简单的常速度场 v(x, t) = constant
        def constant_velocity_fn(x, t, data):
            return torch.ones_like(x) * 0.1
        
        x1 = torch.randn(B, num_grasps, D)
        data = {}
        stats = ODESolverStats()
        
        x0, info = rk4_solver(constant_velocity_fn, x1, data, nfe=32, stats=stats)
        
        assert x0.shape == (B, num_grasps, D), f"输出形状错误: {x0.shape}"
        assert not torch.isnan(x0).any(), "输出包含NaN"
        assert info['solver'] == 'rk4', "求解器类型错误"
        assert stats.nfe == 32, f"NFE错误: {stats.nfe}"
        
        print(f"✅ RK4求解器测试通过")
        print(f"   NFE: {stats.nfe}, 步数: {stats.accepted_steps}")
        print(f"   输入: {x1.shape}, 输出: {x0.shape}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cfg_clipped():
    """测试裁剪CFG"""
    print("\n测试5: CFG Clipped")
    try:
        from models.fm.guidance import apply_cfg_clipped
        
        B, num_grasps, D = 2, 4, 25
        v_cond = torch.randn(B, num_grasps, D)
        v_uncond = torch.randn(B, num_grasps, D)
        
        scale = 3.0
        clip_norm = 5.0
        
        v_cfg = apply_cfg_clipped(v_cond, v_uncond, scale, clip_norm)
        
        assert v_cfg.shape == (B, num_grasps, D), f"输出形状错误: {v_cfg.shape}"
        assert not torch.isnan(v_cfg).any(), "输出包含NaN"
        
        # 验证裁剪效果
        diff = v_cond - v_uncond
        diff_norm = torch.norm(diff, dim=-1)
        
        print(f"✅ CFG Clipped测试通过")
        print(f"   v_cond: {v_cond.shape}, v_uncond: {v_uncond.shape}")
        print(f"   差异范数: min={diff_norm.min():.3f}, max={diff_norm.max():.3f}")
        print(f"   CFG输出: {v_cfg.shape}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dit_fm_forward():
    """测试DiT-FM前向传播"""
    print("\n测试6: DiT-FM Forward (需要GPU)")
    
    # Skip test on CPU (PointNet2 requires CUDA)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        has_gpu = (result.returncode == 0)
    except:
        has_gpu = False
    
    if not has_gpu:
        print("⚠️  跳过测试（PointNet2需要GPU，CPU环境自动跳过）")
        return True
    
    try:
        from omegaconf import OmegaConf
        from models.decoder.dit_fm import DiTFM
        
        # 创建最小配置
        cfg = OmegaConf.create({
            'name': 'dit_fm',
            'mode': 'velocity',
            'rot_type': 'r6d',
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 4,
            'd_head': 64,
            'dropout': 0.1,
            'max_sequence_length': 100,
            'use_learnable_pos_embedding': False,
            'time_embed_dim': 512,
            'use_adaptive_norm': True,
            'continuous_time': True,
            'freq_dim': 128,
            'use_text_condition': False,
            'text_dropout_prob': 0.1,
            'use_negative_prompts': False,
            'use_object_mask': False,
            'use_rgb': False,
            'backbone': {
                'name': 'pointnet2',
                'use_pooling': False,
                'layer1': {
                    'npoint': 512,
                    'radius_list': [0.04],
                    'nsample_list': [32],
                    'mlp_list': [0, 64, 64, 128]
                },
                'layer2': {
                    'npoint': 256,
                    'radius_list': [0.1],
                    'nsample_list': [16],
                    'mlp_list': [128, 128, 128, 256]
                },
                'layer3': {
                    'npoint': 128,
                    'radius_list': [0.2],
                    'nsample_list': [16],
                    'mlp_list': [256, 128, 128, 256]
                },
                'layer4': {
                    'npoint': 64,
                    'radius_list': [0.3],
                    'nsample_list': [8],
                    'mlp_list': [256, 512, 512]
                },
                'use_xyz': True,
                'normalize_xyz': True
            },
            'debug': {
                'check_nan': True,
                'log_tensor_stats': False
            }
        })
        
        model = DiTFM(cfg)
        model.eval()
        model = model.cuda()
        
        B, num_grasps, D = 2, 4, 25
        x_t = torch.randn(B, num_grasps, D).cuda()
        t = torch.rand(B).cuda()
        
        # 最小数据字典
        data = {
            'scene_pc': torch.randn(B, 1024, 3).cuda(),
        }
        
        # 预计算条件
        cond = model.condition(data)
        data.update(cond)
        
        # 前向传播
        with torch.no_grad():
            v_pred = model(x_t, t, data)
        
        assert v_pred.shape == (B, num_grasps, D), f"输出形状错误: {v_pred.shape}"
        assert not torch.isnan(v_pred).any(), "输出包含NaN"
        assert not torch.isinf(v_pred).any(), "输出包含Inf"
        
        print(f"✅ DiT-FM前向传播测试通过")
        print(f"   输入: x_t={x_t.shape}, t={t.shape}")
        print(f"   输出: v_pred={v_pred.shape}")
        print(f"   范围: [{v_pred.min():.3f}, {v_pred.max():.3f}]")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("Flow Matching 功能测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("连续时间嵌入", test_continuous_time_embedding),
        ("Linear OT路径", test_linear_ot_path),
        ("RK4求解器", test_rk4_solver),
        ("CFG裁剪", test_cfg_clipped),
        ("DiT-FM前向", test_dit_fm_forward),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 执行异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n通过率: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！Flow Matching基础功能正常。")
        return 0
    else:
        print(f"\n⚠️  {total - passed}个测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())

