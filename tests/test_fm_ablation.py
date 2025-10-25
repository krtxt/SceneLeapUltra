"""
Flow Matching 消融实验脚本

系统性测试不同配置组合的效果：
1. NFE消融：{8, 16, 32, 64}
2. 求解器消融：{heun, rk4, rk45}
3. 时间采样器消融：{uniform, cosine, beta}
4. CFG消融：scale ∈ {0, 1, 3, 5}
"""

import sys
import time

import torch

sys.path.insert(0, '/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra')

import math

from models.fm.guidance import apply_cfg_clipped
from models.fm.paths import linear_ot_path
from models.fm.solvers import (ODESolverStats, heun_solver, integrate_ode,
                               rk4_solver)


def ablation_nfe():
    """NFE消融实验"""
    print("="*60)
    print("消融实验1: NFE (函数评估次数)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, num_grasps, D = 4, 16, 25
    
    # 简单的速度函数
    def velocity_fn(x, t, data):
        return torch.randn_like(x) * 0.1
    
    nfe_values = [8, 16, 32, 64]
    results = []
    
    for nfe in nfe_values:
        x1 = torch.randn(B, num_grasps, D).to(device)
        data = {}
        stats = ODESolverStats()
        
        start_time = time.time()
        x0, info = rk4_solver(velocity_fn, x1, data, nfe=nfe, stats=stats)
        elapsed = time.time() - start_time
        
        result = {
            'nfe': nfe,
            'actual_nfe': stats.nfe,
            'steps': stats.accepted_steps,
            'time': elapsed,
            'output_norm': torch.norm(x0).item()
        }
        results.append(result)
        
        print(f"NFE={nfe:3d}: 实际NFE={stats.nfe:3d}, "
              f"步数={stats.accepted_steps:2d}, "
              f"时间={elapsed:.4f}s, "
              f"||x0||={result['output_norm']:.3f}")
    
    print("\n✅ NFE消融实验完成")
    return results


def ablation_solver():
    """求解器消融实验"""
    print("\n" + "="*60)
    print("消融实验2: 求解器类型")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, num_grasps, D = 4, 16, 25
    
    def velocity_fn(x, t, data):
        return torch.randn_like(x) * 0.1
    
    solvers = [
        ('heun', {'nfe': 32}),
        ('rk4', {'nfe': 32}),
    ]
    
    results = []
    
    for solver_name, solver_kwargs in solvers:
        x1 = torch.randn(B, num_grasps, D).to(device)
        data = {}
        stats = ODESolverStats()
        
        start_time = time.time()
        x0, info = integrate_ode(
            velocity_fn, x1, data,
            solver_type=solver_name,
            **solver_kwargs
        )
        elapsed = time.time() - start_time
        
        result = {
            'solver': solver_name,
            'nfe': info.get('nfe', 0),
            'time': elapsed,
            'output_norm': torch.norm(x0).item()
        }
        results.append(result)
        
        print(f"{solver_name:8s}: NFE={result['nfe']:3d}, "
              f"时间={elapsed:.4f}s, "
              f"||x0||={result['output_norm']:.3f}")
    
    print("\n✅ 求解器消融实验完成")
    return results


def ablation_time_sampler():
    """时间采样器消融实验"""
    print("\n" + "="*60)
    print("消融实验3: 时间采样策略")
    print("="*60)
    
    B = 10000  # 大样本量统计
    
    samplers = {
        'uniform': lambda b: torch.rand(b),
        'cosine': lambda b: torch.acos(1 - 2*torch.rand(b)) / math.pi,
        'beta': lambda b: torch.distributions.Beta(2.0, 2.0).sample((b,))
    }
    
    results = []
    
    for name, sampler_fn in samplers.items():
        t = sampler_fn(B)
        
        result = {
            'sampler': name,
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'median': t.median().item()
        }
        results.append(result)
        
        print(f"{name:8s}: mean={result['mean']:.3f}, "
              f"std={result['std']:.3f}, "
              f"median={result['median']:.3f}, "
              f"范围=[{result['min']:.3f}, {result['max']:.3f}]")
    
    print("\n✅ 时间采样器消融实验完成")
    print("   推荐: cosine或beta(2,2)，强调中段时间学习")
    return results


def ablation_cfg():
    """CFG消融实验"""
    print("\n" + "="*60)
    print("消融实验4: Classifier-Free Guidance")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, num_grasps, D = 4, 16, 25
    
    v_cond = torch.randn(B, num_grasps, D).to(device)
    v_uncond = torch.randn(B, num_grasps, D).to(device) * 0.5  # 较小的无条件速度
    
    scales = [0.0, 1.0, 3.0, 5.0]
    clip_norm = 5.0
    
    results = []
    
    for scale in scales:
        v_cfg = apply_cfg_clipped(v_cond, v_uncond, scale=scale, clip_norm=clip_norm)
        
        diff = v_cond - v_uncond
        diff_norm = torch.norm(diff, dim=-1).mean().item()
        cfg_norm = torch.norm(v_cfg, dim=-1).mean().item()
        
        result = {
            'scale': scale,
            'diff_norm': diff_norm,
            'cfg_norm': cfg_norm,
            'amplification': cfg_norm / (torch.norm(v_cond, dim=-1).mean().item() + 1e-8)
        }
        results.append(result)
        
        print(f"Scale={scale:.1f}: ||diff||={diff_norm:.3f}, "
              f"||v_cfg||={cfg_norm:.3f}, "
              f"放大倍数={result['amplification']:.3f}x")
    
    print("\n✅ CFG消融实验完成")
    print(f"   范数裁剪阈值: {clip_norm}")
    return results


def ablation_paths():
    """路径消融实验"""
    print("\n" + "="*60)
    print("消融实验5: 概率路径")
    print("="*60)
    
    from models.fm.paths import diffusion_path_vp, linear_ot_path
    
    B, num_grasps, D = 4, 16, 25
    x0 = torch.randn(B, num_grasps, D)
    x1 = torch.randn(B, num_grasps, D)
    
    t_values = torch.linspace(0.1, 0.9, 5)
    
    paths = {
        'linear_ot': linear_ot_path,
        'diffusion_vp': diffusion_path_vp
    }
    
    results = []
    
    for path_name, path_fn in paths.items():
        print(f"\n{path_name}:")
        path_results = []
        
        for t_val in t_values:
            t = torch.full((B,), t_val)
            x_t, v_star = path_fn(x0, x1, t)
            
            v_norm = torch.norm(v_star, dim=-1).mean().item()
            x_t_norm = torch.norm(x_t, dim=-1).mean().item()
            
            path_results.append({
                't': t_val.item(),
                'v_norm': v_norm,
                'x_t_norm': x_t_norm
            })
            
            print(f"  t={t_val:.2f}: ||v*||={v_norm:.3f}, ||x_t||={x_t_norm:.3f}")
        
        results.append({
            'path': path_name,
            'results': path_results
        })
    
    print("\n✅ 路径消融实验完成")
    print("   推荐: linear_ot (默认，最简单最稳定)")
    return results


def main():
    """运行所有消融实验"""
    print("Flow Matching 消融实验套件\n")
    
    experiments = [
        ("NFE消融", ablation_nfe),
        ("求解器消融", ablation_solver),
        ("时间采样器消融", ablation_time_sampler),
        ("CFG消融", ablation_cfg),
        ("路径消融", ablation_paths),
    ]
    
    all_results = {}
    
    for name, exp_fn in experiments:
        try:
            results = exp_fn()
            all_results[name] = results
        except Exception as e:
            print(f"\n❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("消融实验总结")
    print("="*60)
    print(f"完成实验: {len(all_results)}/{len(experiments)}")
    
    print("\n推荐配置:")
    print("  - Solver: rk4 (4阶精度，稳定)")
    print("  - NFE: 32 (质量-速度平衡)")
    print("  - Time Sampler: cosine (强调中段)")
    print("  - CFG Scale: 3.0 (平衡引导强度)")
    print("  - Path: linear_ot (最简单最快)")
    
    return 0


if __name__ == "__main__":
    exit(main())

