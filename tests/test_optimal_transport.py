"""
测试 Optimal Transport 功能

运行方式：
    cd /home/xiantuo/source/grasp/GithubClone/SceneLeapUltra
    source ~/.bashrc && conda activate DexGrasp
    python tests/test_optimal_transport.py
"""

import sys
import os
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fm.optimal_transport import (
    SinkhornOT, 
    apply_optimal_matching,
    compute_matching_quality,
    sinkhorn_matching
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_basic_ot():
    """测试基本的OT功能"""
    print("\n" + "="*60)
    print("测试 1: 基本 Sinkhorn OT 功能")
    print("="*60)
    
    # 创建测试数据
    B, N, D = 4, 128, 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x0 = torch.randn(B, N, D, device=device)
    x1 = torch.randn(B, N, D, device=device)
    
    print(f"数据形状: x0={x0.shape}, x1={x1.shape}")
    print(f"设备: {device}")
    
    # 创建OT求解器
    ot_solver = SinkhornOT(
        reg=0.1,
        num_iters=50,
        distance_metric='euclidean',
        matching_strategy='greedy',
    )
    
    # 计算配对
    print("\n正在计算最优配对...")
    matchings, info = ot_solver(x0, x1, return_info=True)
    
    print(f"✅ 配对完成！")
    print(f"  - 配对索引形状: {matchings.shape}")
    print(f"  - 原始平均距离: {info['random_distance']:.4f}")
    print(f"  - 配对后平均距离: {info['matched_distance']:.4f}")
    print(f"  - 改进百分比: {info['improvement']:.1f}%")
    
    # 验证配对
    x1_matched = apply_optimal_matching(x0, x1, matchings)
    quality = compute_matching_quality(x0, x1, x1_matched)
    
    print(f"\n配对质量统计:")
    print(f"  - 平均距离减少: {quality['improvement_percent']:.1f}%")
    print(f"  - 最小距离: {quality['min_dist_matched']:.4f}")
    print(f"  - 最大距离: {quality['max_dist_matched']:.4f}")
    
    assert matchings.shape == (B, N), "配对索引形状错误"
    assert info['matched_distance'] < info['random_distance'], "OT应该减少平均距离"
    
    print("\n✅ 测试通过！")
    return True


def test_different_configs():
    """测试不同的配置参数"""
    print("\n" + "="*60)
    print("测试 2: 不同配置参数的影响")
    print("="*60)
    
    B, N, D = 2, 64, 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x0 = torch.randn(B, N, D, device=device)
    x1 = torch.randn(B, N, D, device=device)
    
    configs = [
        {'reg': 0.05, 'num_iters': 50, 'desc': '低正则化（更精确）'},
        {'reg': 0.1, 'num_iters': 50, 'desc': '中等正则化（平衡）'},
        {'reg': 0.2, 'num_iters': 50, 'desc': '高正则化（更快）'},
        {'reg': 0.1, 'num_iters': 20, 'desc': '少迭代次数'},
        {'reg': 0.1, 'num_iters': 100, 'desc': '多迭代次数'},
    ]
    
    results = []
    for cfg in configs:
        ot_solver = SinkhornOT(
            reg=cfg['reg'],
            num_iters=cfg['num_iters'],
            distance_metric='euclidean'
        )
        
        matchings, info = ot_solver(x0, x1, return_info=True)
        results.append({
            'config': cfg['desc'],
            'reg': cfg['reg'],
            'iters': cfg['num_iters'],
            'matched_dist': info['matched_distance'],
            'improvement': info['improvement']
        })
        
        print(f"{cfg['desc']:30s}: "
              f"matched_dist={info['matched_distance']:.4f}, "
              f"improvement={info['improvement']:.1f}%")
    
    print("\n✅ 测试通过！")
    return results


def test_batch_sizes():
    """测试不同batch size和grasp数量"""
    print("\n" + "="*60)
    print("测试 3: 不同数据规模的性能")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = 25
    
    test_cases = [
        (4, 64, "小规模"),
        (8, 128, "中规模"),
        (16, 256, "大规模"),
        (32, 512, "超大规模"),
    ]
    
    ot_solver = SinkhornOT(reg=0.1, num_iters=50)
    
    for B, N, desc in test_cases:
        x0 = torch.randn(B, N, D, device=device)
        x1 = torch.randn(B, N, D, device=device)
        
        # 测速
        if device == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start = time.time()
        matchings, info = ot_solver(x0, x1, return_info=True)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"{desc:15s} [B={B:2d}, N={N:4d}]: "
              f"时间={elapsed*1000:.2f}ms, "
              f"改进={info['improvement']:.1f}%")
    
    print("\n✅ 测试通过！")
    return True


def test_visualization():
    """可视化配对效果（2D投影）"""
    print("\n" + "="*60)
    print("测试 4: 可视化配对效果")
    print("="*60)
    
    # 创建2D数据便于可视化
    B, N, D = 1, 100, 2
    device = 'cpu'  # 可视化用CPU
    
    # 创建两个聚类分布
    x0 = torch.cat([
        torch.randn(50, 2) * 0.3 + torch.tensor([1.0, 1.0]),
        torch.randn(50, 2) * 0.3 + torch.tensor([-1.0, -1.0]),
    ]).unsqueeze(0)
    
    x1 = torch.randn(1, N, D) * 1.5
    
    # 计算配对
    ot_solver = SinkhornOT(reg=0.1, num_iters=50)
    matchings, info = ot_solver(x0, x1, return_info=True)
    x1_matched = apply_optimal_matching(x0, x1, matchings)
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1: 数据分布
    axes[0].scatter(x0[0, :, 0], x0[0, :, 1], c='blue', alpha=0.6, label='x0 (真实抓取)')
    axes[0].scatter(x1[0, :, 0], x1[0, :, 1], c='red', alpha=0.6, label='x1 (原始噪声)')
    axes[0].legend()
    axes[0].set_title('数据分布')
    axes[0].grid(True, alpha=0.3)
    
    # 子图2: 随机配对
    axes[1].scatter(x0[0, :, 0], x0[0, :, 1], c='blue', alpha=0.6, s=20)
    axes[1].scatter(x1[0, :, 0], x1[0, :, 1], c='red', alpha=0.6, s=20)
    for i in range(0, N, 5):  # 只画部分线避免太密
        axes[1].plot([x0[0, i, 0], x1[0, i, 0]], 
                     [x0[0, i, 1], x1[0, i, 1]], 
                     'gray', alpha=0.3, linewidth=0.5)
    axes[1].set_title(f'随机配对\n平均距离: {info["random_distance"]:.3f}')
    axes[1].grid(True, alpha=0.3)
    
    # 子图3: OT配对
    axes[2].scatter(x0[0, :, 0], x0[0, :, 1], c='blue', alpha=0.6, s=20)
    axes[2].scatter(x1_matched[0, :, 0], x1_matched[0, :, 1], c='red', alpha=0.6, s=20)
    for i in range(0, N, 5):
        axes[2].plot([x0[0, i, 0], x1_matched[0, i, 0]], 
                     [x0[0, i, 1], x1_matched[0, i, 1]], 
                     'green', alpha=0.5, linewidth=0.8)
    axes[2].set_title(f'OT配对\n平均距离: {info["matched_distance"]:.3f}\n改进: {info["improvement"]:.1f}%')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'tests/ot_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存到: {save_path}")
    
    # 如果不在服务器环境，可以显示图像
    # plt.show()
    
    return True


def test_gradient_flow():
    """测试梯度传播（确保可微分）"""
    print("\n" + "="*60)
    print("测试 5: 梯度传播")
    print("="*60)
    
    B, N, D = 2, 32, 25
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x0 = torch.randn(B, N, D, device=device, requires_grad=True)
    x1 = torch.randn(B, N, D, device=device, requires_grad=True)
    
    ot_solver = SinkhornOT(reg=0.1, num_iters=30)
    matchings = ot_solver(x0, x1, return_info=False)
    
    # 应用配对并计算损失
    x1_matched = apply_optimal_matching(x0, x1, matchings)
    loss = torch.nn.functional.mse_loss(x0, x1_matched)
    
    # 反向传播
    loss.backward()
    
    print(f"损失: {loss.item():.4f}")
    print(f"x0梯度范数: {x0.grad.norm().item():.4f}")
    print(f"x1梯度范数: {x1.grad.norm().item():.4f}")
    
    assert x0.grad is not None, "x0应该有梯度"
    assert x1.grad is not None, "x1应该有梯度"
    assert not torch.isnan(x0.grad).any(), "梯度不应包含NaN"
    
    print("✅ 梯度传播正常！")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" "*15 + "Optimal Transport 功能测试")
    print("="*70)
    
    tests = [
        ("基本功能", test_basic_ot),
        ("配置参数", test_different_configs),
        ("数据规模", test_batch_sizes),
        ("可视化", test_visualization),
        ("梯度传播", test_gradient_flow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, "✅ 通过", result))
            print()
        except Exception as e:
            results.append((name, f"❌ 失败: {str(e)}", None))
            print(f"\n❌ 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*70)
    print("测试总结:")
    print("="*70)
    for name, status, _ in results:
        print(f"  {name:20s}: {status}")
    
    passed = sum(1 for _, s, _ in results if s.startswith("✅"))
    total = len(results)
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查。")
    
    return results


if __name__ == "__main__":
    run_all_tests()

