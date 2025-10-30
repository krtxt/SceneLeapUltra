"""
PTv3 五种 Token 策略综合测试

测试内容：
1. 形状验证：确保所有策略输出固定形状 [B, K, 3] 和 [B, D, K]
2. 性能基准：对比推理速度和显存占用
3. Token质量：分析空间覆盖度和分布均匀性
4. 可视化：生成3D散点图对比不同策略

作者: AI Assistant
日期: 2025-10-29
"""

import sys
from pathlib import Path
import time
import logging

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset
from models.backbone.ptv3_sparse_encoder import PTv3SparseEncoder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 数据加载 ====================

def load_test_data(data_cfg_path: str, num_samples: int = 4):
    """加载测试数据"""
    cfg = OmegaConf.load(data_cfg_path)
    
    # 设置必要的默认值
    if '${target_num_grasps}' in str(cfg):
        OmegaConf.update(cfg, "target_num_grasps", 8, force_add=True)
    if '${batch_size}' in str(cfg):
        OmegaConf.update(cfg, "batch_size", 2, force_add=True)
    if '${exhaustive_sampling_strategy}' in str(cfg):
        OmegaConf.update(cfg, "exhaustive_sampling_strategy", "sequential", force_add=True)
    
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    
    test_cfg = cfg.test
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir=test_cfg.succ_grasp_dir,
        obj_root_dir=test_cfg.obj_root_dir,
        num_grasps=test_cfg.num_grasps,
        max_points=test_cfg.max_points,
        max_grasps_per_object=test_cfg.get('max_grasps_per_object', None),
        mesh_scale=test_cfg.mesh_scale,
        grasp_sampling_strategy=test_cfg.grasp_sampling_strategy,
        use_exhaustive_sampling=test_cfg.use_exhaustive_sampling,
        exhaustive_sampling_strategy=test_cfg.exhaustive_sampling_strategy,
        object_sampling_ratio=test_cfg.object_sampling_ratio,
        table_size=test_cfg.table_size,
    )
    
    subset_indices = list(range(min(num_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=ObjectCentricGraspDataset.collate_fn
    )
    
    return dataloader


def create_model_for_strategy(strategy: str, target_tokens: int = 128, device: str = 'cuda'):
    """为指定策略创建模型"""
    cfg = OmegaConf.create({
        'grid_size': 0.003,  # 改为0.003以增加token密度
        'use_flash_attention': True,
        'encoder_channels': [32, 64, 128, 256],
        'encoder_depths': [1, 1, 2, 2],
        'encoder_num_head': [2, 4, 8, 16],
        'enc_patch_size': [1024, 1024, 1024, 1024],
        'stride': [2, 2, 2],
        'out_dim': 256,
        'input_feature_dim': 1,
        'mlp_ratio': 2,
        'target_num_tokens': target_tokens,
        'token_strategy': strategy,
        'grid_resolution': (5, 5, 5) if strategy == 'grid' else (8, 8, 8)
    })
    
    model = PTv3SparseEncoder(cfg, target_num_tokens=target_tokens, token_strategy=strategy)
    model = model.to(device)
    model.eval()
    
    return model


# ==================== 测试模块1：形状验证 ====================

def test_shape_validation(dataloader, strategies, target_tokens=128, device='cuda'):
    """
    测试所有策略的输出形状
    
    验证：
    - xyz 形状是否为 [B, K, 3]
    - features 形状是否为 [B, D, K]
    - K 是否等于 target_tokens
    """
    logger.info("\n" + "="*80)
    logger.info("测试模块1：形状验证")
    logger.info("="*80)
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n测试策略: {strategy}")
        
        try:
            model = create_model_for_strategy(strategy, target_tokens, device)
            
            shape_checks = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    scene_pc = batch['scene_pc'].to(device)
                    B, N, C = scene_pc.shape
                    
                    # 前向传播
                    xyz_out, feat_out = model(scene_pc)
                    
                    # 验证形状
                    B_out, K_out, _ = xyz_out.shape
                    B_feat, D_out, K_feat = feat_out.shape
                    
                    check = {
                        'batch_idx': batch_idx,
                        'input_shape': (B, N, C),
                        'xyz_shape': tuple(xyz_out.shape),
                        'feat_shape': tuple(feat_out.shape),
                        'K_correct': K_out == target_tokens and K_feat == target_tokens,
                        'B_correct': B_out == B and B_feat == B,
                        'xyz_dim_correct': xyz_out.shape[-1] == 3,
                        'feat_dim_correct': D_out == 256,
                        'passed': True
                    }
                    
                    check['passed'] = all([
                        check['K_correct'],
                        check['B_correct'],
                        check['xyz_dim_correct'],
                        check['feat_dim_correct']
                    ])
                    
                    shape_checks.append(check)
                    
                    if batch_idx == 0:  # 只打印第一个batch的详细信息
                        logger.info(f"  输入形状: {check['input_shape']}")
                        logger.info(f"  xyz输出: {check['xyz_shape']} ✓" if check['K_correct'] else f"  xyz输出: {check['xyz_shape']} ✗")
                        logger.info(f"  feat输出: {check['feat_shape']} ✓" if check['feat_dim_correct'] else f"  feat输出: {check['feat_shape']} ✗")
            
            all_passed = all(c['passed'] for c in shape_checks)
            results[strategy] = {
                'passed': all_passed,
                'checks': shape_checks
            }
            
            logger.info(f"  结果: {'✓ 通过' if all_passed else '✗ 失败'}")
            
        except Exception as e:
            logger.error(f"  ✗ 策略 {strategy} 测试失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[strategy] = {'passed': False, 'error': str(e)}
    
    return results


# ==================== 测试模块2：性能基准 ====================

def test_performance_benchmark(dataloader, strategies, target_tokens=128, device='cuda', num_warmup=2, num_runs=5):
    """
    性能基准测试
    
    测量：
    - 前向推理时间（多次运行取平均）
    - GPU显存占用
    - 相对速度对比
    """
    logger.info("\n" + "="*80)
    logger.info("测试模块2：性能基准")
    logger.info("="*80)
    
    results = {}
    
    # 获取一个batch用于测试
    test_batch = next(iter(dataloader))
    scene_pc = test_batch['scene_pc'].to(device)
    
    for strategy in strategies:
        logger.info(f"\n测试策略: {strategy}")
        
        try:
            model = create_model_for_strategy(strategy, target_tokens, device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(scene_pc)
            
            torch.cuda.synchronize()
            
            # 测量推理时间
            timings = []
            with torch.no_grad():
                for _ in range(num_runs):
                    torch.cuda.reset_peak_memory_stats(device)
                    start_time = time.time()
                    
                    xyz_out, feat_out = model(scene_pc)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    elapsed = (end_time - start_time) * 1000  # 转换为毫秒
                    timings.append(elapsed)
                    
                    # 记录显存（只记录一次）
                    if len(timings) == 1:
                        memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            
            avg_time = np.mean(timings)
            std_time = np.std(timings)
            
            results[strategy] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'memory_mb': memory_mb,
                'timings': timings
            }
            
            logger.info(f"  平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
            logger.info(f"  显存占用: {memory_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"  ✗ 策略 {strategy} 性能测试失败: {e}")
            results[strategy] = {'error': str(e)}
    
    # 计算相对速度
    if results:
        baseline_time = results[strategies[0]]['avg_time_ms']
        for strategy in strategies:
            if 'avg_time_ms' in results[strategy]:
                results[strategy]['relative_speed'] = baseline_time / results[strategy]['avg_time_ms']
    
    return results


# ==================== 测试模块3：Token质量分析 ====================

def compute_chamfer_distance(xyz1, xyz2):
    """
    计算Chamfer Distance
    
    Args:
        xyz1: (N, 3)
        xyz2: (M, 3)
    Returns:
        chamfer_dist: float
    """
    # xyz1 到 xyz2 的最近距离
    dist_matrix = torch.cdist(xyz1, xyz2)  # (N, M)
    min_dist_1to2 = dist_matrix.min(dim=1)[0]  # (N,)
    min_dist_2to1 = dist_matrix.min(dim=0)[0]  # (M,)
    
    chamfer = (min_dist_1to2.mean() + min_dist_2to1.mean()) / 2
    return chamfer.item()


def compute_uniformity(xyz):
    """
    计算点分布的均匀性（使用最近邻距离的方差）
    
    Args:
        xyz: (N, 3)
    Returns:
        uniformity: float (越小越均匀)
    """
    dist_matrix = torch.cdist(xyz, xyz)  # (N, N)
    # 排除自己到自己的距离
    dist_matrix = dist_matrix + torch.eye(dist_matrix.shape[0], device=xyz.device) * 1e6
    nearest_dist = dist_matrix.min(dim=1)[0]  # (N,)
    
    uniformity = nearest_dist.std().item() / (nearest_dist.mean().item() + 1e-6)
    return uniformity


def test_token_quality(dataloader, strategies, target_tokens=128, device='cuda'):
    """
    Token质量分析
    
    评估：
    - 覆盖度（Chamfer Distance to input）：越小越好，说明tokens覆盖了原始点云
    - 均匀性（最近邻距离变异系数）：越小越好，说明token分布均匀
    """
    logger.info("\n" + "="*80)
    logger.info("测试模块3：Token质量分析")
    logger.info("="*80)
    
    results = {}
    
    # 获取一个batch
    test_batch = next(iter(dataloader))
    scene_pc = test_batch['scene_pc'].to(device)
    
    for strategy in strategies:
        logger.info(f"\n测试策略: {strategy}")
        
        try:
            model = create_model_for_strategy(strategy, target_tokens, device)
            
            with torch.no_grad():
                xyz_out, feat_out = model(scene_pc)
            
            # 取第一个样本进行分析
            input_xyz = scene_pc[0, :, :3]  # (N, 3)
            output_xyz = xyz_out[0]  # (K, 3)
            
            # 移除填充的零点
            valid_mask = (output_xyz.abs().sum(dim=-1) > 0)
            output_xyz_valid = output_xyz[valid_mask]
            
            # 计算覆盖度（Chamfer Distance）
            coverage = compute_chamfer_distance(input_xyz, output_xyz_valid)
            
            # 计算均匀性
            uniformity = compute_uniformity(output_xyz_valid)
            
            results[strategy] = {
                'coverage': coverage,
                'uniformity': uniformity,
                'num_valid_tokens': valid_mask.sum().item(),
                'num_total_tokens': target_tokens
            }
            
            logger.info(f"  覆盖度 (Chamfer): {coverage:.6f}")
            logger.info(f"  均匀性 (CV): {uniformity:.4f}")
            logger.info(f"  有效token数: {valid_mask.sum().item()} / {target_tokens}")
            
        except Exception as e:
            logger.error(f"  ✗ 策略 {strategy} 质量分析失败: {e}")
            results[strategy] = {'error': str(e)}
    
    return results


# ==================== 测试模块4：可视化对比 ====================

def test_visualization(dataloader, strategies, target_tokens=128, device='cuda', save_dir='tests/visualizations/ptv3_strategies'):
    """
    可视化对比
    
    生成：
    - 每种策略的3D散点图
    - 原始点云（灰色半透明）+ 选中tokens（彩色）
    """
    logger.info("\n" + "="*80)
    logger.info("测试模块4：可视化对比")
    logger.info("="*80)
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 获取一个样本
    test_batch = next(iter(dataloader))
    scene_pc = test_batch['scene_pc'].to(device)
    
    # 只可视化第一个样本
    input_xyz = scene_pc[0, :, :3].cpu().numpy()  # (N, 3)
    
    for strategy in strategies:
        logger.info(f"\n可视化策略: {strategy}")
        
        try:
            model = create_model_for_strategy(strategy, target_tokens, device)
            
            with torch.no_grad():
                xyz_out, _ = model(scene_pc)
            
            output_xyz = xyz_out[0].cpu().numpy()  # (K, 3)
            
            # 移除填充的零点
            valid_mask = np.abs(output_xyz).sum(axis=-1) > 0
            output_xyz_valid = output_xyz[valid_mask]
            
            # 创建图形
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制原始点云（灰色半透明）
            ax.scatter(
                input_xyz[:, 0], input_xyz[:, 1], input_xyz[:, 2],
                c='gray', alpha=0.1, s=1, label='原始点云'
            )
            
            # 绘制选中的tokens（彩色）
            ax.scatter(
                output_xyz_valid[:, 0], output_xyz_valid[:, 1], output_xyz_valid[:, 2],
                c=output_xyz_valid[:, 2], cmap='viridis', s=50, 
                edgecolors='black', linewidth=0.5,
                label=f'选中tokens ({len(output_xyz_valid)})'
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'策略: {strategy} | Tokens: {len(output_xyz_valid)}/{target_tokens}')
            ax.legend()
            
            # 设置相同的坐标轴范围
            all_points = np.vstack([input_xyz, output_xyz_valid])
            margin = 0.1
            ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
            ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
            ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
            
            # 保存图片
            save_file = save_path / f"strategy_{strategy}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ 保存到: {save_file}")
            
        except Exception as e:
            logger.error(f"  ✗ 策略 {strategy} 可视化失败: {e}")
    
    logger.info(f"\n所有可视化结果保存在: {save_path}")


# ==================== 主测试流程 ====================

def generate_summary_report(shape_results, perf_results, quality_results, strategies):
    """生成汇总报告"""
    logger.info("\n" + "="*120)
    logger.info("汇总报告")
    logger.info("="*120)
    
    # 表1：形状验证
    logger.info("\n【表1：形状验证结果】")
    logger.info(f"{'策略':<20} {'输出形状':<25} {'是否通过':<10}")
    logger.info("-"*60)
    for strategy in strategies:
        if strategy in shape_results:
            passed = shape_results[strategy].get('passed', False)
            if passed and 'checks' in shape_results[strategy]:
                first_check = shape_results[strategy]['checks'][0]
                xyz_shape = str(first_check['xyz_shape'])
                feat_shape = str(first_check['feat_shape'])
                status = "✓ 通过" if passed else "✗ 失败"
            else:
                xyz_shape = "错误"
                feat_shape = ""
                status = "✗ 失败"
            
            logger.info(f"{strategy:<20} {xyz_shape:<25} {status:<10}")
    
    # 表2：性能对比
    logger.info("\n【表2：性能对比】")
    logger.info(f"{'策略':<20} {'推理时间(ms)':<20} {'显存(MB)':<15} {'相对速度':<10}")
    logger.info("-"*70)
    for strategy in strategies:
        if strategy in perf_results and 'avg_time_ms' in perf_results[strategy]:
            res = perf_results[strategy]
            time_str = f"{res['avg_time_ms']:.2f} ± {res['std_time_ms']:.2f}"
            memory_str = f"{res['memory_mb']:.1f}"
            speed_str = f"{res.get('relative_speed', 1.0):.2f}x"
            logger.info(f"{strategy:<20} {time_str:<20} {memory_str:<15} {speed_str:<10}")
    
    # 表3：质量分析
    logger.info("\n【表3：Token质量分析】")
    logger.info(f"{'策略':<20} {'覆盖度(↓)':<15} {'均匀性(↓)':<15} {'有效Token':<15}")
    logger.info("-"*70)
    for strategy in strategies:
        if strategy in quality_results and 'coverage' in quality_results[strategy]:
            res = quality_results[strategy]
            coverage_str = f"{res['coverage']:.6f}"
            uniformity_str = f"{res['uniformity']:.4f}"
            tokens_str = f"{res['num_valid_tokens']}/{res['num_total_tokens']}"
            logger.info(f"{strategy:<20} {coverage_str:<15} {uniformity_str:<15} {tokens_str:<15}")
    
    # 综合评分
    logger.info("\n【综合评价】")
    logger.info("方案①(last_layer): 实现简单，适合快速原型")
    logger.info("方案②(fps):         推荐首选，固定输出且空间均匀")
    logger.info("方案③(grid):        空间规整，适合结构化场景")
    logger.info("方案④(learned):     可学习，需要训练（当前随机初始化）")
    logger.info("方案⑤(multiscale):  多尺度信息，适合复杂任务")
    logger.info("="*120)


def main():
    """主测试函数"""
    logger.info("="*120)
    logger.info("PTv3 五种 Token 策略综合测试")
    logger.info("="*120)
    
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_tokens = 128
    strategies = ['last_layer', 'fps', 'grid', 'learned', 'multiscale']
    
    logger.info(f"\n设备: {device}")
    logger.info(f"目标Token数: {target_tokens}")
    logger.info(f"测试策略: {strategies}\n")
    
    # 加载数据
    data_cfg_path = project_root / "config/data_cfg/objectcentric.yaml"
    logger.info("加载测试数据...")
    try:
        dataloader = load_test_data(str(data_cfg_path), num_samples=4)
        logger.info(f"✓ 数据加载成功\n")
    except Exception as e:
        logger.error(f"✗ 数据加载失败: {e}")
        return
    
    # 运行测试
    try:
        # 模块1：形状验证
        shape_results = test_shape_validation(dataloader, strategies, target_tokens, device)
        
        # 模块2：性能基准
        perf_results = test_performance_benchmark(dataloader, strategies, target_tokens, device)
        
        # 模块3：质量分析
        quality_results = test_token_quality(dataloader, strategies, target_tokens, device)
        
        # 模块4：可视化
        test_visualization(dataloader, strategies, target_tokens, device)
        
        # 生成汇总报告
        generate_summary_report(shape_results, perf_results, quality_results, strategies)
        
        logger.info("\n✓ 所有测试完成！")
        
    except Exception as e:
        logger.error(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()

