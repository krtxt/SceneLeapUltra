"""
测试 PTv3 骨干网络在不同配置下输出的 scene tokens 数量 K

测试场景：
1. 不同的 grid_size 值
2. 不同的 encoder_channels 配置
3. 不同的 encoder_depths 配置
4. 组合配置测试

输出格式：[B, K, D] 中的 K 值
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset
from models.backbone.ptv3_backbone import PTV3Backbone

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(data_cfg_path: str, num_samples: int = 4):
    """
    加载测试数据
    
    Args:
        data_cfg_path: 数据配置文件路径
        num_samples: 加载的样本数量
        
    Returns:
        DataLoader: 测试数据加载器
    """
    # 加载配置
    cfg = OmegaConf.load(data_cfg_path)
    
    # 设置必要的默认值（如果配置中有占位符）
    if '${target_num_grasps}' in str(cfg):
        OmegaConf.update(cfg, "target_num_grasps", 8, force_add=True)
    if '${batch_size}' in str(cfg):
        OmegaConf.update(cfg, "batch_size", 2, force_add=True)
    if '${exhaustive_sampling_strategy}' in str(cfg):
        OmegaConf.update(cfg, "exhaustive_sampling_strategy", "sequential", force_add=True)
    
    # 解析占位符
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    
    logger.info(f"加载数据配置：{data_cfg_path}")
    logger.info(f"使用 test 模式配置")
    
    # 创建数据集
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
    
    logger.info(f"数据集大小：{len(dataset)}")
    
    # 创建 DataLoader（限制样本数量）
    subset_indices = list(range(min(num_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    dataloader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # 使用单线程便于调试
        collate_fn=ObjectCentricGraspDataset.collate_fn
    )
    
    return dataloader


def create_ptv3_model(config_dict):
    """
    根据配置字典创建 PTv3 模型
    
    Args:
        config_dict: 配置字典
        
    Returns:
        PTV3Backbone: PTv3 骨干网络模型
    """
    # 转换为 OmegaConf 对象以支持属性访问
    cfg = OmegaConf.create(config_dict)
    model = PTV3Backbone(cfg)
    model.eval()  # 设置为评估模式
    return model


def test_configuration(config_name: str, config_dict: dict, dataloader: DataLoader, device: str = 'cuda'):
    """
    测试单个配置
    
    Args:
        config_name: 配置名称
        config_dict: 配置字典
        dataloader: 数据加载器
        device: 设备（cuda 或 cpu）
        
    Returns:
        dict: 测试结果统计
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"测试配置: {config_name}")
    logger.info(f"{'='*80}")
    
    # 打印配置详情
    logger.info("配置参数:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    try:
        # 创建模型
        model = create_ptv3_model(config_dict)
        model = model.to(device)
        
        # 收集统计信息
        k_values = []
        batch_sizes = []
        feature_dims = []
        
        # 测试多个 batch
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 获取点云数据
                scene_pc = batch['scene_pc'].to(device)  # [B, N, 3]
                B, N, C = scene_pc.shape
                
                logger.info(f"\nBatch {batch_idx + 1}:")
                logger.info(f"  输入点云形状: {scene_pc.shape}")
                
                # 前向传播
                try:
                    xyz_out, feat_out = model(scene_pc)
                    
                    # 记录输出形状
                    B_out, K, _ = xyz_out.shape
                    _, D, K_feat = feat_out.shape
                    
                    logger.info(f"  输出 xyz 形状: {xyz_out.shape} -> [B={B_out}, K={K}, 3]")
                    logger.info(f"  输出特征形状: {feat_out.shape} -> [B={B_out}, D={D}, K={K_feat}]")
                    
                    # 验证一致性
                    assert K == K_feat, f"K 值不一致: xyz 中 K={K}, features 中 K={K_feat}"
                    
                    k_values.append(K)
                    batch_sizes.append(B_out)
                    feature_dims.append(D)
                    
                except Exception as e:
                    logger.error(f"  ❌ 前向传播失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
        
        # 计算统计信息
        results = {
            'config_name': config_name,
            'config': config_dict,
            'k_min': min(k_values) if k_values else 0,
            'k_max': max(k_values) if k_values else 0,
            'k_mean': sum(k_values) / len(k_values) if k_values else 0,
            'k_values': k_values,
            'feature_dim': feature_dims[0] if feature_dims else 0,
            'success': True
        }
        
        # 打印总结
        logger.info(f"\n{'='*80}")
        logger.info(f"配置 '{config_name}' 测试结果:")
        logger.info(f"  ✓ 成功")
        logger.info(f"  K 值范围: [{results['k_min']}, {results['k_max']}]")
        logger.info(f"  K 平均值: {results['k_mean']:.1f}")
        logger.info(f"  特征维度 D: {results['feature_dim']}")
        logger.info(f"  所有 K 值: {k_values}")
        logger.info(f"{'='*80}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 配置 '{config_name}' 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'config_name': config_name,
            'config': config_dict,
            'success': False,
            'error': str(e)
        }


def main():
    """主测试函数"""
    
    # 检查 CUDA 是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if device == 'cpu':
        logger.warning("⚠️  CUDA 不可用，使用 CPU 进行测试（速度可能较慢）")
    
    # 加载测试数据
    data_cfg_path = project_root / "config/data_cfg/objectcentric.yaml"
    logger.info(f"加载测试数据...")
    
    try:
        dataloader = load_test_data(str(data_cfg_path), num_samples=4)
        logger.info(f"✓ 数据加载成功，共 {len(dataloader)} 个 batch\n")
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 定义测试配置
    test_configs = [
        # 1. 基础配置（来自 ptv3_light.yaml）
        {
            'name': 'ptv3_light_baseline',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.02,
                'encoder_channels': [32, 64, 128, 256, 512],
                'encoder_depths': [1, 1, 2, 2, 1],
                'encoder_num_head': [2, 4, 8, 16, 16],
                'enc_patch_size': [1024, 1024, 1024, 1024, 1024],
                'decoder_channels': [512, 256, 128, 64],
                'decoder_depths': [1, 2, 1, 1],
                'dec_patch_size': [1024, 1024, 1024, 1024],
                'dec_num_head': [4, 4, 8, 16],
                'mlp_ratio': 2,
                'out_dim': 512,
                'input_feature_dim': 1,
            }
        },
        
        # 2. 测试不同 grid_size（更小 -> 更多 tokens）
        {
            'name': 'grid_size_0.01',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.01,  # 减小 grid_size
                'encoder_channels': [32, 64, 128, 256, 512],
                'encoder_depths': [1, 1, 2, 2, 1],
                'encoder_num_head': [2, 4, 8, 16, 16],
                'enc_patch_size': [1024, 1024, 1024, 1024, 1024],
                'decoder_channels': [512, 256, 128, 64],
                'decoder_depths': [1, 2, 1, 1],
                'dec_patch_size': [1024, 1024, 1024, 1024],
                'dec_num_head': [4, 4, 8, 16],
                'mlp_ratio': 2,
                'out_dim': 512,
                'input_feature_dim': 1,
            }
        },
        
        # 3. 测试不同 grid_size（更大 -> 更少 tokens）
        {
            'name': 'grid_size_0.04',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.04,  # 增大 grid_size
                'encoder_channels': [32, 64, 128, 256, 512],
                'encoder_depths': [1, 1, 2, 2, 1],
                'encoder_num_head': [2, 4, 8, 16, 16],
                'enc_patch_size': [1024, 1024, 1024, 1024, 1024],
                'decoder_channels': [512, 256, 128, 64],
                'decoder_depths': [1, 2, 1, 1],
                'dec_patch_size': [1024, 1024, 1024, 1024],
                'dec_num_head': [4, 4, 8, 16],
                'mlp_ratio': 2,
                'out_dim': 512,
                'input_feature_dim': 1,
            }
        },
        
        # 4. 极轻量配置（注释中的 extremely light version）
        {
            'name': 'extremely_light',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.02,
                'encoder_channels': [16, 32, 64, 128, 256],
                'encoder_depths': [1, 1, 2, 2, 1],
                'encoder_num_head': [2, 4, 8, 8, 8],
                'enc_patch_size': [1024, 1024, 1024, 1024, 1024],
                'decoder_channels': [256, 128, 64, 32],
                'decoder_depths': [1, 1, 1, 1],
                'dec_patch_size': [1024, 1024, 1024, 1024],
                'dec_num_head': [2, 4, 8, 8],
                'mlp_ratio': 2,
                'out_dim': 256,
                'input_feature_dim': 1,
            }
        },
        
        # 5. 测试更深的网络
        {
            'name': 'deeper_network',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.02,
                'encoder_channels': [32, 64, 128, 256, 512],
                'encoder_depths': [2, 2, 4, 4, 2],  # 增加深度
                'encoder_num_head': [2, 4, 8, 16, 16],
                'enc_patch_size': [1024, 1024, 1024, 1024, 1024],
                'decoder_channels': [512, 256, 128, 64],
                'decoder_depths': [2, 4, 2, 2],  # 增加深度
                'dec_patch_size': [1024, 1024, 1024, 1024],
                'dec_num_head': [4, 4, 8, 16],
                'mlp_ratio': 2,
                'out_dim': 512,
                'input_feature_dim': 1,
            }
        },
    ]
    
    # 运行所有测试
    all_results = []
    
    for test_config in test_configs:
        result = test_configuration(
            config_name=test_config['name'],
            config_dict=test_config['config'],
            dataloader=dataloader,
            device=device
        )
        if result:
            all_results.append(result)
    
    # 打印总结
    logger.info("\n" + "="*100)
    logger.info("所有测试完成！总结报告:")
    logger.info("="*100)
    
    # 创建汇总表格
    logger.info(f"\n{'配置名称':<30} {'K值范围':<20} {'K均值':<15} {'特征维度D':<15} {'状态':<10}")
    logger.info("-"*100)
    
    for result in all_results:
        if result['success']:
            config_name = result['config_name']
            k_range = f"[{result['k_min']}, {result['k_max']}]"
            k_mean = f"{result['k_mean']:.1f}"
            feat_dim = str(result['feature_dim'])
            status = "✓"
            
            logger.info(f"{config_name:<30} {k_range:<20} {k_mean:<15} {feat_dim:<15} {status:<10}")
        else:
            config_name = result['config_name']
            logger.info(f"{config_name:<30} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'✗':<10}")
            logger.info(f"  错误: {result.get('error', 'Unknown error')}")
    
    logger.info("="*100)
    
    # 分析 grid_size 对 K 的影响
    logger.info("\n" + "="*100)
    logger.info("grid_size 对 token 数量 K 的影响分析:")
    logger.info("="*100)
    
    grid_size_results = [
        (0.01, next((r for r in all_results if r['config_name'] == 'grid_size_0.01'), None)),
        (0.02, next((r for r in all_results if r['config_name'] == 'ptv3_light_baseline'), None)),
        (0.04, next((r for r in all_results if r['config_name'] == 'grid_size_0.04'), None)),
    ]
    
    for grid_size, result in grid_size_results:
        if result and result['success']:
            logger.info(f"  grid_size = {grid_size:.3f}: K 均值 = {result['k_mean']:.1f}, K 范围 = [{result['k_min']}, {result['k_max']}]")
    
    logger.info("\n结论:")
    logger.info("  - grid_size 越小，生成的 token 数量 K 越多（更精细的空间分辨率）")
    logger.info("  - grid_size 越大，生成的 token 数量 K 越少（更粗糙的空间分辨率）")
    logger.info("  - encoder/decoder 的深度和通道数不直接影响 K，但影响特征维度 D 和计算复杂度")
    logger.info("="*100 + "\n")


if __name__ == "__main__":
    main()

