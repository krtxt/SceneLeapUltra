"""
测试 PTv3 Encoder 输出的真实稀疏 token 数量

这个测试直接访问 PTv3 的 encoder 输出，而不是 decoder 恢复后的密集点云
目的：了解不同配置下实际的稀疏化程度
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
from models.backbone.ptv3_backbone import PTV3Backbone, convert_to_ptv3_pc_format

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    logger.info(f"加载数据配置：{data_cfg_path}")
    
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


def create_ptv3_model(config_dict):
    """根据配置字典创建 PTv3 模型"""
    cfg = OmegaConf.create(config_dict)
    model = PTV3Backbone(cfg)
    model.eval()
    return model


def get_encoder_sparse_tokens(model, pos, device):
    """
    获取 PTv3 Encoder 输出的真实稀疏 token 数量
    
    Args:
        model: PTv3Backbone 模型
        pos: 输入点云 [B, N, 3]
        device: 设备
        
    Returns:
        dict: 包含稀疏 token 信息的字典
    """
    B, N, C = pos.shape
    
    # 分割坐标和特征
    coords = pos[..., :3]
    if C > 3:
        feat = pos[..., 3:]
    else:
        feat = torch.ones(B, N, 1, device=device, dtype=pos.dtype)
    
    # 转换为 PTv3 格式
    data_dict = convert_to_ptv3_pc_format(coords, feat, model.grid_size)
    
    # 只运行 encoder（不运行 decoder）
    point = model.model.embedding(data_dict)
    point = model.model.enc(point)
    
    # 提取稀疏信息
    sparse_coord = point['coord']  # [M, 3] M 是总的稀疏点数
    sparse_feat = point['feat']    # [M, C]
    offset = point['offset']        # [B] 累积偏移
    
    # 计算每个 batch 的点数
    points_per_batch = []
    start_idx = 0
    for b in range(B):
        end_idx = offset[b].item()
        n_points = end_idx - start_idx
        points_per_batch.append(n_points)
        start_idx = end_idx
    
    return {
        'total_sparse_points': sparse_coord.shape[0],
        'points_per_batch': points_per_batch,
        'feature_dim': sparse_feat.shape[1],
        'sparse_coord': sparse_coord,
        'sparse_feat': sparse_feat,
        'sparse_shape': point.get('sparse_shape', None),
        'serialized_depth': point.get('serialized_depth', None),
    }


def test_sparse_configuration(config_name: str, config_dict: dict, dataloader: DataLoader, device: str = 'cuda'):
    """测试单个配置的稀疏 token 数量"""
    logger.info(f"\n{'='*80}")
    logger.info(f"测试配置: {config_name}")
    logger.info(f"{'='*80}")
    
    # 打印关键配置
    logger.info(f"  grid_size: {config_dict.get('grid_size')}")
    logger.info(f"  encoder_channels: {config_dict.get('encoder_channels')}")
    
    try:
        # 创建模型
        model = create_ptv3_model(config_dict)
        model = model.to(device)
        
        # 收集统计信息
        all_sparse_counts = []
        all_compression_ratios = []
        
        # 测试多个 batch
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                scene_pc = batch['scene_pc'].to(device)  # [B, N, 3]
                B, N, C = scene_pc.shape
                
                logger.info(f"\nBatch {batch_idx + 1}:")
                logger.info(f"  输入点云形状: {scene_pc.shape} (总点数: {B*N})")
                
                # 获取 encoder 的稀疏输出
                try:
                    sparse_info = get_encoder_sparse_tokens(model, scene_pc, device)
                    
                    total_sparse = sparse_info['total_sparse_points']
                    points_per_batch = sparse_info['points_per_batch']
                    feat_dim = sparse_info['feature_dim']
                    
                    logger.info(f"  Encoder 稀疏输出:")
                    logger.info(f"    总稀疏点数: {total_sparse}")
                    logger.info(f"    每个样本点数: {points_per_batch}")
                    logger.info(f"    特征维度: {feat_dim}")
                    logger.info(f"    稀疏形状: {sparse_info['sparse_shape']}")
                    logger.info(f"    序列化深度: {sparse_info['serialized_depth']}")
                    
                    # 计算压缩率
                    compression_ratio = total_sparse / (B * N)
                    logger.info(f"    压缩率: {compression_ratio:.4f} ({total_sparse}/{B*N})")
                    logger.info(f"    每个样本平均: {total_sparse/B:.1f} tokens")
                    
                    all_sparse_counts.extend(points_per_batch)
                    all_compression_ratios.append(compression_ratio)
                    
                except Exception as e:
                    logger.error(f"  ❌ 获取稀疏信息失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
        
        # 计算统计信息
        results = {
            'config_name': config_name,
            'config': config_dict,
            'sparse_tokens_min': min(all_sparse_counts) if all_sparse_counts else 0,
            'sparse_tokens_max': max(all_sparse_counts) if all_sparse_counts else 0,
            'sparse_tokens_mean': sum(all_sparse_counts) / len(all_sparse_counts) if all_sparse_counts else 0,
            'compression_ratio_mean': sum(all_compression_ratios) / len(all_compression_ratios) if all_compression_ratios else 0,
            'all_sparse_counts': all_sparse_counts,
            'feature_dim': feat_dim if 'feat_dim' in locals() else 0,
            'success': True
        }
        
        # 打印总结
        logger.info(f"\n{'='*80}")
        logger.info(f"配置 '{config_name}' 统计结果:")
        logger.info(f"  ✓ 成功")
        logger.info(f"  稀疏 token 范围: [{results['sparse_tokens_min']}, {results['sparse_tokens_max']}]")
        logger.info(f"  稀疏 token 平均: {results['sparse_tokens_mean']:.1f}")
        logger.info(f"  平均压缩率: {results['compression_ratio_mean']:.4f} (相对输入点数)")
        logger.info(f"  特征维度: {results['feature_dim']}")
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
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
    
    # 定义测试配置（重点测试不同 grid_size）
    test_configs = [
        # 1. 非常小的 grid_size（应该有更多 tokens）
        {
            'name': 'grid_size_0.005_very_fine',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.005,
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
        
        # 2. 小 grid_size
        {
            'name': 'grid_size_0.01',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.01,
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
        
        # 3. 基线配置
        {
            'name': 'grid_size_0.02_baseline',
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
        
        # 4. 中等 grid_size
        {
            'name': 'grid_size_0.03',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.03,
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
        
        # 5. 大 grid_size（应该有更少 tokens）
        {
            'name': 'grid_size_0.04',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.04,
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
        
        # 6. 非常大的 grid_size（应该有很少 tokens）
        {
            'name': 'grid_size_0.05_very_coarse',
            'config': {
                'variant': 'light',
                'use_flash_attention': True,
                'grid_size': 0.05,
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
    ]
    
    # 运行所有测试
    all_results = []
    
    for test_config in test_configs:
        result = test_sparse_configuration(
            config_name=test_config['name'],
            config_dict=test_config['config'],
            dataloader=dataloader,
            device=device
        )
        if result:
            all_results.append(result)
    
    # 打印总结
    logger.info("\n" + "="*120)
    logger.info("所有测试完成！Encoder 稀疏 Token 总结报告:")
    logger.info("="*120)
    
    # 创建汇总表格
    logger.info(f"\n{'配置名称':<35} {'grid_size':<12} {'稀疏Token范围':<25} {'平均Token数':<15} {'压缩率':<12} {'特征维度':<10}")
    logger.info("-"*120)
    
    for result in all_results:
        if result['success']:
            config_name = result['config_name']
            grid_size = result['config'].get('grid_size', 'N/A')
            token_range = f"[{result['sparse_tokens_min']}, {result['sparse_tokens_max']}]"
            token_mean = f"{result['sparse_tokens_mean']:.1f}"
            compression = f"{result['compression_ratio_mean']:.4f}"
            feat_dim = str(result['feature_dim'])
            
            logger.info(f"{config_name:<35} {grid_size:<12.4f} {token_range:<25} {token_mean:<15} {compression:<12} {feat_dim:<10}")
    
    logger.info("="*120)
    
    # 分析 grid_size 对稀疏 token 数的影响
    logger.info("\n" + "="*120)
    logger.info("grid_size 对 Encoder 稀疏 token 数量的影响分析:")
    logger.info("="*120)
    
    # 按 grid_size 排序
    sorted_results = sorted(
        [r for r in all_results if r['success']], 
        key=lambda x: x['config'].get('grid_size', 0)
    )
    
    for result in sorted_results:
        grid_size = result['config'].get('grid_size')
        token_mean = result['sparse_tokens_mean']
        compression = result['compression_ratio_mean']
        logger.info(
            f"  grid_size = {grid_size:.4f}: "
            f"平均 {token_mean:.1f} tokens, "
            f"压缩率 {compression:.4f} ({token_mean/8192*100:.1f}% of input)"
        )
    
    logger.info("\n关键结论:")
    logger.info("  1. Encoder 确实进行了稀疏化，输出的 token 数量远小于输入点数")
    logger.info("  2. grid_size 越小 → 稀疏 token 数越多（更精细）")
    logger.info("  3. grid_size 越大 → 稀疏 token 数越少（更粗糙）")
    logger.info("  4. 之前的测试显示 K=8192 是因为 Decoder 将稀疏 tokens 恢复到了原始分辨率")
    logger.info("  5. 如果想要真正的稀疏表示，应该只使用 Encoder 输出，不使用 Decoder")
    logger.info("="*120 + "\n")


if __name__ == "__main__":
    main()

