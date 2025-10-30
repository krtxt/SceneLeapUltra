#!/usr/bin/env python3
"""
测试不同点云骨干网络处理真实点云数据后的输出shape

运行方式:
cd /home/xiantuo/source/grasp/GithubClone/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python tests/test_backbone_output_shapes.py
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'bin'))

# 设置grasp_gen导入路径
import types

def setup_grasp_gen_imports():
    """设置grasp_gen包的导入路径"""
    # 创建grasp_gen模块
    if 'grasp_gen' not in sys.modules:
        grasp_gen_module = types.ModuleType('grasp_gen')
        sys.modules['grasp_gen'] = grasp_gen_module
    
    # 创建grasp_gen.models
    if 'grasp_gen.models' not in sys.modules:
        models_module = types.ModuleType('grasp_gen.models')
        sys.modules['grasp_gen.models'] = models_module
    
    # 创建grasp_gen.models.pointnet (需要设置为包)
    if 'grasp_gen.models.pointnet' not in sys.modules:
        pointnet_module = types.ModuleType('grasp_gen.models.pointnet')
        pointnet_module.__path__ = [os.path.join(project_root, 'bin', 'grasp_gen_models', 'pointnet')]
        sys.modules['grasp_gen.models.pointnet'] = pointnet_module
    
    # 创建grasp_gen.models.ptv3 (需要设置为包)
    if 'grasp_gen.models.ptv3' not in sys.modules:
        ptv3_module = types.ModuleType('grasp_gen.models.ptv3')
        ptv3_module.__path__ = [os.path.join(project_root, 'bin', 'grasp_gen_models', 'ptv3')]
        sys.modules['grasp_gen.models.ptv3'] = ptv3_module
    
    # 创建grasp_gen.utils
    if 'grasp_gen.utils' not in sys.modules:
        utils_module = types.ModuleType('grasp_gen.utils')
        sys.modules['grasp_gen.utils'] = utils_module
    
    # 创建grasp_gen.utils.logging_config
    if 'grasp_gen.utils.logging_config' not in sys.modules:
        logging_config_module = types.ModuleType('grasp_gen.utils.logging_config')
        import logging
        logging.basicConfig(level=logging.INFO)
        def get_logger(name):
            return logging.getLogger(name)
        logging_config_module.get_logger = get_logger
        sys.modules['grasp_gen.utils.logging_config'] = logging_config_module
    
    # 导入并映射pointnet2_utils
    try:
        from bin.grasp_gen_models.pointnet import pointnet2_utils
        sys.modules['grasp_gen.models.pointnet.pointnet2_utils'] = pointnet2_utils
    except Exception as e:
        print(f"Warning: Could not import pointnet2_utils: {e}")
    
    # 导入并映射pointnet2_modules
    try:
        from bin.grasp_gen_models.pointnet import pointnet2_modules
        sys.modules['grasp_gen.models.pointnet.pointnet2_modules'] = pointnet2_modules
    except Exception as e:
        print(f"Warning: Could not import pointnet2_modules: {e}")

setup_grasp_gen_imports()

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset

# 只导入PTV3相关的，避免导入pointnet2相关依赖
from bin.grasp_gen_models.ptv3.ptv3 import PointTransformerV3

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_ptv3_pc_format(point_cloud: torch.Tensor, grid_size: float = 0.01):
    """
    将点云转换为PTV3所需格式
    复制自 bin/grasp_gen_models/model_utils.py，避免导入pointnet2依赖
    """
    device = point_cloud.device
    batch_size = point_cloud.shape[0]
    num_points = point_cloud.shape[1]
    data_dict = dict()
    data_dict["coord"] = point_cloud.reshape([-1, 3]).to(device)
    data_dict["grid_size"] = grid_size
    data_dict["feat"] = point_cloud.reshape([-1, 3]).to(device)
    data_dict["offset"] = (
        torch.tensor([num_points]).repeat(batch_size).cumsum(dim=0).to(device)
    )
    return data_dict


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
        num_workers=0,  # 设置为0避免多进程问题
        collate_fn=ObjectCentricGraspDataset.collate_fn,
    )
    
    return dataloader


def create_ptv3_backbone(
    output_dim: int = 512,
    grid_size: float = 0.01,
    enable_flash: bool = False,
    device: str = 'cuda',
):
    """
    创建PTV3骨干网络
    
    Args:
        output_dim: 输出特征维度（通过修改最后一层实现）
        grid_size: 网格大小
        enable_flash: 是否启用Flash Attention
        device: 设备
        
    Returns:
        PointTransformerV3模型和grid_size
    """
    # 根据output_dim设置encoder_channels的最后一层
    # 标准配置是512，如果需要其他维度，需要添加投影层
    enc_channels = [32, 64, 128, 256, output_dim]
    
    model = PointTransformerV3(
        in_channels=3,  # 只有xyz坐标
        enc_channels=enc_channels,
        enc_depths=(2, 2, 2, 2, 2),
        enc_num_head=(2, 4, 8, 16, 16),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        cls_mode=True,  # 分类模式，输出全局特征
        enable_flash=enable_flash,
    )
    model = model.to(device)
    model.eval()
    return model, grid_size


def test_ptv3_output_shape(
    dataloader,
    output_dim: int = 512,
    grid_sizes: list = [0.001, 0.003, 0.01, 0.02, 0.03],
    enable_flash: bool = False,
    device: str = 'cuda',
):
    """
    测试PTV3在不同grid_size下的输出shape
    
    Args:
        dataloader: 数据加载器
        output_dim: 输出特征维度
        grid_sizes: 要测试的grid_size列表
        enable_flash: 是否启用Flash Attention
        device: 设备
    """
    print("=" * 80)
    print("测试 PTV3 Backbone (不同grid_size)")
    print("=" * 80)
    
    results = {}
    
    for grid_size in grid_sizes:
        print(f"\n{'=' * 80}")
        print(f"测试 grid_size = {grid_size}")
        print(f"{'=' * 80}")
        
        model, _ = create_ptv3_backbone(
            output_dim=output_dim,
            grid_size=grid_size,
            enable_flash=enable_flash,
            device=device,
        )
        
        # 统计模型参数量（只统计一次，因为模型结构相同）
        if grid_size == grid_sizes[0]:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        all_shapes = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                points = batch['scene_pc'].to(device)  # (B, N, 3)
                
                print(f"\n批次 {batch_idx + 1}:")
                print(f"  输入点云shape: {points.shape}")
                
                # 转换为PTV3格式
                try:
                    ptv3_data = convert_to_ptv3_pc_format(points, grid_size=grid_size)
                    
                    print(f"  PTV3数据格式:")
                    print(f"    coord shape: {ptv3_data['coord'].shape}")
                    print(f"    feat shape: {ptv3_data['feat'].shape}")
                    print(f"    offset: {ptv3_data['offset']}")
                    print(f"    grid_size: {ptv3_data['grid_size']}")
                    
                    # PTV3 forward
                    features = model(ptv3_data)
                    
                    # PTV3在cls_mode=True时，forward返回的是torch.Tensor (B, enc_channels[-1])
                    print(f"  输出特征shape: {features.shape}")
                    print(f"  预期输出shape: ({points.shape[0]}, {output_dim})")
                    
                    # 验证输出shape
                    if isinstance(features, torch.Tensor):
                        actual_shape = features.shape
                        expected_shape = (points.shape[0], output_dim)
                        if actual_shape != expected_shape:
                            print(f"  ⚠ 警告: 输出shape不匹配！期望 {expected_shape}, 实际 {actual_shape}")
                            print(f"  实际输出维度: {actual_shape[1]}")
                        else:
                            print(f"  ✓ 输出shape匹配！")
                    else:
                        print(f"  ✗ 错误: 输出类型为 {type(features)}，期望torch.Tensor")
                    
                    all_shapes.append(features.shape if isinstance(features, torch.Tensor) else None)
                    
                except Exception as e:
                    print(f"  ✗ 错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    all_shapes.append(None)
                    continue
                
                if batch_idx >= 2:  # 只测试前3个批次
                    break
        
        results[grid_size] = all_shapes
        successful_shapes = [s for s in all_shapes if s is not None]
        if successful_shapes:
            print(f"\n✓ grid_size={grid_size} 测试完成！输出shapes: {successful_shapes}")
        else:
            print(f"\n✗ grid_size={grid_size} 测试失败，无成功输出")
    
    return results


def main():
    """主测试函数"""
    print("=" * 80)
    print("点云骨干网络输出Shape测试")
    print("=" * 80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 设置输出特征维度
    output_dim = 512
    print(f"输出特征维度: {output_dim}")
    
    # 加载数据
    data_cfg_path = "config/data_cfg/objectcentric.yaml"
    print(f"\n加载数据集配置: {data_cfg_path}")
    
    try:
        dataloader = load_test_data(data_cfg_path, num_samples=4)
        print(f"✓ 数据集加载成功")
    except Exception as e:
        print(f"✗ 数据集加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试PTV3（不同grid_size）
    print("\n" + "=" * 80)
    print("测试 PTV3 (不同grid_size)")
    print("=" * 80)
    
    # 测试不同的grid_size
    grid_sizes = [0.001, 0.003, 0.01, 0.02, 0.03]
    
    try:
        ptv3_results = test_ptv3_output_shape(
            dataloader,
            output_dim=output_dim,
            grid_sizes=grid_sizes,
            enable_flash=False,  # 设置为False避免Flash Attention依赖问题
            device=device,
        )
    except Exception as e:
        print(f"✗ PTV3测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        ptv3_results = None
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    print(f"\n输出特征维度: {output_dim}")
    
    if ptv3_results:
        print(f"\n✓ PTV3 (不同grid_size):")
        for grid_size, shapes in ptv3_results.items():
            successful_shapes = [s for s in shapes if s is not None]
            if successful_shapes:
                unique_shapes = list(set(successful_shapes))
                print(f"  grid_size={grid_size}:")
                print(f"    输出shapes: {unique_shapes}")
                if all(s == (2, output_dim) for s in successful_shapes):
                    print(f"    ✓ 所有批次输出shape一致: (batch_size, {output_dim})")
                else:
                    print(f"    ⚠ 警告: 输出shapes不一致或维度不匹配")
            else:
                print(f"  grid_size={grid_size}: ✗ 测试失败")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

