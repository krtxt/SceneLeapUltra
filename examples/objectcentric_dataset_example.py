"""
ObjectCentricGraspDataset 使用示例

演示如何使用ObjectCentricGraspDataset进行数据加载和可视化。

运行方式:
cd /home/xiantuo/source/grasp/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python examples/objectcentric_dataset_example.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset


def example_basic_usage():
    """示例1: 基本使用"""
    print("=" * 80)
    print("示例1: 基本使用")
    print("=" * 80)
    
    # 创建数据集
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=8,
        max_points=4096,
        mesh_scale=0.1,
        grasp_sampling_strategy="random"
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"物体数量: {len(dataset.hand_pose_data)}")
    
    # 获取单个样本
    sample = dataset[0]
    
    print("\n样本数据:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")


def example_with_dataloader():
    """示例2: 使用DataLoader"""
    print("\n" + "=" * 80)
    print("示例2: 使用DataLoader")
    print("=" * 80)
    
    # 创建数据集
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=2048,
        mesh_scale=0.1,
        grasp_sampling_strategy="farthest_point"
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # 使用0避免多进程问题
        collate_fn=dataset.collate_fn
    )
    
    print(f"DataLoader配置:")
    print(f"  Batch size: 8")
    print(f"  Num batches: {len(dataloader)}")
    
    # 迭代几个batch
    print("\n迭代前3个batch:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  scene_pc: {batch['scene_pc'].shape}")
        print(f"  hand_model_pose: {batch['hand_model_pose'].shape}")
        print(f"  se3: {batch['se3'].shape}")
        print(f"  物体数量: {len(batch['obj_code'])}")


def example_exhaustive_sampling():
    """示例3: 穷尽采样模式"""
    print("\n" + "=" * 80)
    print("示例3: 穷尽采样模式")
    print("=" * 80)
    
    # 标准模式
    dataset_standard = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=1024,
        mesh_scale=0.1,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=False
    )
    
    # 穷尽模式
    dataset_exhaustive = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=1024,
        mesh_scale=0.1,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=True,
        exhaustive_sampling_strategy="sequential"
    )
    
    print(f"标准采样数据集大小: {len(dataset_standard)}")
    print(f"穷尽采样数据集大小: {len(dataset_exhaustive)}")
    print(f"扩展倍数: {len(dataset_exhaustive) / len(dataset_standard):.2f}x")


def example_sampling_strategies():
    """示例4: 不同采样策略对比"""
    print("\n" + "=" * 80)
    print("示例4: 不同采样策略对比")
    print("=" * 80)
    
    strategies = ["random", "first_n", "farthest_point", "nearest_point"]
    
    for strategy in strategies:
        dataset = ObjectCentricGraspDataset(
            succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
            obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
            num_grasps=4,
            max_points=512,
            mesh_scale=0.1,
            grasp_sampling_strategy=strategy
        )
        
        sample = dataset[0]
        hand_poses = sample['hand_model_pose']  # (4, 23)
        
        # 计算抓取位置的平均距离
        positions = hand_poses[:, :3]  # (4, 3)
        distances = torch.cdist(positions, positions)  # (4, 4)
        avg_distance = distances[torch.triu(torch.ones_like(distances), diagonal=1) > 0].mean().item()
        
        print(f"\n策略 '{strategy}':")
        print(f"  抓取位置平均距离: {avg_distance:.4f}")


def example_coordinate_verification():
    """示例5: 验证坐标系"""
    print("\n" + "=" * 80)
    print("示例5: 验证OMF坐标系")
    print("=" * 80)
    
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=1024,
        mesh_scale=0.1
    )
    
    sample = dataset[0]
    
    # 计算各种中心
    pc_center = sample['scene_pc'].mean(dim=0)
    verts_center = sample['obj_verts'].mean(dim=0)
    grasp_center = sample['hand_model_pose'][:, :3].mean(dim=0)
    
    print(f"物体代码: {sample['obj_code']}")
    print(f"\n中心位置 (OMF坐标系):")
    print(f"  点云中心: [{pc_center[0]:.4f}, {pc_center[1]:.4f}, {pc_center[2]:.4f}]")
    print(f"  顶点中心: [{verts_center[0]:.4f}, {verts_center[1]:.4f}, {verts_center[2]:.4f}]")
    print(f"  抓取中心: [{grasp_center[0]:.4f}, {grasp_center[1]:.4f}, {grasp_center[2]:.4f}]")
    
    # 验证点云确实来自物体表面
    print(f"\n点云范围:")
    print(f"  X: [{sample['scene_pc'][:, 0].min():.4f}, {sample['scene_pc'][:, 0].max():.4f}]")
    print(f"  Y: [{sample['scene_pc'][:, 1].min():.4f}, {sample['scene_pc'][:, 1].max():.4f}]")
    print(f"  Z: [{sample['scene_pc'][:, 2].min():.4f}, {sample['scene_pc'][:, 2].max():.4f}]")
    
    print(f"\n顶点范围:")
    print(f"  X: [{sample['obj_verts'][:, 0].min():.4f}, {sample['obj_verts'][:, 0].max():.4f}]")
    print(f"  Y: [{sample['obj_verts'][:, 1].min():.4f}, {sample['obj_verts'][:, 1].max():.4f}]")
    print(f"  Z: [{sample['obj_verts'][:, 2].min():.4f}, {sample['obj_verts'][:, 2].max():.4f}]")


def main():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("ObjectCentricGraspDataset 使用示例集")
    print("=" * 80 + "\n")
    
    try:
        example_basic_usage()
        example_with_dataloader()
        example_exhaustive_sampling()
        example_sampling_strategies()
        example_coordinate_verification()
        
        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"示例运行失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

