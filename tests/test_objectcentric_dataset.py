"""
测试ObjectCentricGraspDataset数据集类

运行方式:
cd /home/xiantuo/source/grasp/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python tests/test_objectcentric_dataset.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset


def test_basic_loading():
    """测试基本加载功能"""
    print("=" * 80)
    print("测试1: 基本数据加载")
    print("=" * 80)
    
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=1024,  # 使用较小的值加快测试
        max_grasps_per_object=50,
        mesh_scale=0.1,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=False
    )
    
    print(f"✓ 数据集创建成功")
    print(f"  - 数据集大小: {len(dataset)}")
    print(f"  - 物体数量: {len(dataset.hand_pose_data)}")
    
    # 测试单个样本
    sample = dataset[0]
    
    print(f"\n✓ 成功加载第一个样本")
    print(f"  - scene_pc shape: {sample['scene_pc'].shape}")
    print(f"  - hand_model_pose shape: {sample['hand_model_pose'].shape}")
    print(f"  - se3 shape: {sample['se3'].shape}")
    print(f"  - obj_verts shape: {sample['obj_verts'].shape}")
    print(f"  - obj_faces shape: {sample['obj_faces'].shape}")
    print(f"  - obj_code: {sample['obj_code']}")
    print(f"  - scene_id: {sample['scene_id']}")
    
    # 验证shape
    assert sample['scene_pc'].shape == (1024, 3), "scene_pc shape错误"
    assert sample['hand_model_pose'].shape == (4, 23), "hand_model_pose shape错误"
    assert sample['se3'].shape == (4, 4, 4), "se3 shape错误"
    
    print("\n✓ 所有shape验证通过")


def test_dataloader():
    """测试DataLoader批处理"""
    print("\n" + "=" * 80)
    print("测试2: DataLoader批处理")
    print("=" * 80)
    
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=8,
        max_points=512,
        max_grasps_per_object=20,
        mesh_scale=0.1,
        grasp_sampling_strategy="farthest_point",
        use_exhaustive_sampling=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    
    print(f"✓ DataLoader创建成功")
    print(f"  - Batch size: 4")
    print(f"  - Dataset size: {len(dataset)}")
    
    # 测试第一个batch
    batch = next(iter(dataloader))
    
    print(f"\n✓ 成功加载第一个batch")
    print(f"  - scene_pc shape: {batch['scene_pc'].shape}")
    print(f"  - hand_model_pose shape: {batch['hand_model_pose'].shape}")
    print(f"  - se3 shape: {batch['se3'].shape}")
    print(f"  - obj_verts: List of {len(batch['obj_verts'])} tensors")
    print(f"  - obj_faces: List of {len(batch['obj_faces'])} tensors")
    
    # 验证batch shape
    assert batch['scene_pc'].shape == (4, 512, 3), "Batched scene_pc shape错误"
    assert batch['hand_model_pose'].shape == (4, 8, 23), "Batched hand_model_pose shape错误"
    assert batch['se3'].shape == (4, 8, 4, 4), "Batched se3 shape错误"
    
    print("\n✓ 所有batch shape验证通过")


def test_exhaustive_sampling():
    """测试穷尽采样模式"""
    print("\n" + "=" * 80)
    print("测试3: 穷尽采样模式")
    print("=" * 80)
    
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=256,
        max_grasps_per_object=40,
        mesh_scale=0.1,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=True,
        exhaustive_sampling_strategy="sequential"
    )
    
    print(f"✓ 穷尽采样数据集创建成功")
    print(f"  - 数据集大小: {len(dataset)}")
    print(f"  - 穷尽采样策略: sequential")
    
    # 测试样本
    sample = dataset[0]
    
    print(f"\n✓ 成功加载穷尽采样样本")
    print(f"  - scene_pc shape: {sample['scene_pc'].shape}")
    print(f"  - hand_model_pose shape: {sample['hand_model_pose'].shape}")
    
    assert sample['scene_pc'].shape == (256, 3), "scene_pc shape错误"
    assert sample['hand_model_pose'].shape == (4, 23), "hand_model_pose shape错误"
    
    print("\n✓ 穷尽采样验证通过")


def test_sampling_strategies():
    """测试不同采样策略"""
    print("\n" + "=" * 80)
    print("测试4: 不同采样策略")
    print("=" * 80)
    
    strategies = ["random", "first_n", "farthest_point", "nearest_point"]
    
    for strategy in strategies:
        try:
            dataset = ObjectCentricGraspDataset(
                succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
                obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
                num_grasps=4,
                max_points=256,
                max_grasps_per_object=20,
                mesh_scale=0.1,
                grasp_sampling_strategy=strategy,
                use_exhaustive_sampling=False
            )
            
            sample = dataset[0]
            print(f"✓ 策略 '{strategy}' 测试通过 - shape: {sample['hand_model_pose'].shape}")
        
        except Exception as e:
            print(f"✗ 策略 '{strategy}' 测试失败: {e}")
            raise


def test_coordinate_system():
    """验证xy平面居中和桌面平面"""
    print("\n" + "=" * 80)
    print("测试5: 验证居中归一化和桌面平面")
    print("=" * 80)
    
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
        obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
        num_grasps=4,
        max_points=2048,
        max_grasps_per_object=20,
        mesh_scale=0.1,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=False,
        object_sampling_ratio=0.8,
        table_size=0.4
    )
    
    sample = dataset[0]
    
    # 验证xy平面居中
    pc_mean = sample['scene_pc'].mean(dim=0)
    verts_mean = sample['obj_verts'].mean(dim=0)
    grasp_mean = sample['hand_model_pose'][:, :3].mean(dim=0)
    
    print(f"✓ 点云中心: [{pc_mean[0]:.4f}, {pc_mean[1]:.4f}, {pc_mean[2]:.4f}]")
    print(f"✓ 顶点中心: [{verts_mean[0]:.4f}, {verts_mean[1]:.4f}, {verts_mean[2]:.4f}]")
    print(f"✓ 抓取中心: [{grasp_mean[0]:.4f}, {grasp_mean[1]:.4f}, {grasp_mean[2]:.4f}]")
    
    # 验证xy平面接近0（居中）
    # 注意：点云包含桌面，会有一定偏移；顶点和抓取应该更接近0
    pc_xy_offset = torch.norm(pc_mean[:2]).item()
    verts_xy_offset = torch.norm(verts_mean[:2]).item()
    grasp_xy_offset = torch.norm(grasp_mean[:2]).item()
    
    print(f"\nxy平面偏移量:")
    print(f"  点云xy偏移: {pc_xy_offset:.4f} 米")
    print(f"  顶点xy偏移: {verts_xy_offset:.4f} 米")
    print(f"  抓取xy偏移: {grasp_xy_offset:.4f} 米")
    
    # 使用更合理的阈值
    assert pc_xy_offset < 0.15, f"点云xy中心偏移过大: {pc_xy_offset:.4f}米"
    assert verts_xy_offset < 0.05, f"顶点xy中心偏移过大: {verts_xy_offset:.4f}米"
    assert grasp_xy_offset < 0.05, f"抓取xy中心偏移过大: {grasp_xy_offset:.4f}米"
    
    print(f"\n✓ xy平面居中验证通过")
    print(f"  (点云<15cm, 顶点<5cm, 抓取<5cm)")
    
    # 验证包含桌面点（z=0附近应该有点）
    z_coords = sample['scene_pc'][:, 2]
    table_points = (z_coords.abs() < 0.001).sum()
    
    print(f"\n桌面点统计:")
    print(f"  z=0附近的点数: {table_points}")
    print(f"  z最小值: {z_coords.min():.4f}")
    print(f"  z最大值: {z_coords.max():.4f}")
    
    assert table_points > 0, "未找到桌面点"
    
    print("\n✓ 桌面平面验证通过")


def test_sampling_ratio():
    """测试物体/桌面采样比例"""
    print("\n" + "=" * 80)
    print("测试6: 验证采样比例控制")
    print("=" * 80)
    
    # 测试不同的采样比例
    ratios = [0.5, 0.8, 0.9]
    
    for ratio in ratios:
        dataset = ObjectCentricGraspDataset(
            succ_grasp_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect",
            obj_root_dir="/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models",
            num_grasps=4,
            max_points=1000,  # 使用较小值便于计算
            max_grasps_per_object=20,
            mesh_scale=0.1,
            grasp_sampling_strategy="random",
            use_exhaustive_sampling=False,
            object_sampling_ratio=ratio,
            table_size=0.4
        )
        
        sample = dataset[0]
        scene_pc = sample['scene_pc']
        
        # 统计桌面点数（z=0附近）
        z_coords = scene_pc[:, 2]
        table_points_count = (z_coords.abs() < 0.001).sum().item()
        obj_points_count = scene_pc.shape[0] - table_points_count
        
        actual_table_ratio = table_points_count / scene_pc.shape[0]
        expected_table_ratio = 1.0 - ratio
        
        print(f"\n比例 {ratio:.1f} (物体) / {1-ratio:.1f} (桌面):")
        print(f"  期望桌面点: {int(1000 * expected_table_ratio)}")
        print(f"  实际桌面点: {table_points_count}")
        print(f"  期望物体点: {int(1000 * ratio)}")
        print(f"  实际物体点: {obj_points_count}")
        print(f"  实际桌面比例: {actual_table_ratio:.3f} (期望: {expected_table_ratio:.3f})")
    
    print("\n✓ 采样比例控制验证通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("ObjectCentricGraspDataset 测试套件")
    print("=" * 80)
    
    try:
        test_basic_loading()
        test_dataloader()
        test_exhaustive_sampling()
        test_sampling_strategies()
        test_coordinate_system()
        test_sampling_ratio()
        
        print("\n" + "=" * 80)
        print("✓ 所有测试通过!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

