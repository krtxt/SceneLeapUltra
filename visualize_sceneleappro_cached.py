#!/usr/bin/env python3
"""
SceneLeapProDatasetCached 可视化测试脚本
验证修复后的缓存数据集类的功能，包括缓存文件创建、数据加载和可视化

主要功能：
1. 测试修复后的 SceneLeapProDatasetCached 类
2. 验证缓存文件的创建和读取
3. 可视化6D点云（xyz+rgb颜色）
4. 可视化LEAP灵巧手抓取姿态
5. 可视化目标物体mesh
6. 验证修复的参数（succ_grasp_dir, obj_root_dir, max_grasps_per_object）
7. 测试正负提示词的加载
8. 测试四种坐标系统模式的缓存数据集

支持的坐标系统模式：
- camera_centric: 相机坐标系（原始）
- object_centric: 物体模型坐标系
- camera_centric_obj_mean_normalized: 相机坐标系 + 物体中心归一化
- camera_centric_scene_mean_normalized: 相机坐标系 + 场景中心归一化

可视化组件：
- 红色点云：目标物体点云（基于object_mask）
- 灰色点云：背景点云
- 绿色mesh：目标物体mesh（来自obj_verts和obj_faces）
- 蓝色mesh：LEAP手部mesh
- RGB坐标轴：世界坐标系参考
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
import time
import logging
from typing import Optional, Tuple, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets.sceneleappro_cached import SceneLeapProDatasetCached, ForMatchSceneLeapProDatasetCached
    from datasets.sceneleappro_dataset import SceneLeapProDataset
    from utils.hand_model_origin import HandModel, HandModelType
except ImportError as e:
    logger.error(f"导入错误: {e}")
    sys.exit(1)

# 真实数据路径
ROOT_DIR = "/home/xiantuo/source/grasp/SceneLeapPro/data/723_sub_15"
SUCC_GRASP_DIR = "/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect"
OBJ_ROOT_DIR = "/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models"

def verify_paths():
    """验证数据路径是否存在"""
    paths = {
        'root_dir': ROOT_DIR,
        'succ_grasp_dir': SUCC_GRASP_DIR,
        'obj_root_dir': OBJ_ROOT_DIR
    }
    
    for name, path in paths.items():
        if os.path.exists(path):
            logger.info(f"✅ {name}: {path}")
        else:
            logger.error(f"❌ {name} 不存在: {path}")
            return False
    return True

def test_cached_dataset_creation():
    """测试缓存数据集的创建"""
    logger.info("=" * 60)
    logger.info("测试 SceneLeapProDatasetCached 创建")
    logger.info("=" * 60)
    
    try:
        # 使用小规模参数进行快速测试
        logger.info("正在创建缓存数据集（小规模测试）...")
        start_time = time.time()
        
        cached_dataset = SceneLeapProDatasetCached(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            mode="camera_centric",
            max_grasps_per_object=2,  # 限制数据量
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,  # 限制点云大小
            cache_version="v2.0_test"
        )
        
        creation_time = time.time() - start_time
        logger.info(f"✅ 缓存数据集创建成功！")
        logger.info(f"   - 数据量: {len(cached_dataset)}")
        logger.info(f"   - 创建时间: {creation_time:.2f} 秒")
        
        # 获取缓存信息
        cache_info = cached_dataset.get_cache_info()
        logger.info(f"   - 缓存路径: {cache_info['cache_path']}")
        logger.info(f"   - 缓存状态: {'已加载' if cache_info['cache_loaded'] else '未加载'}")
        
        return cached_dataset
        
    except Exception as e:
        logger.error(f"❌ 缓存数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_formatch_cached_dataset():
    """测试 ForMatch 缓存数据集"""
    logger.info("=" * 60)
    logger.info("测试 ForMatchSceneLeapProDatasetCached 创建")
    logger.info("=" * 60)
    
    try:
        logger.info("正在创建 ForMatch 缓存数据集...")
        start_time = time.time()
        
        formatch_dataset = ForMatchSceneLeapProDatasetCached(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            mode="camera_centric_scene_mean_normalized",
            max_grasps_per_object=2,
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,
            cache_version="v1.0_test",
            cache_mode="val"
        )
        
        creation_time = time.time() - start_time
        logger.info(f"✅ ForMatch 缓存数据集创建成功！")
        logger.info(f"   - 数据量: {len(formatch_dataset)}")
        logger.info(f"   - 创建时间: {creation_time:.2f} 秒")
        
        return formatch_dataset
        
    except Exception as e:
        logger.error(f"❌ ForMatch 缓存数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_sample_data(sample: Dict[str, Any], sample_idx: int):
    """分析样本数据"""
    logger.info(f"=" * 50)
    logger.info(f"样本 {sample_idx} 数据分析")
    logger.info(f"=" * 50)
    
    # 基本信息
    if 'scene_id' in sample:
        logger.info(f"场景ID: {sample['scene_id']}")
    if 'obj_code' in sample:
        logger.info(f"物体代码: {sample['obj_code']}")
    if 'positive_prompt' in sample:
        logger.info(f"正面提示词: '{sample['positive_prompt']}'")
    if 'negative_prompts' in sample:
        logger.info(f"负面提示词: {sample['negative_prompts']}")
    
    # 数据形状分析
    logger.info("数据形状:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  - {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, (list, tuple)):
            logger.info(f"  - {key}: {type(value).__name__}[{len(value)}]")
        else:
            logger.info(f"  - {key}: {type(value).__name__}")
    
    # 验证修复的功能
    logger.info("修复验证:")
    
    # 检查是否有成功抓取数据
    if 'hand_model_pose' in sample:
        hand_pose = sample['hand_model_pose']
        if isinstance(hand_pose, torch.Tensor) and hand_pose.numel() > 0:
            logger.info("  ✅ 成功加载手部姿态数据 (succ_grasp_dir 参数工作正常)")
        else:
            logger.warning("  ⚠️ 手部姿态数据为空")
    
    # 检查是否有物体网格数据
    if 'obj_verts' in sample and 'obj_faces' in sample:
        obj_verts = sample['obj_verts']
        obj_faces = sample['obj_faces']
        if isinstance(obj_verts, torch.Tensor) and obj_verts.numel() > 0:
            logger.info("  ✅ 成功加载物体网格数据 (obj_root_dir 参数工作正常)")
        else:
            logger.warning("  ⚠️ 物体网格数据为空")
    
    # 检查点云数据
    if 'scene_pc' in sample:
        scene_pc = sample['scene_pc']
        if isinstance(scene_pc, torch.Tensor) and scene_pc.shape[1] == 6:
            logger.info("  ✅ 6D点云数据格式正确 (xyz+rgb)")
        else:
            logger.warning(f"  ⚠️ 点云数据格式异常: {scene_pc.shape if hasattr(scene_pc, 'shape') else type(scene_pc)}")
    
    return sample

def create_coordinate_frame(size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """创建坐标轴参考框架"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_point_cloud_from_sample(scene_pc: torch.Tensor) -> o3d.geometry.PointCloud:
    """从样本数据创建Open3D点云"""
    if isinstance(scene_pc, torch.Tensor):
        scene_pc_np = scene_pc.detach().cpu().numpy()
    else:
        scene_pc_np = scene_pc
    
    # 确保形状正确
    if len(scene_pc_np.shape) != 2 or scene_pc_np.shape[1] != 6:
        logger.warning(f"点云数据形状异常: {scene_pc_np.shape}")
        return o3d.geometry.PointCloud()
    
    # 提取xyz坐标和rgb颜色
    xyz = scene_pc_np[:, :3]
    rgb = scene_pc_np[:, 3:6]
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def create_object_mesh(obj_verts: torch.Tensor, obj_faces: torch.Tensor,
                      color: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> Optional[o3d.geometry.TriangleMesh]:
    """从顶点和面数据创建目标物体mesh"""
    try:
        if obj_verts.numel() == 0 or obj_faces.numel() == 0:
            return None
        
        vertices_np = obj_verts.detach().cpu().numpy()
        faces_np = obj_faces.detach().cpu().numpy()
        
        if vertices_np.shape[1] != 3 or faces_np.shape[1] != 3:
            return None
        
        object_mesh = o3d.geometry.TriangleMesh()
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        object_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
        object_mesh.paint_uniform_color(color)
        object_mesh.compute_vertex_normals()
        
        return object_mesh
    except Exception as e:
        logger.error(f"创建物体mesh失败: {e}")
        return None

def create_hand_mesh(hand_pose: torch.Tensor, hand_model: HandModel) -> Optional[o3d.geometry.TriangleMesh]:
    """从手部姿态创建手部mesh"""
    try:
        if hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)

        # 根据SceneLeapProDataset的格式 [P, Joints, Q_wxyz]，重新排列为 [P, Q_wxyz, Joints]
        P = hand_pose[:, :3]  # 平移
        Joints = hand_pose[:, 3:19]  # 关节角度
        Q_wxyz = hand_pose[:, 19:23]  # 四元数

        # 重新排列为 [P, Q_wxyz, Joints]
        reordered_pose = torch.cat([P, Q_wxyz, Joints], dim=1)
        hand_model.set_parameters_quat(reordered_pose)

        # 获取trimesh数据
        trimesh_data = hand_model.get_trimesh_data(0)

        # 转换为Open3D mesh
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(trimesh_data.vertices)
        hand_mesh.triangles = o3d.utility.Vector3iVector(trimesh_data.faces)
        hand_mesh.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
        hand_mesh.compute_vertex_normals()

        return hand_mesh
    except Exception as e:
        logger.error(f"创建手部mesh失败: {e}")
        return None

def visualize_cached_sample(dataset, sample_idx: int = 0):
    """可视化缓存数据集样本"""
    logger.info(f"=" * 60)
    logger.info(f"可视化缓存数据集样本 {sample_idx}")
    logger.info(f"=" * 60)

    try:
        # 获取样本数据
        start_time = time.time()
        sample = dataset[sample_idx]
        load_time = time.time() - start_time
        logger.info(f"样本加载时间: {load_time:.4f} 秒")

        # 分析样本数据
        sample = analyze_sample_data(sample, sample_idx)

        # 检查是否有错误
        if 'error' in sample:
            logger.error(f"样本包含错误: {sample['error']}")
            return

        # 创建可视化对象列表
        vis_objects = []

        # 添加坐标轴
        coordinate_frame = create_coordinate_frame(size=0.1)
        vis_objects.append(coordinate_frame)

        # 创建点云
        if 'scene_pc' in sample:
            scene_pc = sample['scene_pc']
            pcd = create_point_cloud_from_sample(scene_pc)
            if len(pcd.points) > 0:
                vis_objects.append(pcd)
                logger.info(f"✅ 点云创建成功: {len(pcd.points)} 个点")

        # 创建目标物体mesh
        if 'obj_verts' in sample and 'obj_faces' in sample:
            obj_verts = sample['obj_verts']
            obj_faces = sample['obj_faces']
            object_mesh = create_object_mesh(obj_verts, obj_faces, color=(0.0, 0.8, 0.0))
            if object_mesh is not None:
                vis_objects.append(object_mesh)
                logger.info(f"✅ 物体mesh创建成功")

        # 创建手部模型
        if 'hand_model_pose' in sample:
            try:
                hand_model = HandModel(hand_model_type=HandModelType.LEAP, device='cpu')
                hand_pose = sample['hand_model_pose']
                hand_mesh = create_hand_mesh(hand_pose, hand_model)
                if hand_mesh is not None:
                    vis_objects.append(hand_mesh)
                    logger.info(f"✅ 手部mesh创建成功")
            except Exception as e:
                logger.warning(f"手部模型创建失败: {e}")

        # 显示可视化
        window_name = f"SceneLeapProDatasetCached - Sample {sample_idx}"
        if 'obj_code' in sample:
            window_name += f" - {sample['obj_code']}"

        logger.info(f"启动可视化窗口...")
        logger.info("可视化说明:")
        logger.info("  - RGB坐标轴: 世界坐标系")
        logger.info("  - 彩色点云: 场景点云 (xyz+rgb)")
        logger.info("  - 绿色mesh: 目标物体")
        logger.info("  - 蓝色mesh: 手部模型")

        o3d.visualization.draw_geometries(
            vis_objects,
            window_name=window_name,
            width=1200,
            height=800,
            left=50,
            top=50
        )

    except Exception as e:
        logger.error(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()

def test_different_modes():
    """测试不同的坐标系统模式"""
    logger.info("=" * 60)
    logger.info("测试不同坐标系统模式的缓存数据集")
    logger.info("=" * 60)

    # 支持的四种模式
    modes = ["camera_centric", "object_centric", "camera_centric_obj_mean_normalized", "camera_centric_scene_mean_normalized"]

    datasets = {}

    for mode in modes:
        try:
            logger.info(f"创建 {mode} 模式的缓存数据集...")
            start_time = time.time()

            dataset = SceneLeapProDatasetCached(
                root_dir=ROOT_DIR,
                succ_grasp_dir=SUCC_GRASP_DIR,
                obj_root_dir=OBJ_ROOT_DIR,
                mode=mode,
                max_grasps_per_object=2,
                mesh_scale=0.1,
                num_neg_prompts=4,
                enable_cropping=True,
                max_points=5000,  # 减少点云大小以加快测试
                cache_version=f"v2.0_test_{mode}"
            )

            creation_time = time.time() - start_time
            datasets[mode] = dataset

            logger.info(f"✅ {mode} 模式创建成功")
            logger.info(f"   - 数据量: {len(dataset)}")
            logger.info(f"   - 创建时间: {creation_time:.2f} 秒")

            # 分析第一个样本的点云统计信息
            if len(dataset) > 0:
                sample = dataset[0]
                if 'scene_pc' in sample:
                    scene_pc = sample['scene_pc']
                    if isinstance(scene_pc, torch.Tensor) and scene_pc.numel() > 0:
                        xyz = scene_pc[:, :3]
                        mean_xyz = torch.mean(xyz, dim=0)
                        std_xyz = torch.std(xyz, dim=0)
                        min_xyz = torch.min(xyz, dim=0)[0]
                        max_xyz = torch.max(xyz, dim=0)[0]

                        logger.info(f"   - 点云统计 (xyz):")
                        logger.info(f"     均值: [{mean_xyz[0]:.3f}, {mean_xyz[1]:.3f}, {mean_xyz[2]:.3f}]")
                        logger.info(f"     标准差: [{std_xyz[0]:.3f}, {std_xyz[1]:.3f}, {std_xyz[2]:.3f}]")
                        logger.info(f"     范围: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}]")

        except Exception as e:
            logger.error(f"❌ {mode} 模式创建失败: {e}")
            continue

    # 提供交互式选择可视化
    if datasets:
        logger.info(f"\n成功创建了 {len(datasets)} 个模式的数据集")
        logger.info("可用模式:")
        for i, mode in enumerate(datasets.keys(), 1):
            logger.info(f"  {i}. {mode}")

        try:
            choice = input("\n请选择要可视化的模式 (输入数字) 或按回车跳过: ").strip()
            if choice.isdigit():
                mode_idx = int(choice) - 1
                mode_list = list(datasets.keys())
                if 0 <= mode_idx < len(mode_list):
                    selected_mode = mode_list[mode_idx]
                    logger.info(f"可视化 {selected_mode} 模式...")
                    visualize_cached_sample(datasets[selected_mode], 0)
        except Exception as e:
            logger.warning(f"可视化选择失败: {e}")

    return datasets

def test_data_consistency():
    """测试数据一致性"""
    logger.info("=" * 60)
    logger.info("测试原始数据集与缓存数据集的一致性")
    logger.info("=" * 60)

    try:
        # 创建原始数据集（小规模）
        logger.info("创建原始数据集...")
        original_dataset = SceneLeapProDataset(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            mode="camera_centric",
            max_grasps_per_object=2,
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000  # 限制点云大小
        )

        # 创建缓存数据集
        logger.info("创建缓存数据集...")
        cached_dataset = SceneLeapProDatasetCached(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            mode="camera_centric_scene_mean_normalized",
            max_grasps_per_object=2,  # 限制数据量
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,  # 限制点云大小
            cache_version="v2.0_test"
        )

        # 比较数据量
        logger.info(f"原始数据集大小: {len(original_dataset)}")
        logger.info(f"缓存数据集大小: {len(cached_dataset)}")

        if len(original_dataset) == len(cached_dataset):
            logger.info("✅ 数据量一致")
        else:
            logger.warning("⚠️ 数据量不一致")

        # 比较第一个样本的关键字段
        if len(cached_dataset) > 0:
            logger.info("比较第一个样本...")
            original_sample = original_dataset[0]
            cached_sample = cached_dataset[0]

            # 比较关键字段
            key_fields = ['scene_pc', 'hand_model_pose', 'positive_prompt']
            for field in key_fields:
                if field in original_sample and field in cached_sample:
                    if isinstance(original_sample[field], torch.Tensor):
                        if torch.allclose(original_sample[field], cached_sample[field], atol=1e-6):
                            logger.info(f"  ✅ {field} 数据一致")
                        else:
                            logger.warning(f"  ⚠️ {field} 数据不一致")
                    else:
                        if original_sample[field] == cached_sample[field]:
                            logger.info(f"  ✅ {field} 数据一致")
                        else:
                            logger.warning(f"  ⚠️ {field} 数据不一致")

        return True

    except Exception as e:
        logger.error(f"一致性测试失败: {e}")
        return False

def interactive_visualization(dataset):
    """交互式可视化，允许浏览不同样本"""
    logger.info("=" * 60)
    logger.info("交互式可视化模式")
    logger.info("=" * 60)

    if len(dataset) == 0:
        logger.error("数据集为空，无法进行可视化")
        return

    logger.info(f"数据集包含 {len(dataset)} 个样本")
    logger.info("输入样本索引进行可视化，输入 'q' 退出")

    while True:
        try:
            user_input = input(f"\n请输入样本索引 (0-{len(dataset)-1}) 或 'q' 退出: ").strip()

            if user_input.lower() == 'q':
                logger.info("退出交互式可视化")
                break

            sample_idx = int(user_input)
            if 0 <= sample_idx < len(dataset):
                visualize_cached_sample(dataset, sample_idx)
            else:
                logger.warning(f"索引超出范围，请输入 0-{len(dataset)-1} 之间的数字")

        except ValueError:
            logger.warning("请输入有效的数字或 'q'")
        except KeyboardInterrupt:
            logger.info("\n用户中断，退出交互式可视化")
            break
        except Exception as e:
            logger.error(f"可视化过程中出错: {e}")

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("SceneLeapProDatasetCached 可视化测试脚本")
    logger.info("验证修复后的缓存数据集类功能")
    logger.info("=" * 80)

    # 验证路径
    if not verify_paths():
        logger.error("数据路径验证失败，请检查路径设置")
        return

    # 测试选项
    tests = [
        ("1", "创建和测试 SceneLeapProDatasetCached", test_cached_dataset_creation),
        ("2", "创建和测试 ForMatchSceneLeapProDatasetCached", test_formatch_cached_dataset),
        ("3", "测试不同坐标系统模式", test_different_modes),
        ("4", "数据一致性测试", test_data_consistency),
        ("5", "交互式可视化", None),  # 特殊处理
        ("6", "运行所有测试", None),  # 特殊处理
    ]

    logger.info("可用测试选项:")
    for option, description, _ in tests:
        logger.info(f"  {option}. {description}")

    while True:
        try:
            choice = input("\n请选择测试选项 (1-6) 或 'q' 退出: ").strip()

            if choice.lower() == 'q':
                logger.info("退出测试")
                break

            if choice == "1":
                dataset = test_cached_dataset_creation()
                if dataset is not None:
                    # 可视化第一个样本
                    if len(dataset) > 0:
                        visualize_cached_sample(dataset, 0)
                    else:
                        logger.warning("数据集为空")

            elif choice == "2":
                dataset = test_formatch_cached_dataset()
                if dataset is not None:
                    # 可视化第一个样本
                    if len(dataset) > 0:
                        visualize_cached_sample(dataset, 0)
                    else:
                        logger.warning("数据集为空")

            elif choice == "3":
                test_different_modes()

            elif choice == "4":
                test_data_consistency()

            elif choice == "5":
                # 交互式可视化
                logger.info("选择要用于交互式可视化的数据集:")
                logger.info("  1. SceneLeapProDatasetCached")
                logger.info("  2. ForMatchSceneLeapProDatasetCached")
                logger.info("  3. 不同模式测试")

                dataset_choice = input("请选择数据集 (1-3): ").strip()

                if dataset_choice == "1":
                    dataset = test_cached_dataset_creation()
                    if dataset is not None:
                        interactive_visualization(dataset)
                elif dataset_choice == "2":
                    dataset = test_formatch_cached_dataset()
                    if dataset is not None:
                        interactive_visualization(dataset)
                elif dataset_choice == "3":
                    datasets = test_different_modes()
                    if datasets:
                        # 让用户选择一个模式进行交互式可视化
                        mode_list = list(datasets.keys())
                        logger.info("可用模式:")
                        for i, mode in enumerate(mode_list, 1):
                            logger.info(f"  {i}. {mode}")

                        mode_choice = input(f"请选择模式 (1-{len(mode_list)}): ").strip()
                        if mode_choice.isdigit():
                            mode_idx = int(mode_choice) - 1
                            if 0 <= mode_idx < len(mode_list):
                                selected_mode = mode_list[mode_idx]
                                interactive_visualization(datasets[selected_mode])
                else:
                    logger.warning("无效选择")

            elif choice == "6":
                # 运行所有测试
                logger.info("运行所有测试...")

                # 测试1: SceneLeapProDatasetCached
                dataset1 = test_cached_dataset_creation()

                # 测试2: ForMatchSceneLeapProDatasetCached
                dataset2 = test_formatch_cached_dataset()

                # 测试3: 不同模式测试
                test_different_modes()

                # 测试4: 数据一致性
                test_data_consistency()

                # 可视化第一个可用的数据集
                if dataset1 is not None and len(dataset1) > 0:
                    logger.info("可视化 SceneLeapProDatasetCached 第一个样本...")
                    visualize_cached_sample(dataset1, 0)
                elif dataset2 is not None and len(dataset2) > 0:
                    logger.info("可视化 ForMatchSceneLeapProDatasetCached 第一个样本...")
                    visualize_cached_sample(dataset2, 0)

                logger.info("所有测试完成！")

            else:
                logger.warning("无效选择，请输入 1-6 或 'q'")

        except KeyboardInterrupt:
            logger.info("\n用户中断，退出测试")
            break
        except Exception as e:
            logger.error(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
