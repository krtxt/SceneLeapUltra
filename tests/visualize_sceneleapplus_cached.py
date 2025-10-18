#!/usr/bin/env python3
"""
SceneLeapPlusDatasetCached 可视化测试脚本
验证修复后的缓存数据集类的功能，包括缓存文件创建、数据加载和可视化

主要功能：
1. 测试修复后的 SceneLeapPlusDatasetCached 类
2. 验证缓存文件的创建和读取
3. 可视化6D点云（xyz+rgb颜色）
4. 可视化多个LEAP灵巧手抓取姿态
5. 可视化目标物体mesh
6. 验证修复的参数（succ_grasp_dir, obj_root_dir, max_grasps_per_object）
7. 测试正负提示词的加载
8. 测试四种坐标系统模式的缓存数据集
9. 支持多抓取并行学习架构的可视化

支持的坐标系统模式：
- camera_centric: 相机坐标系（原始）
- object_centric: 物体模型坐标系
- camera_centric_obj_mean_normalized: 相机坐标系 + 物体中心归一化
- camera_centric_scene_mean_normalized: 相机坐标系 + 场景中心归一化

可视化组件：
- 红色点云：目标物体点云（基于object_mask）
- 灰色点云：背景点云
- 绿色mesh：目标物体mesh（来自obj_verts和obj_faces）
- 彩色mesh：多个LEAP手部mesh（不同颜色区分不同抓取）
- RGB坐标轴：世界坐标系参考

SceneLeapPlusDatasetCached特性：
- 返回固定数量的抓取姿态 (num_grasps, 23)
- 支持多抓取并行学习架构
- HDF5缓存存储，提高数据加载效率
- 分布式训练支持
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
import time
import logging
from typing import Optional, Tuple, Dict, Any, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from datasets.sceneleapplus_cached import SceneLeapPlusDatasetCached
    from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
    from utils.hand_model import HandModel
    from utils.hand_types import HandModelType
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
    logger.info("测试 SceneLeapPlusDatasetCached 创建")
    logger.info("=" * 60)
    
    try:
        # 使用小规模参数进行快速测试
        logger.info("正在创建缓存数据集（小规模测试）...")
        start_time = time.time()
        
        cached_dataset = SceneLeapPlusDatasetCached(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            num_grasps=4,  # SceneLeapPlus特有参数
            mode="camera_centric",
            max_grasps_per_object=2,  # 限制数据量
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,  # 限制点云大小
            grasp_sampling_strategy="random",  # SceneLeapPlus特有参数
            cache_version="v2.0_plus_test",  # 更新缓存版本
            # 穷尽采样参数 - 可以在这里测试穷尽采样
            use_exhaustive_sampling=False,  # 设为True可测试穷尽采样
            exhaustive_sampling_strategy="sequential"  # 可选策略
        )
        
        creation_time = time.time() - start_time
        logger.info(f"✅ 缓存数据集创建成功！")
        logger.info(f"   - 数据量: {len(cached_dataset)}")
        logger.info(f"   - 创建时间: {creation_time:.2f} 秒")
        logger.info(f"   - 每个样本抓取数量: {cached_dataset.num_grasps}")
        
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
    
    # 验证SceneLeapPlus特有功能
    logger.info("SceneLeapPlus特性验证:")
    
    # 检查多抓取数据
    if 'hand_model_pose' in sample:
        hand_pose = sample['hand_model_pose']
        print("hand_pose : " , hand_pose)
        if isinstance(hand_pose, torch.Tensor):
            if hand_pose.dim() == 2:
                num_grasps, pose_dim = hand_pose.shape
                logger.info(f"  ✅ 多抓取数据格式正确: ({num_grasps}, {pose_dim})")
                logger.info(f"     - 抓取数量: {num_grasps}")
                logger.info(f"     - 姿态维度: {pose_dim}")
            else:
                logger.warning(f"  ⚠️ 手部姿态数据维度异常: {hand_pose.shape}")
        else:
            logger.warning("  ⚠️ 手部姿态数据类型异常")
    
    # 检查SE3矩阵
    if 'se3' in sample:
        se3_matrices = sample['se3']
        if isinstance(se3_matrices, torch.Tensor):
            if se3_matrices.dim() == 3 and se3_matrices.shape[1:] == (4, 4):
                num_se3 = se3_matrices.shape[0]
                logger.info(f"  ✅ SE3矩阵格式正确: ({num_se3}, 4, 4)")
            else:
                logger.warning(f"  ⚠️ SE3矩阵格式异常: {se3_matrices.shape}")
    
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

def create_hand_meshes(hand_poses: torch.Tensor, hand_model: HandModel,
                      grasp_indices: Optional[List[int]] = None) -> List[o3d.geometry.TriangleMesh]:
    """
    从多个手部姿态创建手部mesh列表
    使用新的 utils/hand_model.py，支持批量输入格式 (B, num_grasps, pose_dim)

    Args:
        hand_poses: torch.Tensor of shape (num_grasps, 23) - 多个手部姿态参数
        hand_model: HandModel - 手部模型实例
        grasp_indices: Optional[List[int]] - 要可视化的抓取索引，None表示显示所有抓取

    Returns:
        List[o3d.geometry.TriangleMesh]: 手部mesh列表
    """
    hand_meshes = []

    if hand_poses.dim() == 1:
        hand_poses = hand_poses.unsqueeze(0)

    num_grasps = hand_poses.shape[0]

    # 确定要显示的抓取索引
    indices_to_show = []
    if grasp_indices is not None:
        indices_to_show = [i for i in grasp_indices if 0 <= i < num_grasps]
    else:
        # 默认显示所有抓取
        indices_to_show = list(range(num_grasps))

    logger.info(f"准备可视化抓取索引: {indices_to_show}")
    
    if not indices_to_show:
        return []

    selected_poses = hand_poses[indices_to_show]

    # 关键步骤：过滤掉无效的（零填充的）抓取姿态
    valid_mask = selected_poses.abs().sum(dim=-1) > 1e-6
    valid_poses = selected_poses[valid_mask]
    
    # 获取通过过滤的原始索引，用于日志记录
    original_indices_of_valid_poses = [indices_to_show[i] for i, is_valid in enumerate(valid_mask) if is_valid]

    if valid_poses.shape[0] < len(indices_to_show):
        logger.info(f"过滤掉 {len(indices_to_show) - valid_poses.shape[0]} 个无效的 (零填充) 抓取。")

    if valid_poses.shape[0] == 0:
        logger.warning("没有有效的抓取姿态可供可视化。")
        return []

    logger.info(f"正在为 {len(valid_poses)} 个有效抓取创建mesh...")

    try:
        # 为每个有效的抓取单独设置参数并创建mesh
        for i, single_pose in enumerate(valid_poses):
            original_grasp_idx = original_indices_of_valid_poses[i]
            try:
                hand_model.set_parameters(single_pose)
                trimesh_data = hand_model.get_trimesh_data(0)

                hand_mesh = o3d.geometry.TriangleMesh()
                hand_mesh.vertices = o3d.utility.Vector3dVector(trimesh_data.vertices)
                hand_mesh.triangles = o3d.utility.Vector3iVector(trimesh_data.faces)

                colors = [
                    [0.0, 0.0, 1.0],  # 蓝色
                    [1.0, 0.0, 1.0],  # 紫色
                    [0.0, 1.0, 1.0],  # 青色
                    [1.0, 0.5, 0.0],  # 橙色
                    [0.5, 0.0, 1.0],  # 紫蓝色
                    [1.0, 1.0, 0.0],  # 黄色
                    [0.0, 0.5, 1.0],  # 浅蓝色
                    [1.0, 0.0, 0.5],  # 粉红色
                ]
                color = colors[i % len(colors)]
                hand_mesh.paint_uniform_color(color)
                hand_mesh.compute_vertex_normals()

                hand_meshes.append(hand_mesh)
                logger.info(f"✅ 抓取 {original_grasp_idx} 的手部mesh创建成功 (颜色: {color})")

            except Exception as e:
                logger.error(f"创建抓取 {original_grasp_idx} 的手部mesh失败: {e}")
                continue

    except Exception as e:
        logger.error(f"设置手部参数或创建mesh时出错: {e}")
        return []

    return hand_meshes

def visualize_cached_sample(dataset, sample_idx: int = 0, grasp_indices: Optional[List[int]] = None):
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

        # 创建多个手部模型
        if 'hand_model_pose' in sample:
            try:
                hand_model = HandModel(hand_model_type=HandModelType.LEAP, device='cpu')
                hand_poses = sample['hand_model_pose']

                # 确定要显示的抓取数量
                if hand_poses.dim() == 2:
                    num_grasps = hand_poses.shape[0]
                    logger.info(f"样本包含 {num_grasps} 个抓取姿态")

                    # 确定要显示的抓取索引
                    if grasp_indices is None:
                        grasp_indices = list(range(num_grasps))
                        logger.info(f"显示所有 {num_grasps} 个抓取")
                    else:
                        logger.info(f"显示指定的抓取: {grasp_indices}")

                    hand_meshes = create_hand_meshes(hand_poses, hand_model, grasp_indices)
                else:
                    # 单个抓取的情况
                    logger.info("样本包含单个抓取姿态")
                    hand_meshes = create_hand_meshes(hand_poses, hand_model, [0])

                if hand_meshes:
                    vis_objects.extend(hand_meshes)
                    logger.info(f"✅ {len(hand_meshes)} 个手部mesh创建成功")
                else:
                    logger.warning("✗ 手部mesh创建失败")

            except Exception as e:
                logger.warning(f"手部模型创建失败: {e}")

        # 显示可视化
        window_name = f"SceneLeapPlusDatasetCached - Sample {sample_idx}"
        if 'obj_code' in sample:
            window_name += f" - {sample['obj_code']}"
        if 'hand_model_pose' in sample and sample['hand_model_pose'].dim() == 2:
            num_grasps = sample['hand_model_pose'].shape[0]
            num_shown = len(grasp_indices) if grasp_indices else num_grasps
            window_name += f" ({num_shown}/{num_grasps} grasps)"

        logger.info(f"启动可视化窗口...")
        logger.info("可视化说明:")
        logger.info("  - RGB坐标轴: 世界坐标系")
        logger.info("  - 彩色点云: 场景点云 (xyz+rgb)")
        logger.info("  - 绿色mesh: 目标物体")
        logger.info("  - 彩色mesh: 多个手部模型")
        logger.info("    * 蓝色: 抓取1, 紫色: 抓取2, 青色: 抓取3, 橙色: 抓取4...")

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

            dataset = SceneLeapPlusDatasetCached(
                root_dir=ROOT_DIR,
                succ_grasp_dir=SUCC_GRASP_DIR,
                obj_root_dir=OBJ_ROOT_DIR,
                num_grasps=4,  # SceneLeapPlus特有参数
                mode=mode,
                max_grasps_per_object=2,
                mesh_scale=0.1,
                num_neg_prompts=4,
                enable_cropping=True,
                max_points=5000,  # 减少点云大小以加快测试
                grasp_sampling_strategy="random",
                cache_version=f"v2.0_plus_test_{mode}",  # 更新缓存版本
                # 穷尽采样参数
                use_exhaustive_sampling=False,  # 可以设为True测试穷尽采样
                exhaustive_sampling_strategy="sequential"
            )

            creation_time = time.time() - start_time
            datasets[mode] = dataset

            logger.info(f"✅ {mode} 模式创建成功")
            logger.info(f"   - 数据量: {len(dataset)}")
            logger.info(f"   - 创建时间: {creation_time:.2f} 秒")
            logger.info(f"   - 每个样本抓取数量: {dataset.num_grasps}")

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
        original_dataset = SceneLeapPlusDataset(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            num_grasps=4,
            mode="camera_centric",
            max_grasps_per_object=2,
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,
            grasp_sampling_strategy="random",
            # 穷尽采样参数
            use_exhaustive_sampling=False,
            exhaustive_sampling_strategy="sequential"
        )

        # 创建缓存数据集
        logger.info("创建缓存数据集...")
        cached_dataset = SceneLeapPlusDatasetCached(
            root_dir=ROOT_DIR,
            succ_grasp_dir=SUCC_GRASP_DIR,
            obj_root_dir=OBJ_ROOT_DIR,
            num_grasps=4,
            mode="camera_centric_scene_mean_normalized",
            max_grasps_per_object=2,  # 限制数据量
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=10000,  # 限制点云大小
            grasp_sampling_strategy="random",
            cache_version="v2.0_plus_consistency_test",  # 更新缓存版本
            # 穷尽采样参数
            use_exhaustive_sampling=False,
            exhaustive_sampling_strategy="sequential"
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
                        # 对于多抓取数据，比较形状
                        if original_sample[field].shape == cached_sample[field].shape:
                            logger.info(f"  ✅ {field} 形状一致: {original_sample[field].shape}")
                        else:
                            logger.warning(f"  ⚠️ {field} 形状不一致: {original_sample[field].shape} vs {cached_sample[field].shape}")
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
    logger.info("可以指定要显示的抓取索引，例如: '0 [0,1,2]' 显示样本0的前3个抓取")

    while True:
        try:
            user_input = input(f"\n请输入样本索引 (0-{len(dataset)-1}) [可选:抓取索引列表] 或 'q' 退出: ").strip()

            if user_input.lower() == 'q':
                logger.info("退出交互式可视化")
                break

            # 解析输入
            parts = user_input.split()
            sample_idx = int(parts[0])

            grasp_indices = None
            if len(parts) > 1:
                # 解析抓取索引列表，例如 [0,1,2]
                grasp_str = parts[1].strip('[]')
                if grasp_str:
                    grasp_indices = [int(x.strip()) for x in grasp_str.split(',')]

            if 0 <= sample_idx < len(dataset):
                if grasp_indices:
                    logger.info(f"可视化样本 {sample_idx}，抓取索引: {grasp_indices}")
                    visualize_cached_sample(dataset, sample_idx, grasp_indices=grasp_indices)
                else:
                    logger.info(f"可视化样本 {sample_idx}，显示所有抓取")
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
    logger.info("SceneLeapPlusDatasetCached 可视化测试脚本")
    logger.info("验证修复后的缓存数据集类功能")
    logger.info("=" * 80)

    # 验证路径
    if not verify_paths():
        logger.error("数据路径验证失败，请检查路径设置")
        return

    # 测试选项
    tests = [
        ("1", "创建和测试 SceneLeapPlusDatasetCached", test_cached_dataset_creation),
        ("2", "测试不同坐标系统模式", test_different_modes),
        ("3", "数据一致性测试", test_data_consistency),
        ("4", "交互式可视化", None),  # 特殊处理
        ("5", "运行所有测试", None),  # 特殊处理
    ]

    logger.info("可用测试选项:")
    for option, description, _ in tests:
        logger.info(f"  {option}. {description}")

    while True:
        try:
            choice = input("\n请选择测试选项 (1-5) 或 'q' 退出: ").strip()

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
                test_different_modes()

            elif choice == "3":
                test_data_consistency()

            elif choice == "4":
                # 交互式可视化
                logger.info("创建数据集用于交互式可视化...")
                dataset = test_cached_dataset_creation()
                if dataset is not None:
                    interactive_visualization(dataset)

            elif choice == "5":
                # 运行所有测试
                logger.info("运行所有测试...")

                # 测试1: SceneLeapPlusDatasetCached
                dataset1 = test_cached_dataset_creation()

                # 测试2: 不同模式测试
                test_different_modes()

                # 测试3: 数据一致性
                test_data_consistency()

                # 可视化第一个可用的数据集
                if dataset1 is not None and len(dataset1) > 0:
                    logger.info("可视化 SceneLeapPlusDatasetCached 第一个样本...")
                    visualize_cached_sample(dataset1, 0)

                logger.info("所有测试完成！")
                logger.info("\nSceneLeapPlusDatasetCached特性总结:")
                logger.info(f"  - 每个样本包含固定数量的抓取姿态")
                logger.info(f"  - 支持多抓取并行学习架构")
                logger.info(f"  - HDF5缓存存储，提高数据加载效率")
                logger.info(f"  - 手部姿态形状: (num_grasps, 23)")
                logger.info(f"  - SE3矩阵形状: (num_grasps, 4, 4)")
                logger.info(f"  - 不同抓取使用不同颜色区分")
                logger.info(f"  - 🆕 支持穷尽采样：实现100%数据利用率")
                logger.info(f"  - 🆕 支持5种穷尽采样策略：sequential, random, interleaved, chunk_farthest_point, chunk_nearest_point")
                logger.info(f"  - 🆕 修复缓存文件名生成，包含所有关键参数")
                logger.info(f"  - 🆕 数据集可扩展10-50倍，大幅提升训练效率")

            else:
                logger.warning("无效选择，请输入 1-5 或 'q'")

        except KeyboardInterrupt:
            logger.info("\n用户中断，退出测试")
            break
        except Exception as e:
            logger.error(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
