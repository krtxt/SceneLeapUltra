#!/usr/bin/env python3
"""
使用Open3D可视化重构后的SceneLeapProDataset类的功能
展示6D点云（xyz+rgb）、灵巧手抓取姿态和目标物体mesh

主要功能：
1. 从SceneLeapProDataset加载样本数据
2. 可视化6D点云（xyz+rgb颜色）
3. 可视化LEAP灵巧手抓取姿态
4. 可视化目标物体mesh（实体或线框模式）
5. 可选：高亮显示目标物体区域
6. 支持四种坐标系统模式的可视化对比

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
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.sceneleappro_dataset import SceneLeapProDataset
from utils.hand_model_origin import HandModel, HandModelType


def create_coordinate_frame(size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """创建坐标轴参考框架"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return coordinate_frame

def create_point_cloud_from_sample(scene_pc: torch.Tensor, object_mask: Optional[torch.Tensor] = None) -> o3d.geometry.PointCloud:
    """
    从样本数据创建Open3D点云
    
    Args:
        scene_pc: torch.Tensor of shape (N, 6) - xyz + rgb
        object_mask: Optional torch.Tensor of shape (M,) - 目标物体掩码
    
    Returns:
        o3d.geometry.PointCloud: Open3D点云对象
    """
    # 转换为numpy数组
    if isinstance(scene_pc, torch.Tensor):
        scene_pc_np = scene_pc.detach().cpu().numpy()
    else:
        scene_pc_np = scene_pc
    
    # 确保形状正确
    assert scene_pc_np.shape[1] == 6, f"期望6维点云 (xyz+rgb)，实际得到 {scene_pc_np.shape[1]} 维"
    
    # 提取xyz坐标和rgb颜色
    xyz = scene_pc_np[:, :3]
    rgb = scene_pc_np[:, 3:6]
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def create_highlighted_point_cloud(scene_pc: torch.Tensor, object_mask: torch.Tensor) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    创建高亮显示目标物体的点云
    
    Args:
        scene_pc: torch.Tensor of shape (N, 6) - xyz + rgb
        object_mask: torch.Tensor of shape (M,) - 目标物体掩码
    
    Returns:
        Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]: (背景点云, 目标物体点云)
    """
    scene_pc_np = scene_pc.detach().cpu().numpy()
    object_mask_np = object_mask.detach().cpu().numpy()
    
    # 注意：object_mask可能与点云大小不匹配（来自原始图像）
    # 我们需要处理这种情况
    if len(object_mask_np) != len(scene_pc_np):
        print(f"警告: object_mask大小 ({len(object_mask_np)}) 与点云大小 ({len(scene_pc_np)}) 不匹配")
        print("将创建统一颜色的点云")
        return create_point_cloud_from_sample(scene_pc), None
    
    # 分离背景和目标物体点
    background_points = scene_pc_np[~object_mask_np]
    object_points = scene_pc_np[object_mask_np]
    
    # 创建背景点云（保持原始颜色但降低亮度）
    background_pcd = o3d.geometry.PointCloud()
    if len(background_points) > 0:
        background_pcd.points = o3d.utility.Vector3dVector(background_points[:, :3])
        background_colors = background_points[:, 3:6] * 0.5  # 降低亮度
        background_pcd.colors = o3d.utility.Vector3dVector(background_colors)
    
    # 创建目标物体点云（高亮显示）
    object_pcd = o3d.geometry.PointCloud()
    if len(object_points) > 0:
        object_pcd.points = o3d.utility.Vector3dVector(object_points[:, :3])
        # 使用红色高亮显示目标物体
        object_colors = np.ones((len(object_points), 3)) * [1.0, 0.0, 0.0]  # 红色
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    
    return background_pcd, object_pcd

def analyze_hand_pose_format(hand_pose: torch.Tensor) -> str:
    """
    分析手部姿态的格式

    根据SceneLeapProDataset的实现，hand_model_pose_reordered的格式是：
    [P, Joints, Q_wxyz] = [平移(3), 关节角度(16), 四元数(4)] = 23维

    Args:
        hand_pose: torch.Tensor of shape (23,) - 手部姿态参数

    Returns:
        str: 'quaternion_pjq' 表示 [P, Joints, Q_wxyz] 格式
    """
    if hand_pose.shape[0] != 23:
        return 'unknown'

    # 根据SceneLeapProDataset的实现，格式固定为 [P, Joints, Q_wxyz]
    # P: [0:3] - 平移
    # Joints: [3:19] - 16个关节角度
    # Q_wxyz: [19:23] - 4维四元数 (w, x, y, z)

    return 'quaternion_pjq'

def create_hand_mesh(hand_pose: torch.Tensor, hand_model: HandModel) -> Optional[o3d.geometry.TriangleMesh]:
    """
    从手部姿态创建手部mesh

    Args:
        hand_pose: torch.Tensor of shape (23,) - 手部姿态参数
        hand_model: HandModel - 手部模型实例

    Returns:
        o3d.geometry.TriangleMesh: 手部mesh，如果失败则返回None
    """
    try:
        # 确保手部姿态是正确的形状
        if hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)  # (1, 23)

        # 分析手部姿态格式
        pose_format = analyze_hand_pose_format(hand_pose.squeeze(0))
        print(f"检测到手部姿态格式: {pose_format}")

        # 根据SceneLeapProDataset的格式 [P, Joints, Q_wxyz]，需要重新排列为 [P, Q_wxyz, Joints]
        if pose_format == 'quaternion_pjq':
            # 原格式: [P(3), Joints(16), Q_wxyz(4)]
            # 目标格式: [P(3), Q_wxyz(4), Joints(16)]
            P = hand_pose[:, :3]  # 平移
            Joints = hand_pose[:, 3:19]  # 关节角度
            Q_wxyz = hand_pose[:, 19:23]  # 四元数

            # 重新排列为 [P, Q_wxyz, Joints]
            reordered_pose = torch.cat([P, Q_wxyz, Joints], dim=1)
            print(f"重新排列手部姿态: {hand_pose.shape} -> {reordered_pose.shape}")
            print(f"P: {P.shape}, Q_wxyz: {Q_wxyz.shape}, Joints: {Joints.shape}")

            hand_model.set_parameters_quat(reordered_pose)
        else:
            print(f"未知的手部姿态格式，尝试直接使用四元数格式")
            hand_model.set_parameters_quat(hand_pose)

        # 获取trimesh数据
        trimesh_data = hand_model.get_trimesh_data(0)

        # 转换为Open3D mesh
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(trimesh_data.vertices)
        hand_mesh.triangles = o3d.utility.Vector3iVector(trimesh_data.faces)

        # 设置颜色（蓝色）
        hand_mesh.paint_uniform_color([0.0, 0.0, 1.0])

        # 计算法向量以获得更好的渲染效果
        hand_mesh.compute_vertex_normals()

        return hand_mesh

    except Exception as e:
        print(f"创建手部mesh失败: {e}")
        return None

def create_object_mesh(obj_verts: torch.Tensor, obj_faces: torch.Tensor,
                      color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
                      wireframe: bool = False) -> Optional[o3d.geometry.TriangleMesh]:
    """
    从顶点和面数据创建目标物体mesh

    Args:
        obj_verts: torch.Tensor of shape (V, 3) - 物体顶点坐标
        obj_faces: torch.Tensor of shape (F, 3) - 物体面索引
        color: Tuple[float, float, float] - mesh颜色 (R, G, B)
        wireframe: bool - 是否显示为线框模式

    Returns:
        o3d.geometry.TriangleMesh: 物体mesh，如果失败则返回None
    """
    try:
        # 检查输入数据
        if obj_verts.numel() == 0 or obj_faces.numel() == 0:
            print("物体mesh数据为空")
            return None

        # 转换为numpy数组
        if isinstance(obj_verts, torch.Tensor):
            vertices_np = obj_verts.detach().cpu().numpy()
        else:
            vertices_np = obj_verts

        if isinstance(obj_faces, torch.Tensor):
            faces_np = obj_faces.detach().cpu().numpy()
        else:
            faces_np = obj_faces

        # 检查数据形状
        if vertices_np.shape[1] != 3:
            print(f"错误：顶点数据应该是 (V, 3)，实际得到 {vertices_np.shape}")
            return None

        if faces_np.shape[1] != 3:
            print(f"错误：面数据应该是 (F, 3)，实际得到 {faces_np.shape}")
            return None

        # 创建Open3D mesh
        object_mesh = o3d.geometry.TriangleMesh()
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        object_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))

        # 设置颜色
        object_mesh.paint_uniform_color(color)

        # 计算法向量以获得更好的渲染效果
        object_mesh.compute_vertex_normals()

        # 如果是线框模式，转换为线框
        if wireframe:
            # 创建线框mesh
            wireframe_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(object_mesh)
            wireframe_mesh.paint_uniform_color(color)
            print(f"✓ 物体线框mesh创建成功: {len(vertices_np)} 个顶点, {len(faces_np)} 个面")
            return wireframe_mesh

        print(f"✓ 物体mesh创建成功: {len(vertices_np)} 个顶点, {len(faces_np)} 个面")

        return object_mesh

    except Exception as e:
        print(f"创建物体mesh失败: {e}")
        return None

def visualize_sample(dataset: SceneLeapProDataset, sample_idx: int = 0, highlight_object: bool = True,
                    show_object_mesh: bool = True, object_mesh_wireframe: bool = False,
                    mode: str = "camera_centric"):
    """
    可视化数据集样本

    Args:
        dataset: SceneLeapProDataset实例
        sample_idx: 样本索引
        highlight_object: 是否高亮显示目标物体
        show_object_mesh: 是否显示目标物体mesh
        object_mesh_wireframe: 是否以线框模式显示物体mesh
        mode: 坐标系统模式
    """
    print(f"正在可视化样本 {sample_idx} (模式: {mode})...")

    # 获取样本数据
    try:
        sample = dataset[sample_idx]
    except Exception as e:
        print(f"获取样本失败: {e}")
        return

    # 打印样本信息
    print(f"样本信息:")
    print(f"  - 坐标模式: {mode}")
    print(f"  - 场景ID: {sample['scene_id']}")
    print(f"  - 视角索引: {sample['depth_view_index']}")
    print(f"  - 物体代码: {sample['obj_code']}")
    print(f"  - 正面提示词: '{sample['positive_prompt']}'")
    print(f"  - 负面提示词: {sample['negative_prompts']}")
    print(f"  - 点云形状: {sample['scene_pc'].shape}")
    print(f"  - 物体掩码形状: {sample['object_mask'].shape}")
    print(f"  - 手部姿态形状: {sample['hand_model_pose'].shape}")
    print(f"  - 物体顶点形状: {sample['obj_verts'].shape}")
    print(f"  - 物体面形状: {sample['obj_faces'].shape}")

    # 创建可视化对象列表
    vis_objects = []

    # 添加坐标轴
    coordinate_frame = create_coordinate_frame(size=0.1)
    vis_objects.append(coordinate_frame)

    # 创建点云
    scene_pc = sample['scene_pc']
    object_mask = sample['object_mask']

    if highlight_object and len(object_mask) == len(scene_pc):
        # 创建高亮显示的点云
        background_pcd, object_pcd = create_highlighted_point_cloud(scene_pc, object_mask)
        if background_pcd is not None:
            vis_objects.append(background_pcd)
        if object_pcd is not None:
            vis_objects.append(object_pcd)
    else:
        # 创建普通点云
        pcd = create_point_cloud_from_sample(scene_pc)
        vis_objects.append(pcd)

    # 创建目标物体mesh
    if show_object_mesh:
        try:
            print("正在创建目标物体mesh...")
            obj_verts = sample['obj_verts']
            obj_faces = sample['obj_faces']

            # 选择mesh颜色和模式
            mesh_color = (0.0, 0.8, 0.0) if not object_mesh_wireframe else (0.0, 1.0, 0.0)

            object_mesh = create_object_mesh(
                obj_verts, obj_faces,
                color=mesh_color,
                wireframe=object_mesh_wireframe
            )

            if object_mesh is not None:
                vis_objects.append(object_mesh)
                mesh_type = "线框" if object_mesh_wireframe else "实体"
                print(f"✓ 目标物体{mesh_type}mesh创建成功")
            else:
                print("✗ 目标物体mesh创建失败")

        except Exception as e:
            print(f"目标物体mesh创建失败: {e}")

    # 创建手部模型
    try:
        print("正在初始化LEAP手部模型...")
        hand_model = HandModel(hand_model_type=HandModelType.LEAP, device='cpu')

        # 创建手部mesh
        hand_pose = sample['hand_model_pose']
        hand_mesh = create_hand_mesh(hand_pose, hand_model)

        if hand_mesh is not None:
            vis_objects.append(hand_mesh)
            print("✓ 手部mesh创建成功")
        else:
            print("✗ 手部mesh创建失败")

    except Exception as e:
        print(f"手部模型初始化失败: {e}")

    # 打印可视化组件总结
    print(f"\n可视化组件总结:")
    print(f"  - 坐标轴: ✓")
    print(f"  - 点云: ✓ ({len(scene_pc)} 个点)")
    if highlight_object and len(object_mask) == len(scene_pc):
        num_object_points = torch.sum(object_mask).item()
        print(f"  - 目标物体高亮: ✓ ({num_object_points} 个目标点)")
    if show_object_mesh and 'obj_verts' in sample:
        print(f"  - 目标物体mesh: {'✓' if any('TriangleMesh' in str(type(obj)) for obj in vis_objects) else '✗'}")
    print(f"  - 手部mesh: {'✓' if len([obj for obj in vis_objects if 'TriangleMesh' in str(type(obj)) and hasattr(obj, 'paint_uniform_color')]) > 0 else '✗'}")

    # 创建可视化窗口
    print("\n正在启动Open3D可视化...")
    print("可视化说明:")
    print("  - 红色点: 目标物体点云")
    print("  - 灰色点: 背景点云")
    if show_object_mesh:
        mesh_type = "线框" if object_mesh_wireframe else "实体"
        print(f"  - 绿色{mesh_type}: 目标物体mesh")
    print("  - 蓝色mesh: 手部mesh")
    print("  - RGB坐标轴: 世界坐标系")
    print("\n操作提示:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 按 'H' 键: 显示更多快捷键帮助")

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name=f"SceneLeapPro [{mode}] Sample {sample_idx} - {sample['obj_code']}",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def main():
    """主函数"""
    print("=" * 60)
    print("SceneLeapProDataset Open3D 可视化测试")
    print("=" * 60)

    # 数据路径配置
    root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/723_sub_15"
    succ_grasp_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect"
    obj_root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models"

    # 四种坐标系统模式
    modes = ["camera_centric_obj_mean_normalized", "camera_centric_scene_mean_normalized", "object_centric", "camera_centric"]

    # 样本索引（可以修改来测试不同样本）
    sample_idx = 0

    try:
        for i, mode in enumerate(modes):
            print(f"\n{'='*60}")
            print(f"测试模式 {i+1}/4: {mode}")
            print(f"{'='*60}")

            # 为每种模式创建数据集实例
            print(f"正在初始化数据集 (模式: {mode})...")
            dataset = SceneLeapProDataset(
                root_dir=root_dir,
                succ_grasp_dir=succ_grasp_dir,
                obj_root_dir=obj_root_dir,
                mode=mode,
                num_neg_prompts=4,
                enable_cropping=True,
                max_points=20000
            )

            print(f"✓ 数据集初始化成功，包含 {len(dataset)} 个样本")

            if len(dataset) == 0:
                print(f"数据集为空，跳过模式 {mode}")
                continue

            # 可视化样本
            print(f"即将可视化样本 {sample_idx}，模式: {mode}")
            print("关闭当前窗口后将显示下一个模式...")

            # 可以选择实体mesh或线框mesh
            # object_mesh_wireframe=False 显示实体mesh
            # object_mesh_wireframe=True 显示线框mesh（更容易看到内部结构）
            visualize_sample(
                dataset, sample_idx,
                highlight_object=True,
                show_object_mesh=True,
                object_mesh_wireframe=False,  # 改为True可显示线框
                mode=mode
            )

            print(f"✓ 模式 {mode} 可视化完成")

        print(f"\n{'='*60}")
        print("所有模式测试完成！")
        print(f"{'='*60}")

    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
