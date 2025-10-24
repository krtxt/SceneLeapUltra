import numpy as np
import torch
from typing import Union, Tuple


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    变换单个点（向后兼容函数）

    Args:
        T: [4, 4] - 变换矩阵
        point: [3] - 输入点

    Returns:
        transformed_point: [3] - 变换后的点
    """
    assert T.shape == (4, 4), f"{T.shape}"
    assert point.shape == (3,), f"{point.shape}"

    transformed_point = T[:3, :3] @ point + T[:3, 3]
    assert transformed_point.shape == (3,)
    return transformed_point


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    变换多个点（向后兼容函数）

    Args:
        T: [4, 4] - 变换矩阵
        points: [N, 3] - 输入点云

    Returns:
        transformed_points: [N, 3] - 变换后的点云
    """
    assert T.shape == (4, 4), f"{T.shape}"
    N = points.shape[0]
    assert points.shape == (N, 3), f"{points.shape}"

    transformed_points = (T[:3, :3] @ points.T + T[:3, 3][:, None]).T
    assert transformed_points.shape == (N, 3)
    return transformed_points


# ==================== 多抓取批量变换函数 ====================

def transform_points_batch_numpy(transforms: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    批量变换点云（NumPy版本）
    支持单变换多点云和多变换多点云

    Args:
        transforms: [4, 4] 或 [B, 4, 4] 或 [B, num_grasps, 4, 4] - 变换矩阵
        points: [N, 3] 或 [B, N, 3] - 输入点云

    Returns:
        transformed_points: 对应维度的变换后点云
    """
    if transforms.ndim == 2 and points.ndim == 2:
        # 单变换单点云: [4,4] × [N,3] -> [N,3]
        return transform_points(transforms, points)

    elif transforms.ndim == 2 and points.ndim == 3:
        # 单变换多点云: [4,4] × [B,N,3] -> [B,N,3]
        B, N, _ = points.shape
        transformed = np.zeros_like(points)
        for b in range(B):
            transformed[b] = transform_points(transforms, points[b])
        return transformed

    elif transforms.ndim == 3 and points.ndim == 3:
        # 多变换多点云: [B,4,4] × [B,N,3] -> [B,N,3]
        B, N, _ = points.shape
        assert transforms.shape[0] == B, f"Batch size mismatch: {transforms.shape[0]} vs {B}"
        transformed = np.zeros_like(points)
        for b in range(B):
            transformed[b] = transform_points(transforms[b], points[b])
        return transformed

    elif transforms.ndim == 4 and points.ndim == 3:
        # 多抓取变换: [B,num_grasps,4,4] × [B,N,3] -> [B,num_grasps,N,3]
        B, num_grasps, _, _ = transforms.shape
        _, N, _ = points.shape
        assert transforms.shape[0] == B, f"Batch size mismatch: {transforms.shape[0]} vs {B}"

        transformed = np.zeros((B, num_grasps, N, 3))
        for b in range(B):
            for g in range(num_grasps):
                transformed[b, g] = transform_points(transforms[b, g], points[b])
        return transformed

    else:
        raise ValueError(f"Unsupported dimensions: transforms {transforms.shape}, points {points.shape}")


def transform_points_batch_torch(transforms: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    批量变换点云（PyTorch版本）
    支持单变换多点云和多变换多点云，GPU加速

    Args:
        transforms: [4, 4] 或 [B, 4, 4] 或 [B, num_grasps, 4, 4] - 变换矩阵
        points: [N, 3] 或 [B, N, 3] - 输入点云

    Returns:
        transformed_points: 对应维度的变换后点云
    """
    if transforms.dim() == 2 and points.dim() == 2:
        # 单变换单点云: [4,4] × [N,3] -> [N,3]
        return _transform_points_single_torch(transforms, points)

    elif transforms.dim() == 2 and points.dim() == 3:
        # 单变换多点云: [4,4] × [B,N,3] -> [B,N,3]
        B, N, _ = points.shape
        # 扩展变换矩阵
        transforms_expanded = transforms.unsqueeze(0).expand(B, -1, -1)  # [B,4,4]
        return _transform_points_batch_torch(transforms_expanded, points)

    elif transforms.dim() == 3 and points.dim() == 3:
        # 多变换多点云: [B,4,4] × [B,N,3] -> [B,N,3]
        return _transform_points_batch_torch(transforms, points)

    elif transforms.dim() == 4 and points.dim() == 3:
        # 多抓取变换: [B,num_grasps,4,4] × [B,N,3] -> [B,num_grasps,N,3]
        B, num_grasps, _, _ = transforms.shape
        _, N, _ = points.shape

        # 扩展点云到多抓取维度
        points_expanded = points.unsqueeze(1).expand(-1, num_grasps, -1, -1).contiguous()  # [B,num_grasps,N,3]

        # 重塑为批量处理
        transforms_flat = transforms.view(B * num_grasps, 4, 4)  # [B*num_grasps,4,4]
        points_flat = points_expanded.view(B * num_grasps, N, 3)  # [B*num_grasps,N,3]

        # 批量变换
        transformed_flat = _transform_points_batch_torch(transforms_flat, points_flat)

        # 重塑回多抓取格式
        transformed = transformed_flat.view(B, num_grasps, N, 3)
        return transformed

    else:
        raise ValueError(f"Unsupported dimensions: transforms {transforms.shape}, points {points.shape}")


def _transform_points_single_torch(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """PyTorch单变换实现"""
    # points: [N, 3], T: [4, 4]
    rotation = T[:3, :3]  # [3, 3]
    translation = T[:3, 3]  # [3]

    # 变换: R @ points.T + t
    transformed = torch.mm(rotation, points.t()).t() + translation  # [N, 3]
    return transformed


def _transform_points_batch_torch(transforms: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """PyTorch批量变换实现"""
    # transforms: [B, 4, 4], points: [B, N, 3]
    B, N, _ = points.shape

    rotation = transforms[:, :3, :3]  # [B, 3, 3]
    translation = transforms[:, :3, 3]  # [B, 3]

    # 批量矩阵乘法: [B, 3, 3] @ [B, 3, N] -> [B, 3, N]
    transformed = torch.bmm(rotation, points.transpose(1, 2)).transpose(1, 2)  # [B, N, 3]
    transformed = transformed + translation.unsqueeze(1)  # [B, N, 3]

    return transformed


# ==================== 统一接口函数 ====================

def transform_points_multi_grasp(transforms: Union[np.ndarray, torch.Tensor],
                                points: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    多抓取点云变换统一接口
    自动检测输入类型并调用对应的实现

    Args:
        transforms: 变换矩阵，支持多种格式:
            - [4, 4]: 单变换矩阵
            - [B, 4, 4]: 批量变换矩阵
            - [B, num_grasps, 4, 4]: 多抓取变换矩阵
        points: 点云数据，支持格式:
            - [N, 3]: 单点云
            - [B, N, 3]: 批量点云

    Returns:
        transformed_points: 变换后的点云，维度根据输入自动确定
    """
    if isinstance(transforms, np.ndarray) and isinstance(points, np.ndarray):
        return transform_points_batch_numpy(transforms, points)
    elif isinstance(transforms, torch.Tensor) and isinstance(points, torch.Tensor):
        return transform_points_batch_torch(transforms, points)
    else:
        raise TypeError("transforms and points must be both numpy arrays or both torch tensors")


# ==================== 点云采样和网格生成 ====================

def get_points_in_grid(
    lb: np.ndarray,
    ub: np.ndarray,
    num_pts_x: int,
    num_pts_y: int,
    num_pts_z: int,
) -> np.ndarray:
    """
    在指定边界内生成网格点（向后兼容函数）

    Args:
        lb: [3] - 下边界 [x_min, y_min, z_min]
        ub: [3] - 上边界 [x_max, y_max, z_max]
        num_pts_x: X方向点数
        num_pts_y: Y方向点数
        num_pts_z: Z方向点数

    Returns:
        query_points: [num_pts_x, num_pts_y, num_pts_z, 3] - 网格点
    """
    x_min, y_min, z_min = lb
    x_max, y_max, z_max = ub
    x_coords = np.linspace(x_min, x_max, num_pts_x)
    y_coords = np.linspace(y_min, y_max, num_pts_y)
    z_coords = np.linspace(z_min, z_max, num_pts_z)

    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    query_points_in_region = np.stack([xx, yy, zz], axis=-1)
    assert query_points_in_region.shape == (num_pts_x, num_pts_y, num_pts_z, 3)
    return query_points_in_region


def get_points_in_grid_batch(
    bounds: Union[np.ndarray, torch.Tensor],
    num_pts: Tuple[int, int, int],
    batch_size: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    批量生成网格点，支持多个边界框

    Args:
        bounds: [B, 2, 3] 或 [2, 3] - 边界框 [下界, 上界]
        num_pts: (num_x, num_y, num_z) - 各方向点数
        batch_size: 批量大小（当bounds为[2,3]时使用）

    Returns:
        grid_points: [B, num_x, num_y, num_z, 3] - 批量网格点
    """
    num_x, num_y, num_z = num_pts

    if isinstance(bounds, np.ndarray):
        if bounds.ndim == 2:
            # 单个边界框，复制到批量
            bounds = np.tile(bounds[None, :, :], (batch_size, 1, 1))

        B = bounds.shape[0]
        grid_points = np.zeros((B, num_x, num_y, num_z, 3))

        for b in range(B):
            lb, ub = bounds[b, 0], bounds[b, 1]
            grid_points[b] = get_points_in_grid(lb, ub, num_x, num_y, num_z)

        return grid_points

    elif isinstance(bounds, torch.Tensor):
        if bounds.dim() == 2:
            # 单个边界框，复制到批量
            bounds = bounds.unsqueeze(0).expand(batch_size, -1, -1)

        B = bounds.shape[0]
        device = bounds.device

        # 生成坐标网格
        grid_points = torch.zeros((B, num_x, num_y, num_z, 3), device=device)

        for b in range(B):
            lb, ub = bounds[b, 0], bounds[b, 1]  # [3], [3]

            x_coords = torch.linspace(lb[0], ub[0], num_x, device=device)
            y_coords = torch.linspace(lb[1], ub[1], num_y, device=device)
            z_coords = torch.linspace(lb[2], ub[2], num_z, device=device)

            xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
            grid_points[b] = torch.stack([xx, yy, zz], dim=-1)

        return grid_points

    else:
        raise TypeError("bounds must be numpy array or torch tensor")


# ==================== 点云处理工具函数 ====================

def apply_se3_to_points(se3_matrices: Union[np.ndarray, torch.Tensor],
                       points: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    应用SE(3)变换到点云，支持多抓取格式

    Args:
        se3_matrices: SE(3)变换矩阵，支持格式:
            - [4, 4]: 单变换
            - [B, 4, 4]: 批量变换
            - [B, num_grasps, 4, 4]: 多抓取变换
        points: 点云数据:
            - [N, 3]: 单点云
            - [B, N, 3]: 批量点云

    Returns:
        transformed_points: 变换后的点云
    """
    return transform_points_multi_grasp(se3_matrices, points)


def compute_point_distances(points1: Union[np.ndarray, torch.Tensor],
                          points2: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    计算两组点云之间的距离，支持批量处理

    Args:
        points1: [B, N1, 3] 或 [N1, 3] - 第一组点云
        points2: [B, N2, 3] 或 [N2, 3] - 第二组点云

    Returns:
        distances: [B, N1, N2] 或 [N1, N2] - 点间距离矩阵
    """
    if isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray):
        return _compute_point_distances_numpy(points1, points2)
    elif isinstance(points1, torch.Tensor) and isinstance(points2, torch.Tensor):
        return _compute_point_distances_torch(points1, points2)
    else:
        raise TypeError("points1 and points2 must be both numpy arrays or both torch tensors")


def _compute_point_distances_numpy(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """NumPy版本的点距离计算"""
    if points1.ndim == 2 and points2.ndim == 2:
        # 单点云距离: [N1, 3] × [N2, 3] -> [N1, N2]
        diff = points1[:, None, :] - points2[None, :, :]  # [N1, N2, 3]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # [N1, N2]
        return distances

    elif points1.ndim == 3 and points2.ndim == 3:
        # 批量点云距离: [B, N1, 3] × [B, N2, 3] -> [B, N1, N2]
        B, N1, _ = points1.shape
        _, N2, _ = points2.shape

        diff = points1[:, :, None, :] - points2[:, None, :, :]  # [B, N1, N2, 3]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # [B, N1, N2]
        return distances

    else:
        raise ValueError(f"Unsupported dimensions: points1 {points1.shape}, points2 {points2.shape}")


def _compute_point_distances_torch(points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
    """PyTorch版本的点距离计算"""
    if points1.dim() == 2 and points2.dim() == 2:
        # 单点云距离: [N1, 3] × [N2, 3] -> [N1, N2]
        diff = points1.unsqueeze(1) - points2.unsqueeze(0)  # [N1, N2, 3]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [N1, N2]
        return distances

    elif points1.dim() == 3 and points2.dim() == 3:
        # 批量点云距离: [B, N1, 3] × [B, N2, 3] -> [B, N1, N2]
        diff = points1.unsqueeze(2) - points2.unsqueeze(1)  # [B, N1, N2, 3]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [B, N1, N2]
        return distances

    else:
        raise ValueError(f"Unsupported dimensions: points1 {points1.shape}, points2 {points2.shape}")


def sample_points_on_surface(vertices: Union[np.ndarray, torch.Tensor],
                            faces: Union[np.ndarray, torch.Tensor],
                            num_samples: int) -> Union[np.ndarray, torch.Tensor]:
    """
    在网格表面采样点，支持批量处理

    Args:
        vertices: [B, V, 3] 或 [V, 3] - 顶点坐标
        faces: [B, F, 3] 或 [F, 3] - 面片索引
        num_samples: 采样点数

    Returns:
        sampled_points: [B, num_samples, 3] 或 [num_samples, 3] - 采样点
    """
    # 这里提供一个简化的实现，实际使用中可能需要更复杂的采样算法
    if isinstance(vertices, np.ndarray):
        return _sample_points_numpy(vertices, faces, num_samples)
    elif isinstance(vertices, torch.Tensor):
        return _sample_points_torch(vertices, faces, num_samples)
    else:
        raise TypeError("vertices must be numpy array or torch tensor")


def _sample_points_numpy(vertices: np.ndarray, faces: np.ndarray, num_samples: int) -> np.ndarray:
    """NumPy版本的表面采样（简化实现）"""
    if vertices.ndim == 2:
        # 单网格采样
        V, _ = vertices.shape
        # 简化：随机选择顶点作为采样点
        indices = np.random.choice(V, num_samples, replace=True)
        return vertices[indices]

    elif vertices.ndim == 3:
        # 批量网格采样
        B, V, _ = vertices.shape
        sampled = np.zeros((B, num_samples, 3))
        for b in range(B):
            indices = np.random.choice(V, num_samples, replace=True)
            sampled[b] = vertices[b, indices]
        return sampled

    else:
        raise ValueError(f"Unsupported vertices dimension: {vertices.ndim}")


def _sample_points_torch(vertices: torch.Tensor, faces: torch.Tensor, num_samples: int) -> torch.Tensor:
    """PyTorch版本的表面采样（简化实现）"""
    if vertices.dim() == 2:
        # 单网格采样
        V, _ = vertices.shape
        # 简化：随机选择顶点作为采样点
        indices = torch.randint(0, V, (num_samples,), device=vertices.device)
        return vertices[indices]

    elif vertices.dim() == 3:
        # 批量网格采样
        B, V, _ = vertices.shape
        sampled = torch.zeros((B, num_samples, 3), device=vertices.device)
        for b in range(B):
            indices = torch.randint(0, V, (num_samples,), device=vertices.device)
            sampled[b] = vertices[b, indices]
        return sampled

    else:
        raise ValueError(f"Unsupported vertices dimension: {vertices.dim()}")
