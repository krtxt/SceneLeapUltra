import torch


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from
    https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html

    支持单抓取和多抓取格式的6D旋转表示转换

    Args:
        poses: [B, 6] 或 [B, num_grasps, 6] - 6D旋转表示

    Returns:
        rotation_matrices: [B, 3, 3] 或 [B, num_grasps, 3, 3] - 旋转矩阵
    """
    if poses.dim() == 2:
        # 单抓取格式：[B, 6] -> [B, 3, 3]
        return _compute_rotation_matrix_from_ortho6d_2d(poses)
    elif poses.dim() == 3:
        # 多抓取格式：[B, num_grasps, 6] -> [B, num_grasps, 3, 3]
        return _compute_rotation_matrix_from_ortho6d_3d(poses)
    else:
        raise ValueError(
            f"Unsupported poses dimension: {poses.dim()}. Expected 2 or 3."
        )


def _compute_rotation_matrix_from_ortho6d_2d(poses):
    """处理单抓取格式的6D旋转表示转换"""
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def _compute_rotation_matrix_from_ortho6d_3d(poses):
    """处理多抓取格式的6D旋转表示转换"""
    B, num_grasps, _ = poses.shape

    # 重塑为2D格式进行批量处理
    poses_flat = poses.view(B * num_grasps, 6)  # [B*num_grasps, 6]

    # 使用2D版本进行计算
    matrices_flat = _compute_rotation_matrix_from_ortho6d_2d(
        poses_flat
    )  # [B*num_grasps, 3, 3]

    # 重塑回多抓取格式
    matrices = matrices_flat.view(B, num_grasps, 3, 3)  # [B, num_grasps, 3, 3]

    return matrices


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally

    支持单抓取和多抓取格式的鲁棒6D旋转表示转换

    Args:
        poses: [B, 6] 或 [B, num_grasps, 6] - 6D旋转表示

    Returns:
        rotation_matrices: [B, 3, 3] 或 [B, num_grasps, 3, 3] - 旋转矩阵
    """
    if poses.dim() == 2:
        # 单抓取格式：[B, 6] -> [B, 3, 3]
        return _robust_compute_rotation_matrix_from_ortho6d_2d(poses)
    elif poses.dim() == 3:
        # 多抓取格式：[B, num_grasps, 6] -> [B, num_grasps, 3, 3]
        return _robust_compute_rotation_matrix_from_ortho6d_3d(poses)
    else:
        raise ValueError(
            f"Unsupported poses dimension: {poses.dim()}. Expected 2 or 3."
        )


def _robust_compute_rotation_matrix_from_ortho6d_2d(poses):
    """处理单抓取格式的鲁棒6D旋转表示转换"""
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def _robust_compute_rotation_matrix_from_ortho6d_3d(poses):
    """处理多抓取格式的鲁棒6D旋转表示转换"""
    B, num_grasps, _ = poses.shape

    # 重塑为2D格式进行批量处理
    poses_flat = poses.view(B * num_grasps, 6)  # [B*num_grasps, 6]

    # 使用2D版本进行计算
    matrices_flat = _robust_compute_rotation_matrix_from_ortho6d_2d(
        poses_flat
    )  # [B*num_grasps, 3, 3]

    # 重塑回多抓取格式
    matrices = matrices_flat.view(B, num_grasps, 3, 3)  # [B, num_grasps, 3, 3]

    return matrices


def normalize_vector(v):
    """
    向量归一化函数，支持批量处理

    Args:
        v: [batch, 3] - 输入向量

    Returns:
        normalized_v: [batch, 3] - 归一化后的向量
    """
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    """
    向量叉积计算函数，支持批量处理

    Args:
        u: [batch, 3] - 第一个向量
        v: [batch, 3] - 第二个向量

    Returns:
        cross: [batch, 3] - 叉积结果
    """
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


# 向后兼容性别名
def compute_rotation_matrix_from_ortho6d_legacy(poses):
    """
    向后兼容的函数别名，仅支持2D输入格式

    Args:
        poses: [B, 6] - 6D旋转表示

    Returns:
        rotation_matrices: [B, 3, 3] - 旋转矩阵
    """
    if poses.dim() != 2:
        raise ValueError("Legacy function only supports 2D input [B, 6]")
    return _compute_rotation_matrix_from_ortho6d_2d(poses)


def robust_compute_rotation_matrix_from_ortho6d_legacy(poses):
    """
    向后兼容的鲁棒函数别名，仅支持2D输入格式

    Args:
        poses: [B, 6] - 6D旋转表示

    Returns:
        rotation_matrices: [B, 3, 3] - 旋转矩阵
    """
    if poses.dim() != 2:
        raise ValueError("Legacy function only supports 2D input [B, 6]")
    return _robust_compute_rotation_matrix_from_ortho6d_2d(poses)
