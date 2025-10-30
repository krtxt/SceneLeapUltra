"""
几何注意力偏置模块

在scene cross-attention中引入基于相对几何关系的注意力偏置，
用于增强模型对grasp与点云空间关系的感知能力。

参考: 3DETR, V-DETR中的3D相对位置编码(Vertex RPE)
"""

import torch
import torch.nn as nn
from typing import Optional, List
import logging


class GeometricAttentionBias(nn.Module):
    """
    计算grasp tokens与场景点之间的几何注意力偏置
    
    公式: bias_ij = w^T φ(R_i^T(p_j - t_i))
    
    其中:
    - t_i, R_i: 第i个grasp token的平移和旋转
    - p_j: 第j个场景点
    - φ: MLP网络，将几何特征映射为标量
    
    Args:
        d_model: 模型隐藏维度
        hidden_dims: MLP隐藏层维度列表
        feature_types: 使用的几何特征类型列表
        num_heads: 注意力头数（用于生成per-head bias）
        rot_type: 旋转表示类型 ('quat' or 'r6d')
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dims: List[int] = [128, 64],
        feature_types: List[str] = ["relative_pos", "distance"],
        num_heads: int = 8,
        rot_type: str = "quat"
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dims = hidden_dims
        self.feature_types = feature_types
        self.num_heads = num_heads
        self.rot_type = rot_type
        
        # 计算输入特征维度
        self.feature_dim = self._compute_feature_dim()
        
        # MLP: 将几何特征映射为per-head bias
        layers = []
        in_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
        # 输出层: 生成每个head的bias (不使用激活函数)
        layers.append(nn.Linear(in_dim, num_heads))
        
        self.mlp = nn.Sequential(*layers)
        
        # 可学习偏置缩放参数（逐头；零初始化，训练初期不引入几何偏置影响）
        self.bias_scale = nn.Parameter(torch.zeros(num_heads))
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"GeometricAttentionBias initialized: feature_dim={self.feature_dim}, "
            f"hidden_dims={hidden_dims}, num_heads={num_heads}, "
            f"feature_types={feature_types}, rot_type={rot_type}"
        )
    
    def _compute_feature_dim(self) -> int:
        """计算特征维度"""
        dim = 0
        for feat_type in self.feature_types:
            if feat_type == "relative_pos":
                dim += 3  # 3D相对位置
            elif feat_type == "distance":
                dim += 1  # 欧氏距离
            elif feat_type == "direction":
                dim += 3  # 归一化方向向量
            elif feat_type == "distance_log":
                dim += 1  # log(distance + eps)
            else:
                raise ValueError(f"Unknown feature type: {feat_type}")
        return dim
    
    def forward(
        self,
        grasp_tokens: torch.Tensor,
        scene_points: torch.Tensor,
        grasp_poses: torch.Tensor
    ) -> torch.Tensor:
        """
        计算几何注意力偏置
        
        Args:
            grasp_tokens: (B, N_grasps, d_model) - 查询tokens（仅用于获取shape）
            scene_points: (B, N_points, 3) - 场景点云坐标
            grasp_poses: (B, N_grasps, d_x) - grasp姿态（包含平移和旋转）
        
        Returns:
            bias: (B, num_heads, N_grasps, N_points) - 注意力偏置
        """
        B, N_grasps, _ = grasp_tokens.shape
        N_points = scene_points.shape[1]
        
        # 提取平移和旋转
        translations = grasp_poses[..., :3]  # (B, N_grasps, 3)
        rotation_repr = grasp_poses[..., 3:]  # (B, N_grasps, rot_dim)
        
        # 将旋转表示转换为旋转矩阵
        rotation_matrices = self._rotation_repr_to_matrix(rotation_repr)  # (B, N_grasps, 3, 3)
        
        # 计算几何特征
        geometric_features = self._compute_geometric_features(
            translations, rotation_matrices, scene_points
        )  # (B, N_grasps, N_points, feature_dim)
        
        # 通过MLP计算bias
        # Reshape for MLP: (B*N_grasps*N_points, feature_dim)
        features_flat = geometric_features.reshape(-1, self.feature_dim)
        bias_flat = self.mlp(features_flat)  # (B*N_grasps*N_points, num_heads)
        
        # Reshape to attention bias format: (B, num_heads, N_grasps, N_points)
        bias = bias_flat.reshape(B, N_grasps, N_points, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, num_heads, N_grasps, N_points)
        
        # 应用逐头可学习缩放，避免早期训练不稳定
        bias = bias * self.bias_scale.view(1, self.num_heads, 1, 1)
        
        return bias
    
    def _rotation_repr_to_matrix(self, rotation_repr: torch.Tensor) -> torch.Tensor:
        """
        将旋转表示转换为旋转矩阵
        
        Args:
            rotation_repr: (B, N, rot_dim) - 旋转表示
        
        Returns:
            rotation_matrices: (B, N, 3, 3)
        """
        if self.rot_type == "quat":
            # 四元数: (w, x, y, z) 或 (x, y, z, w) - 需要根据实际情况调整
            # 假设格式为 (x, y, z, w)，共4维
            return self._quaternion_to_matrix(rotation_repr[..., :4])
        elif self.rot_type == "r6d":
            # 6D rotation representation
            return self._r6d_to_matrix(rotation_repr[..., :6])
        else:
            raise ValueError(f"Unknown rotation type: {self.rot_type}")
    
    def _quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        四元数转旋转矩阵
        
        Args:
            quaternions: (..., 4) - 四元数 (x, y, z, w)
        
        Returns:
            matrices: (..., 3, 3)
        """
        # 归一化四元数
        quaternions = quaternions / (quaternions.norm(dim=-1, keepdim=True) + 1e-8)
        
        x, y, z, w = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
        
        # 计算旋转矩阵元素
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        matrix = torch.stack([
            torch.stack([1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)], dim=-1)
        ], dim=-2)
        
        return matrix
    
    def _r6d_to_matrix(self, r6d: torch.Tensor) -> torch.Tensor:
        """
        6D旋转表示转旋转矩阵 (Zhou et al. CVPR 2019)
        
        Args:
            r6d: (..., 6) - 6D representation (前两列的列向量)
        
        Returns:
            matrices: (..., 3, 3)
        """
        # 取前两个3D向量
        v1 = r6d[..., :3]  # 第一列
        v2 = r6d[..., 3:6]  # 第二列
        
        # Gram-Schmidt正交化
        b1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
        b2 = v2 - (b1 * v2).sum(dim=-1, keepdim=True) * b1
        b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        
        matrix = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)
        
        return matrix
    
    def _compute_geometric_features(
        self,
        translations: torch.Tensor,
        rotation_matrices: torch.Tensor,
        scene_points: torch.Tensor
    ) -> torch.Tensor:
        """
        计算几何特征
        
        Args:
            translations: (B, N_grasps, 3)
            rotation_matrices: (B, N_grasps, 3, 3)
            scene_points: (B, N_points, 3)
        
        Returns:
            features: (B, N_grasps, N_points, feature_dim)
        """
        B, N_grasps, _ = translations.shape
        N_points = scene_points.shape[1]
        
        # 扩展维度以进行广播计算
        # translations: (B, N_grasps, 1, 3)
        # scene_points: (B, 1, N_points, 3)
        t_expanded = translations.unsqueeze(2)  # (B, N_grasps, 1, 3)
        p_expanded = scene_points.unsqueeze(1)  # (B, 1, N_points, 3)
        
        # 计算相对位置: p_j - t_i
        relative_pos_world = p_expanded - t_expanded  # (B, N_grasps, N_points, 3)
        
        # 转换到grasp局部坐标系: R_i^T (p_j - t_i)
        # rotation_matrices: (B, N_grasps, 3, 3)
        # 需要转置: R^T
        rotation_matrices_T = rotation_matrices.transpose(-2, -1)  # (B, N_grasps, 3, 3)
        
        # 批量矩阵乘法
        # relative_pos_world: (B, N_grasps, N_points, 3) -> (B, N_grasps, N_points, 3, 1)
        # rotation_matrices_T: (B, N_grasps, 3, 3) -> (B, N_grasps, 1, 3, 3)
        relative_pos_local = torch.matmul(
            rotation_matrices_T.unsqueeze(2),  # (B, N_grasps, 1, 3, 3)
            relative_pos_world.unsqueeze(-1)   # (B, N_grasps, N_points, 3, 1)
        ).squeeze(-1)  # (B, N_grasps, N_points, 3)
        
        # 构建特征向量
        features = []
        
        for feat_type in self.feature_types:
            if feat_type == "relative_pos":
                features.append(relative_pos_local)  # (B, N_grasps, N_points, 3)
            
            elif feat_type == "distance":
                distance = relative_pos_local.norm(dim=-1, keepdim=True)  # (B, N_grasps, N_points, 1)
                features.append(distance)
            
            elif feat_type == "direction":
                distance = relative_pos_local.norm(dim=-1, keepdim=True) + 1e-8
                direction = relative_pos_local / distance  # (B, N_grasps, N_points, 3)
                features.append(direction)
            
            elif feat_type == "distance_log":
                distance = relative_pos_local.norm(dim=-1, keepdim=True)
                log_distance = torch.log(distance + 1e-3)  # (B, N_grasps, N_points, 1)
                features.append(log_distance)
        
        # 拼接所有特征
        geometric_features = torch.cat(features, dim=-1)  # (B, N_grasps, N_points, feature_dim)
        
        return geometric_features


def extract_scene_xyz(scene_context: torch.Tensor, data: dict) -> Optional[torch.Tensor]:
    """
    从data中提取场景点云的xyz坐标
    
    Args:
        scene_context: (B, N_points, d_model) - 场景特征
        data: 数据字典，可能包含scene_pc等字段
    
    Returns:
        scene_xyz: (B, N_points, 3) 或 None
    """
    # 尝试从data中获取原始点云坐标
    if 'scene_pc' in data and data['scene_pc'] is not None:
        scene_pc = data['scene_pc']
        if isinstance(scene_pc, torch.Tensor):
            # scene_pc通常是 (B, N_points, C)，其中前3维是xyz
            return scene_pc[..., :3]
    
    # 如果无法获取，返回None
    return None

