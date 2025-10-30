"""
Local neighborhood selectors for grasp-aware scene conditioning.

Provides kNN and ball query methods to select local scene points
based on grasp translations.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple


def knn_query(
    query_points: torch.Tensor,
    key_points: torch.Tensor,
    k: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k-近邻查询
    
    Args:
        query_points: (B, G, 3) - 查询点（抓取平移）
        key_points: (B, N, 3) - 候选点（场景点云）
        k: 邻居数量
        mask: (B, N) or None - 场景掩码，1=valid, 0=padding
        
    Returns:
        indices: (B, G, k) - 每个查询点的 k 个最近邻索引
        distances: (B, G, k) - 对应的距离
    """
    B, G, _ = query_points.shape
    N = key_points.shape[1]
    
    # 计算距离：(B, G, N)
    # query_points: (B, G, 1, 3)
    # key_points: (B, 1, N, 3)
    dists = torch.cdist(query_points, key_points, p=2)  # (B, G, N)
    
    # 如果有 mask，将 padding 位置的距离设为无穷大
    if mask is not None:
        # mask: (B, N) -> (B, 1, N)
        mask_expanded = mask.unsqueeze(1)  # (B, 1, N)
        dists = dists.masked_fill(mask_expanded == 0, float('inf'))
    
    # 取 top-k 最小距离
    k_actual = min(k, N)
    distances, indices = torch.topk(dists, k_actual, dim=-1, largest=False, sorted=True)
    
    # 如果 k_actual < k，用最后一个索引填充
    if k_actual < k:
        last_indices = indices[..., -1:].expand(B, G, k - k_actual)
        last_distances = distances[..., -1:].expand(B, G, k - k_actual)
        indices = torch.cat([indices, last_indices], dim=-1)
        distances = torch.cat([distances, last_distances], dim=-1)
    
    return indices, distances


def ball_query(
    query_points: torch.Tensor,
    key_points: torch.Tensor,
    radius: float,
    max_samples: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    球邻域查询
    
    Args:
        query_points: (B, G, 3) - 查询点（抓取平移）
        key_points: (B, N, 3) - 候选点（场景点云）
        radius: 球半径
        max_samples: 每个查询点最多返回的邻居数量
        mask: (B, N) or None - 场景掩码，1=valid, 0=padding
        
    Returns:
        indices: (B, G, max_samples) - 每个查询点的邻居索引
        within_mask: (B, G, max_samples) - 邻居有效性掩码，1=valid, 0=padding
    """
    B, G, _ = query_points.shape
    N = key_points.shape[1]
    
    # 计算距离：(B, G, N)
    dists = torch.cdist(query_points, key_points, p=2)  # (B, G, N)
    
    # 找出半径内的点
    within_radius = (dists <= radius).float()  # (B, G, N)
    
    # 如果有 mask，过滤掉 padding 位置
    if mask is not None:
        mask_expanded = mask.unsqueeze(1)  # (B, 1, N)
        within_radius = within_radius * mask_expanded
    
    # 对于每个查询点，取前 max_samples 个邻居
    # 先将不在半径内的距离设为无穷大
    dists_masked = dists.clone()
    dists_masked = dists_masked.masked_fill(within_radius == 0, float('inf'))
    
    # 取 top-k 最小距离
    k_actual = min(max_samples, N)
    _, indices = torch.topk(dists_masked, k_actual, dim=-1, largest=False, sorted=True)
    
    # 生成 within_mask：检查选中的索引是否在半径内
    # 使用 gather 从 within_radius 中提取对应位置的值
    within_mask = torch.gather(within_radius, 2, indices)  # (B, G, k_actual)
    
    # 如果 k_actual < max_samples，用 0 填充
    if k_actual < max_samples:
        pad_indices = torch.zeros(B, G, max_samples - k_actual, dtype=indices.dtype, device=indices.device)
        pad_mask = torch.zeros(B, G, max_samples - k_actual, dtype=within_mask.dtype, device=within_mask.device)
        indices = torch.cat([indices, pad_indices], dim=-1)
        within_mask = torch.cat([within_mask, pad_mask], dim=-1)
    
    return indices, within_mask


class KNNSelector(nn.Module):
    """
    k-近邻选择器
    
    基于抓取平移位置，从场景点云中选择 k 个最近邻点。
    """
    
    def __init__(self, k: int = 32, stochastic: bool = False):
        """
        Args:
            k: 邻居数量
            stochastic: 是否在训练时加入随机扰动（暂未实现）
        """
        super().__init__()
        self.k = k
        self.stochastic = stochastic
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KNNSelector initialized: k={k}, stochastic={stochastic}")
    
    def forward(
        self,
        grasp_translations: torch.Tensor,
        scene_xyz: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grasp_translations: (B, G, 3) - 抓取平移
            scene_xyz: (B, N, 3) - 场景点云坐标
            scene_mask: (B, N) or None - 场景掩码，1=valid, 0=padding
            
        Returns:
            indices: (B, G, k) - 每个抓取的局部邻居索引
            local_mask: (B, G, k) - 局部邻居掩码，1=valid, 0=invalid
        """
        indices, distances = knn_query(
            query_points=grasp_translations,
            key_points=scene_xyz,
            k=self.k,
            mask=scene_mask,
        )
        
        # 生成局部掩码：如果距离是有限的，则为有效
        local_mask = torch.isfinite(distances).float()
        
        return indices, local_mask


class BallQuerySelector(nn.Module):
    """
    球邻域选择器
    
    基于抓取平移位置，从场景点云中选择半径内的所有点（最多 max_samples 个）。
    """
    
    def __init__(self, radius: float = 0.05, max_samples: int = 32):
        """
        Args:
            radius: 球半径
            max_samples: 每个查询点最多返回的邻居数量
        """
        super().__init__()
        self.radius = radius
        self.max_samples = max_samples
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"BallQuerySelector initialized: radius={radius}, max_samples={max_samples}"
        )
    
    def forward(
        self,
        grasp_translations: torch.Tensor,
        scene_xyz: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grasp_translations: (B, G, 3) - 抓取平移
            scene_xyz: (B, N, 3) - 场景点云坐标
            scene_mask: (B, N) or None - 场景掩码，1=valid, 0=padding
            
        Returns:
            indices: (B, G, max_samples) - 每个抓取的局部邻居索引
            local_mask: (B, G, max_samples) - 局部邻居掩码，1=valid, 0=padding
        """
        indices, within_mask = ball_query(
            query_points=grasp_translations,
            key_points=scene_xyz,
            radius=self.radius,
            max_samples=self.max_samples,
            mask=scene_mask,
        )
        
        return indices, within_mask


def build_local_selector(selector_type: str, **kwargs) -> nn.Module:
    """
    工厂函数：根据类型构建局部选择器
    
    Args:
        selector_type: 'knn' | 'ball' | 'deformable'
        **kwargs: 传递给选择器的参数
        
    Returns:
        LocalSelector 实例
    """
    if selector_type == 'knn':
        return KNNSelector(
            k=kwargs.get('k', 32),
            stochastic=kwargs.get('stochastic', False),
        )
    elif selector_type == 'ball':
        return BallQuerySelector(
            radius=kwargs.get('radius', 0.05),
            max_samples=kwargs.get('k', 32),  # 使用 k 作为 max_samples
        )
    elif selector_type == 'deformable':
        # 占位：后续实现 Deformable3DAttn
        raise NotImplementedError(
            "Deformable 3D attention is not yet implemented. "
            "Please use 'knn' or 'ball' selector."
        )
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")

