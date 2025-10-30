"""
测试和对比不同的 Scene Token 提取策略

针对抓取生成任务的特点，提出以下改进方案：

现有方案回顾：
① last_layer: 直接使用最后一层稀疏特征 + FPS
② fps: 最远点采样，保证空间均匀分布
③ grid: 规则网格聚合
④ learned: 可学习的cross-attention tokenizer
⑤ multiscale: 多尺度特征融合

新提案：
⑥ surface_aware: 表面感知采样（重点关注高曲率区域）
⑦ hybrid: 混合策略（grid全局 + learned局部细节）
⑧ graspability_guided: 可抓取性引导（预测抓取热力图）
⑨ hierarchical_attention: 层次化注意力池化
⑩ adaptive_density: 自适应密度采样（重要区域多采样）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 方案⑥：表面感知采样 ====================

class SurfaceAwareTokenizer(nn.Module):
    """
    方案⑥：表面感知采样策略
    
    核心思想：
    - 计算局部几何特征（法线、曲率）
    - 优先采样高曲率区域（边缘、角落）- 这些通常是抓取的关键区域
    - 结合FPS保证空间覆盖
    
    优势：
    - 针对抓取任务：边缘和角落是关键抓取点
    - 保留几何细节
    - 不需要额外训练
    """
    
    def __init__(self, target_num_tokens: int, k_neighbors: int = 16):
        super().__init__()
        self.target_num_tokens = target_num_tokens
        self.k_neighbors = k_neighbors
        self.high_curvature_ratio = 0.6  # 60%的tokens来自高曲率区域
        
    def compute_local_curvature(
        self, 
        xyz: torch.Tensor,  # (B, N, 3)
        features: torch.Tensor,  # (B, C, N)
        k: int = 16
    ) -> torch.Tensor:
        """
        计算局部曲率估计
        
        使用PCA方法：在k近邻上做PCA，最小特征值代表曲率
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        # 简化版：使用距离方差作为曲率代理
        # 对每个点，计算到k个最近邻的距离方差
        curvature = torch.zeros(B, N, device=device)
        
        for b in range(B):
            pts = xyz[b]  # (N, 3)
            
            # 计算成对距离 (N, N)
            dist_matrix = torch.cdist(pts, pts)
            
            # 获取k最近邻
            knn_dists, knn_indices = torch.topk(
                dist_matrix, k + 1, largest=False, dim=1
            )  # +1因为包括自己
            
            knn_dists = knn_dists[:, 1:]  # 排除自己
            
            # 距离方差作为曲率指标
            curvature[b] = knn_dists.var(dim=1)
        
        return curvature
    
    def forward(
        self,
        xyz: torch.Tensor,  # (B, K_in, 3)
        features: torch.Tensor  # (B, C, K_in)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, K_in, 3) 输入坐标
            features: (B, C, K_in) 输入特征
        
        Returns:
            xyz_tokens: (B, K_out, 3)
            feat_tokens: (B, C, K_out)
        """
        from models.backbone.pointnet2_utils import farthest_point_sample
        
        B, K_in, _ = xyz.shape
        C = features.shape[1]
        device = xyz.device
        
        # 计算曲率
        curvature = self.compute_local_curvature(xyz, features, self.k_neighbors)
        
        # 按曲率分组
        num_high_curv = int(self.target_num_tokens * self.high_curvature_ratio)
        num_uniform = self.target_num_tokens - num_high_curv
        
        xyz_out = torch.zeros(B, self.target_num_tokens, 3, device=device)
        feat_out = torch.zeros(B, C, self.target_num_tokens, device=device)
        
        for b in range(B):
            valid_mask = (xyz[b].abs().sum(dim=-1) > 0)
            n_valid = valid_mask.sum().item()
            
            if n_valid <= self.target_num_tokens:
                # 不够，直接填充
                xyz_out[b, :n_valid] = xyz[b, valid_mask]
                feat_out[b, :, :n_valid] = features[b, :, valid_mask]
                continue
            
            # 高曲率采样
            curv_valid = curvature[b, valid_mask]
            _, high_curv_local_idx = torch.topk(
                curv_valid, min(num_high_curv, n_valid), largest=True
            )
            valid_indices = torch.where(valid_mask)[0]
            high_curv_idx = valid_indices[high_curv_local_idx]
            
            # 从剩余点中FPS采样
            remaining_mask = valid_mask.clone()
            remaining_mask[high_curv_idx] = False
            n_remaining = remaining_mask.sum().item()
            
            if n_remaining > num_uniform:
                xyz_remaining = xyz[b, remaining_mask].unsqueeze(0)
                fps_local_idx = farthest_point_sample(xyz_remaining, num_uniform)[0]
                remaining_indices = torch.where(remaining_mask)[0]
                uniform_idx = remaining_indices[fps_local_idx]
            else:
                uniform_idx = torch.where(remaining_mask)[0]
            
            # 合并
            selected_idx = torch.cat([high_curv_idx, uniform_idx])[:self.target_num_tokens]
            
            xyz_out[b, :len(selected_idx)] = xyz[b, selected_idx]
            feat_out[b, :, :len(selected_idx)] = features[b, :, selected_idx]
        
        logger.info(f"[SurfaceAware] Sampled {num_high_curv} high-curvature + {num_uniform} uniform tokens")
        return xyz_out, feat_out


# ==================== 方案⑦：混合策略 ====================

class HybridTokenizer(nn.Module):
    """
    方案⑦：混合策略
    
    核心思想：
    - Grid tokens: 提供全局空间结构（均匀覆盖）
    - Learned tokens: 提供局部细节和任务相关特征
    
    优势：
    - 全局+局部的完整表示
    - 学习到任务特定的重要区域
    - 平衡结构化和灵活性
    """
    
    def __init__(
        self,
        target_num_tokens: int,
        token_dim: int,
        grid_ratio: float = 0.5,  # 50%来自grid，50%来自learned
    ):
        super().__init__()
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        
        self.num_grid_tokens = int(target_num_tokens * grid_ratio)
        self.num_learned_tokens = target_num_tokens - self.num_grid_tokens
        
        # Grid tokenizer
        grid_size_per_dim = int(self.num_grid_tokens ** (1/3)) + 1
        self.grid_resolution = (grid_size_per_dim,) * 3
        
        # Learned tokenizer (cross-attention)
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_learned_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(token_dim)
        
        logger.info(
            f"[Hybrid] Grid tokens: {self.num_grid_tokens}, "
            f"Learned tokens: {self.num_learned_tokens}"
        )
    
    def grid_aggregate(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用规则网格聚合"""
        B, K, _ = xyz.shape
        C = features.shape[1]
        device = xyz.device
        
        gx, gy, gz = self.grid_resolution
        grid_total = gx * gy * gz
        
        # 归一化坐标
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_range = xyz_max - xyz_min + 1e-6
        xyz_norm = (xyz - xyz_min) / xyz_range
        
        # 计算网格索引
        grid_idx_x = (xyz_norm[..., 0] * gx).clamp(0, gx - 1).long()
        grid_idx_y = (xyz_norm[..., 1] * gy).clamp(0, gy - 1).long()
        grid_idx_z = (xyz_norm[..., 2] * gz).clamp(0, gz - 1).long()
        grid_idx = grid_idx_x * (gy * gz) + grid_idx_y * gz + grid_idx_z
        
        valid_mask = (xyz.abs().sum(dim=-1) > 0)
        
        # 聚合
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand_as(grid_idx)
        linear_idx = batch_ids * grid_total + grid_idx
        linear_idx_valid = linear_idx[valid_mask]
        
        agg_xyz = torch.zeros(B * grid_total, 3, device=device, dtype=xyz.dtype)
        agg_xyz.index_add_(0, linear_idx_valid, xyz[valid_mask])
        
        agg_feat = torch.zeros(B * grid_total, C, device=device, dtype=features.dtype)
        feat_bkc = features.permute(0, 2, 1)
        agg_feat.index_add_(0, linear_idx_valid, feat_bkc[valid_mask])
        
        agg_cnt = torch.zeros(B * grid_total, 1, device=device, dtype=torch.long)
        ones = torch.ones(linear_idx_valid.shape[0], 1, device=device, dtype=torch.long)
        agg_cnt.index_add_(0, linear_idx_valid, ones)
        
        # 还原形状并平均
        grid_xyz = agg_xyz.view(B, grid_total, 3)
        grid_feat = agg_feat.view(B, grid_total, C).permute(0, 2, 1)
        grid_count = agg_cnt.view(B, grid_total).clamp_min(1).float()
        
        grid_xyz = grid_xyz / grid_count.unsqueeze(-1)
        grid_feat = grid_feat / grid_count.unsqueeze(1)
        
        return grid_xyz[:, :self.num_grid_tokens], grid_feat[:, :, :self.num_grid_tokens]
    
    def learned_aggregate(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用可学习的cross-attention"""
        from models.backbone.pointnet2_utils import farthest_point_sample
        
        B = xyz.shape[0]
        
        # feat: (B, C, K) -> (B, K, C)
        feat_transposed = features.permute(0, 2, 1)
        
        # Cross-attention
        queries = self.query_tokens.expand(B, -1, -1)
        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=feat_transposed,
            value=feat_transposed,
            need_weights=True,
            average_attn_weights=True,
        )
        
        token_feat = self.norm(queries + attn_out)  # (B, num_learned, C)
        token_feat = token_feat.permute(0, 2, 1)  # (B, C, num_learned)
        
        # 坐标：使用attention权重加权
        # attn_weights: (B, num_learned, K)
        token_xyz = torch.bmm(attn_weights, xyz)  # (B, num_learned, 3)
        
        return token_xyz, token_feat
    
    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, C, target_num_tokens)
        """
        # Grid部分
        grid_xyz, grid_feat = self.grid_aggregate(xyz, features)
        
        # Learned部分
        learned_xyz, learned_feat = self.learned_aggregate(xyz, features)
        
        # 拼接
        final_xyz = torch.cat([grid_xyz, learned_xyz], dim=1)
        final_feat = torch.cat([grid_feat, learned_feat], dim=2)
        
        return final_xyz, final_feat


# ==================== 方案⑧：可抓取性引导 ====================

class GraspabilityGuidedTokenizer(nn.Module):
    """
    方案⑧：可抓取性引导的Token选择
    
    核心思想：
    - 预测每个点的"可抓取性"分数
    - 优先采样高可抓取性区域
    - 类似于saliency-guided attention
    
    优势：
    - 直接针对任务（抓取生成）
    - 端到端可学习
    - 自动聚焦重要区域
    
    注意：需要训练数据来学习抓取性预测器
    """
    
    def __init__(
        self,
        target_num_tokens: int,
        token_dim: int,
        feature_dim: int,
    ):
        super().__init__()
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        
        # 抓取性预测器（简单MLP）
        self.graspability_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的分数
        )
        
        # Token聚合器
        self.token_aggregator = nn.Sequential(
            nn.Linear(feature_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        
        logger.info(f"[GraspabilityGuided] Will select {target_num_tokens} tokens")
    
    def forward(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        features: torch.Tensor  # (B, C, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, token_dim, target_num_tokens)
        """
        B, K_in = xyz.shape[:2]
        C = features.shape[1]
        device = xyz.device
        
        # 转换特征: (B, C, K) -> (B, K, C)
        feat_bkc = features.permute(0, 2, 1)
        
        # 预测抓取性
        graspability = self.graspability_head(feat_bkc)  # (B, K, 1)
        graspability = graspability.squeeze(-1)  # (B, K)
        
        # 按抓取性采样
        xyz_out = torch.zeros(B, self.target_num_tokens, 3, device=device)
        feat_out = torch.zeros(B, self.token_dim, self.target_num_tokens, device=device)
        
        for b in range(B):
            valid_mask = (xyz[b].abs().sum(dim=-1) > 0)
            n_valid = valid_mask.sum().item()
            
            if n_valid <= self.target_num_tokens:
                # 不够，直接填充
                selected_feat = self.token_aggregator(feat_bkc[b, valid_mask])
                xyz_out[b, :n_valid] = xyz[b, valid_mask]
                feat_out[b, :, :n_valid] = selected_feat.t()
                continue
            
            # Top-K采样（基于抓取性）
            grasp_scores_valid = graspability[b, valid_mask]
            _, top_k_local_idx = torch.topk(
                grasp_scores_valid,
                self.target_num_tokens,
                largest=True
            )
            
            valid_indices = torch.where(valid_mask)[0]
            selected_idx = valid_indices[top_k_local_idx]
            
            # 提取并聚合特征
            selected_feat = self.token_aggregator(feat_bkc[b, selected_idx])  # (K_out, token_dim)
            
            xyz_out[b] = xyz[b, selected_idx]
            feat_out[b] = selected_feat.t()
        
        logger.debug(f"[GraspabilityGuided] Avg graspability: {graspability.mean().item():.4f}")
        return xyz_out, feat_out


# ==================== 方案⑨：层次化注意力池化 ====================

class HierarchicalAttentionTokenizer(nn.Module):
    """
    方案⑨：层次化注意力池化
    
    核心思想：
    - 多层逐步下采样，类似于Set Transformer
    - 每层使用self-attention + pooling
    - 保留层次结构信息
    
    优势：
    - 渐进式抽象
    - 保留多尺度信息
    - 完全可学习
    
    参考：Set Transformer (ICML 2019)
    """
    
    def __init__(
        self,
        target_num_tokens: int,
        token_dim: int,
        num_levels: int = 3,
    ):
        super().__init__()
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        self.num_levels = num_levels
        
        # 计算每层的token数量（逐步减少）
        self.tokens_per_level = []
        current_tokens = target_num_tokens * (4 ** (num_levels - 1))
        for _ in range(num_levels):
            self.tokens_per_level.append(current_tokens)
            current_tokens = current_tokens // 4
        self.tokens_per_level.reverse()  # 从少到多
        
        # 为每层创建pooling attention
        self.pooling_layers = nn.ModuleList()
        for i in range(num_levels):
            n_queries = self.tokens_per_level[i]
            self.pooling_layers.append(
                PoolingByMultiheadAttention(
                    embed_dim=token_dim,
                    num_queries=n_queries,
                    num_heads=8,
                )
            )
        
        logger.info(
            f"[HierarchicalAttention] Levels: {num_levels}, "
            f"Tokens per level: {self.tokens_per_level}"
        )
    
    def forward(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        features: torch.Tensor  # (B, C, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, C, target_num_tokens)
        """
        from models.backbone.pointnet2_utils import farthest_point_sample
        
        # feat: (B, C, K) -> (B, K, C)
        x = features.permute(0, 2, 1)
        
        # 逐层池化
        for i, pool_layer in enumerate(self.pooling_layers):
            x, attn_weights = pool_layer(x)  # (B, K_i, C), (B, K_i, K_prev)
        
        # x现在是(B, target_num_tokens, C)
        final_feat = x.permute(0, 2, 1)  # (B, C, target_num_tokens)
        
        # 坐标：使用最后一层的attention权重加权
        final_xyz = torch.bmm(attn_weights, xyz)  # (B, target_num_tokens, 3)
        
        return final_xyz, final_feat


class PoolingByMultiheadAttention(nn.Module):
    """使用Multi-head Attention进行池化"""
    
    def __init__(self, embed_dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.num_queries = num_queries
        
        # 可学习的query embeddings
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, K_in, C)
        
        Returns:
            pooled: (B, num_queries, C)
            attn_weights: (B, num_queries, K_in)
        """
        B = x.shape[0]
        
        # 扩展queries
        queries = self.queries.expand(B, -1, -1)
        
        # Cross-attention pooling
        attn_out, attn_weights = self.attn(
            query=queries,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=True,
        )
        
        # Residual + Norm
        pooled = self.norm(queries + attn_out)
        
        # FFN
        pooled = self.norm2(pooled + self.ffn(pooled))
        
        return pooled, attn_weights


# ==================== 方案⑩：自适应密度采样 ====================

class AdaptiveDensityTokenizer(nn.Module):
    """
    方案⑩：自适应密度采样
    
    核心思想：
    - 预测每个区域需要多少tokens（重要性密度）
    - 重要区域分配更多tokens
    - 类似于adaptive mesh refinement
    
    优势：
    - 动态资源分配
    - 细节和效率平衡
    - 适应不同复杂度的场景
    """
    
    def __init__(
        self,
        target_num_tokens: int,
        token_dim: int,
        feature_dim: int,
        num_regions: int = 8,  # 将空间分为8个区域
    ):
        super().__init__()
        self.target_num_tokens = target_num_tokens
        self.token_dim = token_dim
        self.num_regions = num_regions
        
        # 区域重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softplus()  # 输出正数
        )
        
        # Token aggregator
        self.token_aggregator = nn.Linear(feature_dim, token_dim)
        
        logger.info(
            f"[AdaptiveDensity] Target tokens: {target_num_tokens}, "
            f"Regions: {num_regions}"
        )
    
    def forward(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        features: torch.Tensor  # (B, C, K)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, token_dim, target_num_tokens)
        """
        from models.backbone.pointnet2_utils import farthest_point_sample
        
        B, K_in, _ = xyz.shape
        device = xyz.device
        
        # 转换特征
        feat_bkc = features.permute(0, 2, 1)  # (B, K, C)
        
        # 预测每个点的重要性
        importance = self.importance_predictor(feat_bkc).squeeze(-1)  # (B, K)
        
        # 空间分区（简单的octree式分割）
        xyz_min = xyz.min(dim=1, keepdim=True)[0]
        xyz_max = xyz.max(dim=1, keepdim=True)[0]
        xyz_range = xyz_max - xyz_min + 1e-6
        xyz_norm = (xyz - xyz_min) / xyz_range  # [0, 1]
        
        # 计算区域索引（3D网格）
        regions_per_dim = int(self.num_regions ** (1/3)) + 1
        region_idx_x = (xyz_norm[..., 0] * regions_per_dim).clamp(0, regions_per_dim - 1).long()
        region_idx_y = (xyz_norm[..., 1] * regions_per_dim).clamp(0, regions_per_dim - 1).long()
        region_idx_z = (xyz_norm[..., 2] * regions_per_dim).clamp(0, regions_per_dim - 1).long()
        region_idx = (
            region_idx_x * (regions_per_dim ** 2) +
            region_idx_y * regions_per_dim +
            region_idx_z
        )  # (B, K)
        
        # 按batch处理
        xyz_out = torch.zeros(B, self.target_num_tokens, 3, device=device)
        feat_out = torch.zeros(B, self.token_dim, self.target_num_tokens, device=device)
        
        for b in range(B):
            valid_mask = (xyz[b].abs().sum(dim=-1) > 0)
            
            # 计算每个区域的总重要性
            region_importance = torch.zeros(
                regions_per_dim ** 3, device=device
            )
            for r in range(regions_per_dim ** 3):
                region_mask = (region_idx[b] == r) & valid_mask
                if region_mask.any():
                    region_importance[r] = importance[b, region_mask].sum()
            
            # 按重要性分配tokens
            total_importance = region_importance.sum() + 1e-8
            tokens_per_region = (
                region_importance / total_importance * self.target_num_tokens
            ).round().long()
            
            # 从每个区域采样
            selected_indices = []
            for r in range(regions_per_dim ** 3):
                n_tokens_r = tokens_per_region[r].item()
                if n_tokens_r == 0:
                    continue
                
                region_mask = (region_idx[b] == r) & valid_mask
                region_pts_idx = torch.where(region_mask)[0]
                
                if len(region_pts_idx) == 0:
                    continue
                elif len(region_pts_idx) <= n_tokens_r:
                    selected_indices.append(region_pts_idx)
                else:
                    # 在区域内FPS采样
                    xyz_region = xyz[b, region_pts_idx].unsqueeze(0)
                    fps_local = farthest_point_sample(xyz_region, n_tokens_r)[0]
                    selected_indices.append(region_pts_idx[fps_local])
            
            if len(selected_indices) > 0:
                selected_idx = torch.cat(selected_indices)[:self.target_num_tokens]
                
                # 聚合特征
                selected_feat = self.token_aggregator(feat_bkc[b, selected_idx])
                
                xyz_out[b, :len(selected_idx)] = xyz[b, selected_idx]
                feat_out[b, :, :len(selected_idx)] = selected_feat.t()
        
        logger.debug(
            f"[AdaptiveDensity] Region importance range: "
            f"[{region_importance.min().item():.2f}, {region_importance.max().item():.2f}]"
        )
        return xyz_out, feat_out


# ==================== 测试代码 ====================

def test_tokenizers():
    """测试所有tokenizer策略"""
    
    # 模拟输入
    B, N, C = 2, 512, 128  # 批次=2，点数=512，特征维=128
    target_K = 128  # 目标token数
    
    xyz = torch.randn(B, N, 3)
    features = torch.randn(B, C, N)
    
    print("=" * 80)
    print("测试场景token提取策略")
    print("=" * 80)
    print(f"输入: xyz={xyz.shape}, features={features.shape}")
    print(f"目标tokens: {target_K}")
    print()
    
    # 测试各个方案
    tokenizers = {
        "⑥ SurfaceAware": SurfaceAwareTokenizer(target_K),
        "⑦ Hybrid": HybridTokenizer(target_K, C),
        "⑧ GraspabilityGuided": GraspabilityGuidedTokenizer(target_K, C, C),
        "⑨ HierarchicalAttention": HierarchicalAttentionTokenizer(target_K, C, num_levels=3),
        "⑩ AdaptiveDensity": AdaptiveDensityTokenizer(target_K, C, C),
    }
    
    for name, tokenizer in tokenizers.items():
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print(f"{'='*60}")
        
        try:
            xyz_out, feat_out = tokenizer(xyz, features)
            print(f"✓ 输出: xyz={xyz_out.shape}, features={feat_out.shape}")
            
            # 简单的质量检查
            assert xyz_out.shape == (B, target_K, 3), f"xyz shape错误"
            assert feat_out.shape[0] == B and feat_out.shape[2] == target_K, f"feat shape错误"
            
            print(f"✓ 形状验证通过")
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()


def compare_strategies_for_grasping():
    """
    比较不同策略对抓取任务的适用性
    """
    print("\n" + "=" * 80)
    print("抓取任务适用性分析")
    print("=" * 80)
    
    strategies = {
        "① last_layer (现有)": {
            "优点": "简单快速，直接使用PTv3的输出",
            "缺点": "可能丢失关键几何细节",
            "适用性": "⭐⭐⭐",
        },
        "② fps (现有)": {
            "优点": "空间分布均匀，覆盖全面",
            "缺点": "不考虑任务相关性",
            "适用性": "⭐⭐⭐",
        },
        "③ grid (现有)": {
            "优点": "结构化，易于理解",
            "缺点": "固定分辨率，不适应场景复杂度",
            "适用性": "⭐⭐",
        },
        "④ learned (现有)": {
            "优点": "端到端学习，灵活",
            "缺点": "需要大量训练，可能过拟合",
            "适用性": "⭐⭐⭐⭐",
        },
        "⑤ multiscale (现有)": {
            "优点": "多尺度信息丰富",
            "缺点": "计算成本高",
            "适用性": "⭐⭐⭐⭐",
        },
        "⑥ surface_aware (新)": {
            "优点": "关注抓取关键区域（边缘/角落），几何感知",
            "缺点": "需要调整曲率阈值",
            "适用性": "⭐⭐⭐⭐⭐",
            "推荐": "✓",
        },
        "⑦ hybrid (新)": {
            "优点": "全局+局部平衡，结合结构化和学习",
            "缺点": "参数较多",
            "适用性": "⭐⭐⭐⭐⭐",
            "推荐": "✓",
        },
        "⑧ graspability_guided (新)": {
            "优点": "直接针对抓取任务，端到端",
            "缺点": "需要标注数据训练",
            "适用性": "⭐⭐⭐⭐⭐",
            "推荐": "✓ (如果有训练数据)",
        },
        "⑨ hierarchical_attention (新)": {
            "优点": "层次化表示，类似Transformer",
            "缺点": "计算复杂，慢",
            "适用性": "⭐⭐⭐⭐",
        },
        "⑩ adaptive_density (新)": {
            "优点": "动态分配资源，适应复杂度",
            "缺点": "实现复杂",
            "适用性": "⭐⭐⭐⭐",
        },
    }
    
    for name, info in strategies.items():
        print(f"\n{name}")
        print(f"  优点: {info['优点']}")
        print(f"  缺点: {info['缺点']}")
        print(f"  适用性: {info['适用性']}")
        if "推荐" in info:
            print(f"  推荐: {info['推荐']}")


def my_recommendations():
    """给出具体建议"""
    print("\n" + "=" * 80)
    print("针对你的抓取生成任务的具体建议")
    print("=" * 80)
    
    print("""
### 1. 最推荐的策略组合

**首选: 方案⑥ (Surface-Aware) + 方案⑦ (Hybrid)**

理由：
- 抓取任务的关键在于识别可抓取的几何特征（边缘、角落、曲面）
- Surface-Aware能自动关注这些区域
- Hybrid策略提供了全局结构（grid）和任务相关细节（learned）的平衡

实现建议：
```python
# 在PTv3SparseEncoder中添加
self.token_strategy = 'surface_hybrid'  # 新策略

# forward中
if self.token_strategy == 'surface_hybrid':
    # Step 1: Surface-aware预筛选，选出2K个候选点
    surface_xyz, surface_feat = surface_aware_sample(
        xyz_sparse, feat_sparse, target=self.target_num_tokens * 2
    )
    # Step 2: 在候选点上应用hybrid策略
    final_xyz, final_feat = hybrid_tokenize(
        surface_xyz, surface_feat, target=self.target_num_tokens
    )
```

### 2. 如果有训练数据：方案⑧ (Graspability-Guided)

如果你有抓取成功/失败的标注数据，可以训练一个抓取性预测器：
- 监督信号：成功抓取点附近的点云区域 → 高分
- 失败抓取点附近的点云区域 → 低分
- 端到端优化token选择

### 3. 改进现有multiscale策略

你的方案⑤已经很好，可以进一步优化：
- 不同尺度的tokens分配权重（coarse层少，fine层多）
- 添加跨尺度的attention融合
- 使用FPN-like的特征融合

### 4. 计算效率考虑

按计算成本排序（从低到高）：
① last_layer/fps < ③ grid < ⑥ surface < ⑦ hybrid < ⑤ multiscale < 
  ⑨ hierarchical < ⑧/⑩ (需要训练)

如果推理速度重要，选⑥；如果追求最佳效果，选⑦或⑧。

### 5. 与Double Stream Block的配合

考虑到你的架构：
```
Double Stream Block:
  Scene Tokens (K=128) <--attention--> Grasp Tokens (M个抓取)
```

Scene tokens应该：
- 覆盖全局空间（避免grasp token找不到对应区域）
- 突出可抓取区域（提高attention效率）
- 包含多尺度信息（支持粗细不同的抓取规划）

→ 这正是方案⑦ (Hybrid)的设计目标！

### 6. 消融实验建议

建议测试顺序：
1. Baseline: 方案① (last_layer + FPS)
2. 改进几何: 方案⑥ (surface_aware)
3. 混合策略: 方案⑦ (hybrid)
4. 如果有数据: 方案⑧ (graspability_guided)
5. 多尺度增强: 改进方案⑤ (multiscale with FPN)

评估指标：
- 抓取成功率（主要指标）
- Token覆盖率（scene tokens覆盖多少%的空间）
- Attention分布熵（tokens是否有效）
- 推理速度

### 7. 代码实现优先级

如果只选一个实现：**方案⑥ (Surface-Aware)**
- 简单、快速、不需要训练
- 直接针对抓取任务的几何特性
- 可作为其他方案的预处理步骤

如果选两个：**⑥ + ⑦**
- 组合使用效果最佳
- ⑥做粗筛，⑦做精选

如果有充足时间：**全部实现并做消融**
- 每个方案都有其优势
- 不同场景可能需要不同策略
- 可以做成config可选的模块
""")


if __name__ == "__main__":
    print("SceneLeapUltra - Scene Token策略测试\n")
    
    # 运行测试
    test_tokenizers()
    
    # 策略比较
    compare_strategies_for_grasping()
    
    # 我的建议
    my_recommendations()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

