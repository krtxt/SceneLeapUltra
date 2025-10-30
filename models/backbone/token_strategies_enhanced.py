"""
增强的 Scene Token 提取策略

为 PTv3SparseEncoder 提供额外的token选择策略，特别针对抓取生成任务优化。

新增策略：
- surface_aware: 表面感知采样（关注高曲率区域）
- hybrid: 混合策略（grid全局 + learned局部）
- surface_hybrid: 组合策略（surface预筛 + hybrid精选）

使用方法：
1. 将这些方法添加到 PTv3SparseEncoder
2. 在config中设置 token_strategy='surface_aware' 等
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging


class TokenStrategyMixin:
    """
    用于扩展PTv3SparseEncoder的Mixin类
    
    使用方法：
        class PTv3SparseEncoder(TokenStrategyMixin, nn.Module):
            ...
    """
    
    def _strategy_surface_aware(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        orig_coords: Optional[torch.Tensor] = None  # (B, N, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案⑥：表面感知采样
        
        核心思想：
        - 计算局部曲率（使用k近邻距离方差）
        - 60% tokens来自高曲率区域（边缘、角落）
        - 40% tokens使用FPS保证空间覆盖
        
        Args:
            xyz: (B, K, 3) 输入点坐标
            feat: (B, C, K) 输入特征
            orig_coords: (B, N, 3) 原始坐标（未使用）
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, C, target_num_tokens)
        """
        from .pointnet2_utils import farthest_point_sample
        
        B, K_in, _ = xyz.shape
        C = feat.shape[1]
        device = xyz.device
        
        # 参数配置
        k_neighbors = getattr(self, 'surface_k_neighbors', 16)
        high_curv_ratio = getattr(self, 'high_curvature_ratio', 0.6)
        
        num_high_curv = int(self.target_num_tokens * high_curv_ratio)
        num_uniform = self.target_num_tokens - num_high_curv
        
        # 计算曲率
        curvature = self._compute_local_curvature(xyz, k=k_neighbors)
        
        # 初始化输出
        xyz_out = torch.zeros(B, self.target_num_tokens, 3, device=device, dtype=xyz.dtype)
        feat_out = torch.zeros(B, C, self.target_num_tokens, device=device, dtype=feat.dtype)
        
        # 按batch处理
        for b in range(B):
            valid_mask = (xyz[b].abs().sum(dim=-1) > 0)
            n_valid = valid_mask.sum().item()
            
            if n_valid <= self.target_num_tokens:
                # 点数不够，直接填充
                xyz_out[b, :n_valid] = xyz[b, valid_mask]
                feat_out[b, :, :n_valid] = feat[b, :, valid_mask]
                continue
            
            # 高曲率采样
            curv_valid = curvature[b, valid_mask]
            n_high_curv = min(num_high_curv, n_valid)
            valid_indices = torch.where(valid_mask)[0]

            if n_high_curv > 0:
                _, high_curv_local_idx = torch.topk(curv_valid, n_high_curv, largest=True)
                high_curv_idx = valid_indices[high_curv_local_idx]
            else:
                high_curv_idx = torch.empty(0, dtype=torch.long, device=device)
            
            # 从剩余点中FPS采样
            remaining_mask = valid_mask.clone()
            remaining_mask[high_curv_idx] = False
            n_remaining = remaining_mask.sum().item()

            if num_uniform > 0 and n_remaining > 0:
                if n_remaining > num_uniform:
                    xyz_remaining = xyz[b, remaining_mask].unsqueeze(0)
                    fps_local_idx = farthest_point_sample(xyz_remaining, num_uniform)[0]
                    remaining_indices = torch.where(remaining_mask)[0]
                    uniform_idx = remaining_indices[fps_local_idx]
                else:
                    uniform_idx = torch.where(remaining_mask)[0]
            else:
                uniform_idx = torch.empty(0, dtype=torch.long, device=device)
            
            # 合并索引
            selected_idx = torch.cat([high_curv_idx, uniform_idx])[:self.target_num_tokens]
            n_selected = len(selected_idx)
            
            xyz_out[b, :n_selected] = xyz[b, selected_idx]
            feat_out[b, :, :n_selected] = feat[b, :, selected_idx]
        
        self.logger.debug(
            f"[SurfaceAware] Sampled {num_high_curv} high-curvature + "
            f"{num_uniform} uniform tokens"
        )
        
        return xyz_out, feat_out
    
    def _strategy_hybrid(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        orig_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案⑦：混合策略（Grid + Learned）
        
        核心思想：
        - 50% tokens来自规则网格聚合（全局结构）
        - 50% tokens来自可学习的cross-attention（局部细节）
        
        Args:
            xyz: (B, K, 3)
            feat: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, C, target_num_tokens)
        """
        grid_ratio = getattr(self, 'hybrid_grid_ratio', 0.5)
        num_grid = int(self.target_num_tokens * grid_ratio)
        num_learned = self.target_num_tokens - num_grid
        
        if num_grid <= 0 and num_learned <= 0:
            raise ValueError("target_num_tokens 必须大于 0")
        
        if num_grid > 0:
            grid_xyz, grid_feat = self._grid_aggregate_for_hybrid(xyz, feat, num_grid)
        else:
            grid_xyz = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device, dtype=xyz.dtype)
            grid_feat = torch.zeros(feat.shape[0], feat.shape[1], 0, device=feat.device, dtype=feat.dtype)
        
        if num_learned > 0:
            learned_xyz, learned_feat = self._learned_aggregate_for_hybrid(xyz, feat, num_learned)
        else:
            learned_xyz = torch.zeros(xyz.shape[0], 0, 3, device=xyz.device, dtype=xyz.dtype)
            learned_feat = torch.zeros(feat.shape[0], feat.shape[1], 0, device=feat.device, dtype=feat.dtype)
        
        final_xyz = torch.cat([grid_xyz, learned_xyz], dim=1)
        final_feat = torch.cat([grid_feat, learned_feat], dim=2)
        
        self.logger.debug(
            f"[Hybrid] Grid tokens: {num_grid}, Learned tokens: {num_learned}"
        )
        
        return final_xyz, final_feat
    
    def _strategy_surface_hybrid(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        orig_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        方案⑥+⑦：组合策略
        
        两阶段流程：
        1. Surface-aware粗筛：选出2K个候选点（关注高曲率）
        2. Hybrid精选：从候选中选K个最终tokens（grid + learned）
        
        Args:
            xyz: (B, K, 3)
            feat: (B, C, K)
        
        Returns:
            xyz_tokens: (B, target_num_tokens, 3)
            feat_tokens: (B, C, target_num_tokens)
        """
        # Stage 1: Surface-aware粗筛（选2倍数量的候选）
        num_candidates = self.target_num_tokens * 2
        
        # 临时修改target以进行粗筛
        original_target = self.target_num_tokens
        self.target_num_tokens = num_candidates
        
        candidates_xyz, candidates_feat = self._strategy_surface_aware(xyz, feat, orig_coords)
        
        # 恢复target
        self.target_num_tokens = original_target
        
        # Stage 2: Hybrid精选
        final_xyz, final_feat = self._strategy_hybrid(candidates_xyz, candidates_feat)
        
        self.logger.debug(
            f"[SurfaceHybrid] Stage1: {num_candidates} candidates, "
            f"Stage2: {self.target_num_tokens} final tokens"
        )
        
        return final_xyz, final_feat
    
    # ==================== 辅助方法 ====================
    
    def _compute_local_curvature(
        self,
        xyz: torch.Tensor,  # (B, N, 3)
        k: int = 16
    ) -> torch.Tensor:
        """
        计算局部曲率估计
        
        使用k近邻距离方差作为曲率代理指标：
        - 平坦区域：距离方差小
        - 高曲率区域（边缘、角落）：距离方差大
        
        Args:
            xyz: (B, N, 3)
            k: 近邻数量
        
        Returns:
            curvature: (B, N) 曲率估计值
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        curvature = torch.zeros(B, N, device=device)
        
        for b in range(B):
            pts = xyz[b]
            valid_mask = (pts.abs().sum(dim=-1) > 0)
            valid_pts = pts[valid_mask]
            n_valid = valid_pts.shape[0]

            if n_valid <= 1:
                continue

            k_eff = min(k, n_valid - 1)
            if k_eff <= 0:
                continue

            dist_matrix = torch.cdist(valid_pts, valid_pts)
            knn_dists, _ = torch.topk(dist_matrix, k_eff + 1, largest=False, dim=1)
            knn_dists = knn_dists[:, 1:]

            curv_vals = knn_dists.var(dim=1)
            curvature[b, valid_mask] = curv_vals
        
        return curvature
    
    def _build_hybrid_tokenizer(self, device: Optional[torch.device] = None):
        """构建hybrid策略所需的可学习模块"""
        token_dim = self.output_dim
        
        grid_ratio = getattr(self, 'hybrid_grid_ratio', 0.5)
        num_grid = int(self.target_num_tokens * grid_ratio)
        num_learned = max(self.target_num_tokens - num_grid, 0)

        query_count = max(num_learned, 1)  # 至少保留一个，以避免0维Parameter
        self.hybrid_query_tokens = nn.Parameter(
            torch.randn(1, query_count, token_dim)
        )
        nn.init.trunc_normal_(self.hybrid_query_tokens, std=0.02)
        
        # Cross-attention
        self.hybrid_cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.hybrid_norm = nn.LayerNorm(token_dim)
        
        if device is not None:
            self.hybrid_query_tokens.data = self.hybrid_query_tokens.data.to(device)
            self.hybrid_cross_attn = self.hybrid_cross_attn.to(device)
            self.hybrid_norm = self.hybrid_norm.to(device)

        self._hybrid_initialized = True

        self.logger.info(f"Built hybrid tokenizer with {query_count} learned queries (target learned tokens={num_learned})")
    
    def _grid_aggregate_for_hybrid(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        num_grid_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用规则网格聚合特征（hybrid策略的grid部分）
        
        与_strategy_grid类似，但输出固定数量的tokens
        """
        B, K, _ = xyz.shape
        C = feat.shape[1]
        device = xyz.device
        
        # 计算网格分辨率
        grid_size_per_dim = int(num_grid_tokens ** (1/3)) + 1
        gx = gy = gz = grid_size_per_dim
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
        
        # 聚合坐标
        agg_xyz = torch.zeros(B * grid_total, 3, device=device, dtype=xyz.dtype)
        agg_xyz.index_add_(0, linear_idx_valid, xyz[valid_mask])
        
        # 聚合特征
        agg_feat = torch.zeros(B * grid_total, C, device=device, dtype=feat.dtype)
        feat_bkc = feat.permute(0, 2, 1)
        agg_feat.index_add_(0, linear_idx_valid, feat_bkc[valid_mask])
        
        # 计数
        agg_cnt = torch.zeros(B * grid_total, 1, device=device, dtype=torch.long)
        ones = torch.ones(linear_idx_valid.shape[0], 1, device=device, dtype=torch.long)
        agg_cnt.index_add_(0, linear_idx_valid, ones)
        
        # 还原并平均
        grid_xyz = agg_xyz.view(B, grid_total, 3)
        grid_feat = agg_feat.view(B, grid_total, C).permute(0, 2, 1)
        grid_count = agg_cnt.view(B, grid_total).clamp_min(1).float()
        
        grid_xyz = grid_xyz / grid_count.unsqueeze(-1)
        grid_feat = grid_feat / grid_count.unsqueeze(1)
        
        # 如果网格数不等于目标，使用FPS调整
        if grid_total != num_grid_tokens:
            grid_xyz, grid_feat = self._fps_sample(grid_xyz, grid_feat, num_grid_tokens)
        
        return grid_xyz[:, :num_grid_tokens], grid_feat[:, :, :num_grid_tokens]
    
    def _learned_aggregate_for_hybrid(
        self,
        xyz: torch.Tensor,  # (B, K, 3)
        feat: torch.Tensor,  # (B, C, K)
        num_learned_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用可学习的cross-attention聚合特征（hybrid策略的learned部分）
        """
        B = xyz.shape[0]
        device = xyz.device

        if num_learned_tokens <= 0:
            empty_xyz = torch.zeros(B, 0, 3, device=device, dtype=xyz.dtype)
            empty_feat = torch.zeros(B, feat.shape[1], 0, device=device, dtype=feat.dtype)
            return empty_xyz, empty_feat

        if not getattr(self, '_hybrid_initialized', False):
            self._build_hybrid_tokenizer(device=device)

        self.hybrid_query_tokens.data = self.hybrid_query_tokens.data.to(device)
        self.hybrid_cross_attn = self.hybrid_cross_attn.to(device)
        self.hybrid_norm = self.hybrid_norm.to(device)

        # 特征转换: (B, C, K) -> (B, K, C)
        feat_bkc = feat.permute(0, 2, 1)

        max_queries = self.hybrid_query_tokens.shape[1]
        if num_learned_tokens > max_queries:
            self._build_hybrid_tokenizer(device=device)
            self.hybrid_query_tokens.data = self.hybrid_query_tokens.data.to(device)
            self.hybrid_cross_attn = self.hybrid_cross_attn.to(device)
            self.hybrid_norm = self.hybrid_norm.to(device)
            max_queries = self.hybrid_query_tokens.shape[1]
            num_learned_tokens = min(num_learned_tokens, max_queries)
        
        # 确保query tokens的数量匹配
        queries = self.hybrid_query_tokens[:, :num_learned_tokens, :].expand(B, -1, -1)
        queries = queries.contiguous()
        
        # Cross-attention
        attn_out, attn_weights = self.hybrid_cross_attn(
            query=queries,
            key=feat_bkc,
            value=feat_bkc,
            need_weights=True,
            average_attn_weights=True,
        )
        
        # Residual + Norm
        token_feat = self.hybrid_norm(queries + attn_out)  # (B, num_learned, C)
        token_feat = token_feat.permute(0, 2, 1)  # (B, C, num_learned)
        
        # 坐标：使用attention权重加权平均
        # attn_weights: (B, num_learned, K)
        token_xyz = torch.bmm(attn_weights, xyz)  # (B, num_learned, 3)
        
        return token_xyz, token_feat
    
    def _init_enhanced_strategies(self):
        """
        初始化增强策略所需的参数和模块
        
        在PTv3SparseEncoder的__init__中调用
        """
        # Surface-aware参数
        self.surface_k_neighbors = getattr(self.cfg, 'surface_k_neighbors', 16)
        self.high_curvature_ratio = getattr(self.cfg, 'high_curvature_ratio', 0.6)
        
        # Hybrid参数
        self.hybrid_grid_ratio = getattr(self.cfg, 'hybrid_grid_ratio', 0.5)
        self._hybrid_initialized = False
        
        # 如果使用hybrid或surface_hybrid，构建相关模块
        if self.token_strategy in ['hybrid', 'surface_hybrid']:
            device = None
            if hasattr(self, 'hybrid_query_tokens') and self.hybrid_query_tokens is not None:
                device = self.hybrid_query_tokens.device
            self._build_hybrid_tokenizer(device=device)
        
        self.logger.info(
            f"Initialized enhanced token strategies: "
            f"surface_k={self.surface_k_neighbors}, "
            f"curv_ratio={self.high_curvature_ratio}, "
            f"grid_ratio={self.hybrid_grid_ratio}"
        )


# ==================== 使用示例 ====================

"""
在 ptv3_sparse_encoder.py 中集成：

1. 导入mixin:
   from .token_strategies_enhanced import TokenStrategyMixin

2. 修改类定义:
   class PTv3SparseEncoder(TokenStrategyMixin, nn.Module):
       ...

3. 在__init__中初始化:
   def __init__(self, cfg, ...):
       super().__init__()
       ...
       # 在最后添加
       if hasattr(self, '_init_enhanced_strategies'):
           self._init_enhanced_strategies()

4. 在forward中添加新策略:
   def forward(self, pos, ...):
       ...
       elif self.token_strategy == 'surface_aware':
           xyz_out, feat_out = self._strategy_surface_aware(xyz_sparse, feat_sparse, coords)
       elif self.token_strategy == 'hybrid':
           xyz_out, feat_out = self._strategy_hybrid(xyz_sparse, feat_sparse)
       elif self.token_strategy == 'surface_hybrid':
           xyz_out, feat_out = self._strategy_surface_hybrid(xyz_sparse, feat_sparse, coords)
       ...

5. 在config中使用:
   model:
     backbone:
       token_strategy: 'surface_aware'  # 或 'hybrid', 'surface_hybrid'
       surface_k_neighbors: 16
       high_curvature_ratio: 0.6
       hybrid_grid_ratio: 0.5
"""
