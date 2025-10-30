"""
Optimal Transport for Flow Matching

This module implements Sinkhorn optimal transport algorithm for computing
optimal pairings between grasp sets and noise samples in Flow Matching training.

理论背景：
- 标准Flow Matching使用随机索引配对：x0[i] <-> x1[i]
- 对于无序集合数据（如1024个抓取），这种配对是任意的
- 最优传输配对可以最小化总体传输距离，让速度场更平滑，更易学习

References:
- Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
- Lipman et al. (2023). Flow Matching for Generative Modeling.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from models.utils.log_colors import BLUE, GREEN, YELLOW, RED, ENDC


from utils.hand_model import TRANSLATION_SLICE, QPOS_SLICE, ROTATION_SLICE


class SinkhornOT(nn.Module):
    """
    Sinkhorn算法计算最优传输配对
    
    核心思想：
    1. 最优传输问题：找到从x0到x1的最优配对，最小化总传输代价
    2. Sinkhorn算法：通过熵正则化将OT问题转化为矩阵缩放问题
    3. 输出：每个x0[i]对应的最优x1[j]的索引
    
    优势：
    - GPU并行化，batch处理
    - 可微分，支持端到端训练
    - 计算复杂度：O(N^2 * num_iters)，实际很快
    """
    
    def __init__(
        self,
        reg: float = 0.1,
        num_iters: int = 50,
        distance_metric: str = 'euclidean',
        matching_strategy: str = 'greedy',
        normalize_cost: bool = True,
        component_weights: Optional[Dict[str, float]] = None,
        component_slices: Optional[Dict[str, slice]] = None,
    ):
        """
        Args:
            reg: 熵正则化参数，越小越接近真实OT（但收敛慢）
                 推荐范围：0.05-0.2
            num_iters: Sinkhorn迭代次数，通常50-100足够
            distance_metric: 距离度量方式
                - 'euclidean': L2距离（默认）
                - 'squared': L2距离的平方（更强调远距离）
                - 'weighted': 加权距离（需要提供component_weights和component_slices）
            matching_strategy: 从传输计划提取配对的策略
                - 'greedy': 每行取最大值（快速）
                - 'hungarian': Hungarian算法（精确但慢）
            normalize_cost: 是否归一化代价矩阵（提升数值稳定性）
            component_weights: 各组件的权重字典，例如:
                {'translation': 1.0, 'qpos': 0.5, 'rotation': 2.0}
            component_slices: 各组件的slice字典，例如:
                {'translation': slice(0, 3), 'qpos': slice(3, 19), 'rotation': slice(19, None)}
        """
        super().__init__()
        self.reg = reg
        self.num_iters = num_iters
        self.distance_metric = distance_metric
        self.matching_strategy = matching_strategy
        self.normalize_cost = normalize_cost

        # 组件加权配置
        self.component_weights = component_weights
        # 若未提供切片，使用全局默认切片
        if component_slices is None:
            component_slices = {
                'translation': TRANSLATION_SLICE,
                'qpos': QPOS_SLICE,
                'rotation': ROTATION_SLICE,
            }
        self.component_slices = component_slices

        # 验证加权配置
        if distance_metric == 'weighted':
            if component_weights is None:
                raise ValueError(
                    "distance_metric='weighted' requires 'component_weights'; slices are inferred from globals"
                )
            # 允许权重子集（未给的按0处理）

        # 统计信息
        self.register_buffer('total_calls', torch.tensor(0, dtype=torch.long))
        self.register_buffer('avg_distance', torch.tensor(0.0))
        
    def compute_cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        计算代价矩阵 C[b,i,j] = cost(x0[b,i], x1[b,j])

        Args:
            x0: [B, N, D]
            x1: [B, N, D]

        Returns:
            C: [B, N, N]
        """
        if self.distance_metric == 'euclidean':
            # L2距离
            C = torch.cdist(x0, x1, p=2)
        elif self.distance_metric == 'squared':
            # L2距离的平方
            C = torch.cdist(x0, x1, p=2) ** 2
        elif self.distance_metric == 'weighted':
            # 加权距离：对不同组件分别计算距离并加权
            C = self._compute_weighted_cost_matrix(x0, x1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        # 可选：归一化到[0,1]范围，提升数值稳定性
        if self.normalize_cost:
            C = C / (C.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

        return C

    def _compute_weighted_cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        计算加权代价矩阵：对pose的不同组件（平移、关节角、旋转）分别计算距离并加权求和

        Args:
            x0: [B, N, D] 真实抓取
            x1: [B, N, D] 噪声抓取

        Returns:
            C: [B, N, N] 加权代价矩阵
        """
        B, N, D = x0.shape
        device = x0.device

        # 初始化总代价矩阵
        C_total = torch.zeros(B, N, N, device=device)

        # 逐组件计算并加权
        for component_name, component_slice in self.component_slices.items():
            weight = self.component_weights[component_name]

            # 提取该组件
            x0_comp = x0[:, :, component_slice]  # [B, N, D_comp]
            x1_comp = x1[:, :, component_slice]  # [B, N, D_comp]

            # 计算该组件的L2距离
            C_comp = torch.cdist(x0_comp, x1_comp, p=2)  # [B, N, N]

            # 可选：对每个组件归一化（使不同维度的组件在同一尺度）
            # 这样权重才有可比性
            C_comp_normalized = C_comp / (C_comp.mean() + 1e-8)

            # 加权累加
            C_total += weight * C_comp_normalized

        return C_total
    
    def sinkhorn_iterations(
        self, 
        K: torch.Tensor, 
        num_iters: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sinkhorn迭代求解传输计划
        
        目标：找到u, v使得 P = diag(u) @ K @ diag(v) 
             满足行和列和都为1/N（均匀边际分布）
        
        Args:
            K: Gibbs kernel [B, N, N]
            num_iters: 迭代次数（None使用默认值）
            
        Returns:
            u: [B, N]
            v: [B, N]
        """
        B, N, _ = K.shape
        device = K.device
        
        if num_iters is None:
            num_iters = self.num_iters
        
        # 初始化为均匀分布
        u = torch.ones(B, N, device=device) / N
        v = torch.ones(B, N, device=device) / N
        
        # Sinkhorn迭代
        for it in range(num_iters):
            # u = 1 / (K @ v)
            u = 1.0 / (K @ v.unsqueeze(-1) + 1e-10).squeeze(-1)
            
            # v = 1 / (K^T @ u)
            v = 1.0 / (K.transpose(1, 2) @ u.unsqueeze(-1) + 1e-10).squeeze(-1)
            
            # 检查数值稳定性
            if torch.isnan(u).any() or torch.isnan(v).any():
                logging.warning(f"{YELLOW}[SinkhornOT] NaN detected at iteration {it}, "
                              f"try increasing reg parameter{ENDC}")
                # 回退到均匀分布
                u = torch.ones(B, N, device=device) / N
                v = torch.ones(B, N, device=device) / N
                break
        
        return u, v
    
    def extract_matching(
        self, 
        transport_plan: torch.Tensor
    ) -> torch.Tensor:
        """
        从传输计划提取最优配对
        
        Args:
            transport_plan: [B, N, N]
            
        Returns:
            matchings: [B, N] 索引数组
        """
        if self.matching_strategy == 'greedy':
            # 贪婪策略：每行取最大值
            matchings = transport_plan.argmax(dim=-1)
            
        elif self.matching_strategy == 'hungarian':
            # Hungarian算法（精确但需要CPU）
            try:
                from scipy.optimize import linear_sum_assignment
                B, N, _ = transport_plan.shape
                matchings = []
                
                for b in range(B):
                    # 转为最大化问题（Hungarian求最小）
                    cost_matrix = -transport_plan[b].detach().cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    matchings.append(torch.tensor(col_ind, device=transport_plan.device))
                
                matchings = torch.stack(matchings)
            except ImportError:
                logging.warning(f"{YELLOW}[SinkhornOT] scipy not available, "
                              f"falling back to greedy matching{ENDC}")
                matchings = transport_plan.argmax(dim=-1)
        else:
            raise ValueError(f"Unknown matching strategy: {self.matching_strategy}")
        
        return matchings
    
    def forward(
        self, 
        x0: torch.Tensor, 
        x1: torch.Tensor,
        return_info: bool = False
    ) -> torch.Tensor:
        """
        计算最优传输配对
        
        Args:
            x0: 真实抓取集合 [B, N, D]
            x1: 噪声集合 [B, N, D]
            return_info: 是否返回额外信息（用于调试/可视化）
            
        Returns:
            matchings: [B, N] 配对索引
                使得 x1_matched = x1[b, matchings[b]] 与 x0[b] 最优配对
            info (可选): 包含传输计划、代价矩阵等信息的字典
        """
        B, N, D = x0.shape
        device = x0.device
        
        # 1. 计算代价矩阵
        C = self.compute_cost_matrix(x0, x1)  # [B, N, N]
        
        # 2. 计算Gibbs kernel: K = exp(-C/reg)
        K = torch.exp(-C / self.reg)
        
        # 3. Sinkhorn迭代
        u, v = self.sinkhorn_iterations(K)
        
        # 4. 计算传输计划: P = diag(u) @ K @ diag(v)
        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)
        
        # 5. 提取配对
        matchings = self.extract_matching(transport_plan)
        
        # 更新统计信息
        self.total_calls += 1
        
        if return_info:
            # 计算配对后的平均距离
            x1_matched = torch.gather(
                x1, dim=1,
                index=matchings.unsqueeze(-1).expand(B, N, D)
            )
            matched_dist = torch.norm(x0 - x1_matched, dim=-1).mean()
            random_dist = torch.norm(x0 - x1, dim=-1).mean()
            
            info = {
                'transport_plan': transport_plan,
                'cost_matrix': C,
                'matched_distance': matched_dist.item(),
                'random_distance': random_dist.item(),
                'improvement': (1 - matched_dist / random_dist).item() * 100,
                'u': u,
                'v': v,
            }
            return matchings, info
        
        return matchings


def apply_optimal_matching(
    x0: torch.Tensor,
    x1: torch.Tensor,
    matchings: torch.Tensor
) -> torch.Tensor:
    """
    应用最优配对，重排序x1
    
    Args:
        x0: [B, N, D] 真实数据
        x1: [B, N, D] 噪声数据
        matchings: [B, N] 配对索引
        
    Returns:
        x1_matched: [B, N, D] 重排序后的噪声
    """
    B, N, D = x1.shape
    
    x1_matched = torch.gather(
        x1, dim=1,
        index=matchings.unsqueeze(-1).expand(B, N, D)
    )
    
    return x1_matched


def compute_matching_quality(
    x0: torch.Tensor,
    x1_original: torch.Tensor,
    x1_matched: torch.Tensor
) -> Dict[str, float]:
    """
    评估配对质量
    
    Args:
        x0: [B, N, D] 真实数据
        x1_original: [B, N, D] 原始噪声
        x1_matched: [B, N, D] 配对后的噪声
        
    Returns:
        metrics: 包含各种质量指标的字典
    """
    # 计算距离
    dist_original = torch.norm(x0 - x1_original, dim=-1)
    dist_matched = torch.norm(x0 - x1_matched, dim=-1)
    
    metrics = {
        'mean_dist_original': dist_original.mean().item(),
        'mean_dist_matched': dist_matched.mean().item(),
        'std_dist_original': dist_original.std().item(),
        'std_dist_matched': dist_matched.std().item(),
        'min_dist_original': dist_original.min().item(),
        'min_dist_matched': dist_matched.min().item(),
        'max_dist_original': dist_original.max().item(),
        'max_dist_matched': dist_matched.max().item(),
        'improvement_ratio': (1 - dist_matched.mean() / dist_original.mean()).item(),
        'improvement_percent': (1 - dist_matched.mean() / dist_original.mean()).item() * 100,
    }
    
    return metrics


# 便捷函数
def sinkhorn_matching(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.1,
    num_iters: int = 50,
    **kwargs
) -> torch.Tensor:
    """
    便捷函数：一次性计算Sinkhorn最优配对
    
    Args:
        x0: [B, N, D]
        x1: [B, N, D]
        reg: 正则化参数
        num_iters: 迭代次数
        **kwargs: 其他参数传递给SinkhornOT
        
    Returns:
        matchings: [B, N]
    """
    ot_solver = SinkhornOT(reg=reg, num_iters=num_iters, **kwargs)
    return ot_solver(x0, x1)

