"""
Set-based metrics for grasp generation evaluation.

Implements:
- Coverage (COV)
- Minimum Matching Distance (MMD)
- Nearest Neighbor Distance (NND) statistics for diversity
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn


class CoverageMetric(nn.Module):
    """
    Coverage (COV) metric.
    
    Measures the proportion of target grasps covered by predicted grasps
    within a distance threshold.
    """
    
    def __init__(self, metric_cfg):
        """
        Args:
            metric_cfg: Configuration dict with keys:
                - threshold: Distance threshold τ for coverage (default: 0.1)
                - thresholds: List of thresholds to evaluate (default: [0.05, 0.1, 0.2])
        """
        super().__init__()
        
        # Support both single threshold and multiple thresholds
        if 'thresholds' in metric_cfg:
            self.thresholds = metric_cfg['thresholds']
        elif 'threshold' in metric_cfg:
            self.thresholds = [metric_cfg['threshold']]
        else:
            self.thresholds = [0.05, 0.1, 0.2]
        
        logging.info(f"CoverageMetric initialized with thresholds: {self.thresholds}")
    
    def forward(self, distance_matrix):
        """
        Compute coverage metrics.
        
        Args:
            distance_matrix: [B, N, M] - distance from predictions to targets
            
        Returns:
            Dict of coverage values for each threshold
        """
        # For each target, find minimum distance to any prediction
        min_dist_to_target, _ = torch.min(distance_matrix, dim=1)  # [B, M]
        
        metrics = {}
        for tau in self.thresholds:
            # Count targets covered within threshold
            covered = (min_dist_to_target <= tau).float()  # [B, M]
            coverage = covered.mean(dim=-1)  # [B,]
            
            # Average over batch
            cov_value = coverage.mean().item()
            
            metrics[f'coverage@{tau}'] = cov_value
            logging.debug(f"Coverage@{tau}: {cov_value:.4f}")
        
        return metrics


class MinimumMatchingDistanceMetric(nn.Module):
    """
    Minimum Matching Distance (MMD) metric.
    
    Measures the average minimum distance from targets to predictions (fidelity).
    """
    
    def __init__(self, metric_cfg):
        """
        Args:
            metric_cfg: Configuration dict (no specific parameters currently)
        """
        super().__init__()
        logging.info("MinimumMatchingDistanceMetric initialized")
    
    def forward(self, distance_matrix):
        """
        Compute MMD metric.
        
        Args:
            distance_matrix: [B, N, M] - distance from predictions to targets
            
        Returns:
            Dict with 'mmd' value
        """
        # For each target, find minimum distance to any prediction
        min_dist_to_target, _ = torch.min(distance_matrix, dim=1)  # [B, M]
        
        # Average over targets and batch
        mmd = min_dist_to_target.mean().item()
        
        logging.debug(f"MMD: {mmd:.4f}")
        
        return {'mmd': mmd}


class DiversityMetric(nn.Module):
    """
    Diversity metrics based on Nearest Neighbor Distance (NND) statistics.
    
    Measures how uniformly distributed the predicted grasps are.
    """
    
    def __init__(self, metric_cfg):
        """
        Args:
            metric_cfg: Configuration dict (no specific parameters currently)
        """
        super().__init__()
        logging.info("DiversityMetric initialized")
    
    def forward(self, pred_poses):
        """
        Compute diversity metrics.
        
        Args:
            pred_poses: [B, N, D] - predicted grasp poses or embeddings
            
        Returns:
            Dict with diversity metrics:
                - 'mean_nnd': Mean nearest neighbor distance
                - 'std_nnd': Standard deviation of NND
                - 'cv_nnd': Coefficient of variation (CV = std/mean)
        """
        B, N, D = pred_poses.shape
        
        if N <= 1:
            # Cannot compute diversity for single grasp
            return {
                'mean_nnd': 0.0,
                'std_nnd': 0.0,
                'cv_nnd': 0.0
            }
        
        # Compute pairwise distances within predicted set
        # [B, N, 1, D] - [B, 1, N, D] -> [B, N, N]
        pred_expanded1 = pred_poses.unsqueeze(2)
        pred_expanded2 = pred_poses.unsqueeze(1)
        pairwise_dist = torch.norm(pred_expanded1 - pred_expanded2, dim=-1)  # [B, N, N]
        
        # Set diagonal to infinity to exclude self-distance
        pairwise_dist_masked = pairwise_dist.clone()
        pairwise_dist_masked.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        # Find nearest neighbor for each grasp
        nearest_neighbor_dist, _ = torch.min(pairwise_dist_masked, dim=-1)  # [B, N]
        
        # Compute statistics
        mean_nnd = nearest_neighbor_dist.mean().item()
        std_nnd = nearest_neighbor_dist.std().item()
        
        # Coefficient of variation (lower is more uniform)
        if mean_nnd > 1e-8:
            cv_nnd = std_nnd / mean_nnd
        else:
            cv_nnd = 0.0
        
        logging.debug(f"Diversity: mean_nnd={mean_nnd:.4f}, std_nnd={std_nnd:.4f}, cv_nnd={cv_nnd:.4f}")
        
        return {
            'mean_nnd': mean_nnd,
            'std_nnd': std_nnd,
            'cv_nnd': cv_nnd
        }


class PrecisionRecallMetric(nn.Module):
    """
    Precision and Recall metrics for grasp sets.
    
    Precision: What fraction of predictions are close to any target
    Recall: What fraction of targets have a close prediction (similar to coverage)
    """
    
    def __init__(self, metric_cfg):
        """
        Args:
            metric_cfg: Configuration dict with keys:
                - threshold: Distance threshold for matching (default: 0.1)
        """
        super().__init__()
        
        self.threshold = metric_cfg.get('threshold', 0.1)
        
        logging.info(f"PrecisionRecallMetric initialized with threshold: {self.threshold}")
    
    def forward(self, distance_matrix):
        """
        Compute precision and recall.
        
        Args:
            distance_matrix: [B, N, M] - distance from predictions to targets
            
        Returns:
            Dict with 'precision' and 'recall' values
        """
        # Precision: fraction of predictions with min distance to any target < threshold
        min_dist_to_target, _ = torch.min(distance_matrix, dim=-1)  # [B, N]
        precision = (min_dist_to_target <= self.threshold).float().mean().item()
        
        # Recall: fraction of targets with min distance to any prediction < threshold
        min_dist_to_pred, _ = torch.min(distance_matrix, dim=-2)  # [B, M]
        recall = (min_dist_to_pred <= self.threshold).float().mean().item()
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        logging.debug(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def compute_set_metrics(
    pred_poses,
    target_poses,
    distance_matrix,
    metric_cfg
):
    """
    Compute all set-based metrics.
    
    Args:
        pred_poses: [B, N, D] - predicted grasp poses
        target_poses: [B, M, D] - target grasp poses
        distance_matrix: [B, N, M] - precomputed distance matrix
        metric_cfg: Configuration dict for metrics
        
    Returns:
        Dict of metrics
    """
    metrics = {}
    
    # Get metric enable flags
    compute_coverage = metric_cfg.get('compute_coverage', True)
    compute_mmd = metric_cfg.get('compute_mmd', True)
    compute_diversity = metric_cfg.get('compute_diversity', True)
    compute_precision_recall = metric_cfg.get('compute_precision_recall', False)
    
    try:
        # 1. Coverage
        if compute_coverage:
            coverage_fn = CoverageMetric(metric_cfg.get('coverage_config', {}))
            coverage_metrics = coverage_fn(distance_matrix)
            metrics.update(coverage_metrics)
    except Exception as e:
        logging.error(f"Coverage metric computation failed: {e}")
    
    try:
        # 2. Minimum Matching Distance
        if compute_mmd:
            mmd_fn = MinimumMatchingDistanceMetric(metric_cfg.get('mmd_config', {}))
            mmd_metrics = mmd_fn(distance_matrix)
            metrics.update(mmd_metrics)
    except Exception as e:
        logging.error(f"MMD metric computation failed: {e}")
    
    try:
        # 3. Diversity
        if compute_diversity:
            diversity_fn = DiversityMetric(metric_cfg.get('diversity_config', {}))
            diversity_metrics = diversity_fn(pred_poses)
            metrics.update(diversity_metrics)
    except Exception as e:
        logging.error(f"Diversity metric computation failed: {e}")
    
    try:
        # 4. Precision/Recall (optional)
        if compute_precision_recall:
            pr_fn = PrecisionRecallMetric(metric_cfg.get('pr_config', {}))
            pr_metrics = pr_fn(distance_matrix)
            metrics.update(pr_metrics)
    except Exception as e:
        logging.error(f"Precision/Recall metric computation failed: {e}")
    
    return metrics


def compute_symmetry_aware_distance(distance_matrix, symmetry_group=None):
    """
    Compute symmetry-aware distance matrix.
    
    For symmetric objects, consider the minimum distance under symmetry transformations.
    
    Args:
        distance_matrix: [B, N, M] - original distance matrix
        symmetry_group: List of symmetry transformation matrices (not implemented yet)
        
    Returns:
        symmetry_aware_distance_matrix: [B, N, M]
    """
    # Placeholder: symmetry handling not fully implemented
    logging.debug("Symmetry-aware distance requested but not implemented, using original distance")
    return distance_matrix
