"""
Test script for set-based grasp learning losses and metrics.

This script tests:
1. Distance computation
2. Set losses (OT, Chamfer, Repulsion, Physics)
3. Set metrics (Coverage, MMD, Diversity)
"""

import logging
import sys
sys.path.insert(0, '/home/engine/project')

import torch
from omegaconf import OmegaConf

from models.loss import (
    GraspSetDistance,
    extract_grasp_components,
    compute_pairwise_grasp_distance,
    SinkhornOTLoss,
    ChamferDistanceLoss,
    RepulsionLoss,
    compute_set_losses,
    CoverageMetric,
    MinimumMatchingDistanceMetric,
    DiversityMetric,
    compute_set_metrics
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_distance_computation():
    """Test grasp distance computation."""
    logger.info("=" * 80)
    logger.info("Testing Distance Computation")
    logger.info("=" * 80)
    
    # Create synthetic grasp data
    # Format: [B, N, pose_dim] where pose_dim = 3 (trans) + 16 (qpos) + 3 (rot6d) = 22
    B, N, M = 2, 10, 15
    pred_poses = torch.randn(B, N, 22)
    target_poses = torch.randn(B, M, 22)
    
    # Configure distance function
    distance_cfg = {
        'alpha_translation': 1.0,
        'alpha_rotation': 1.0,
        'alpha_qpos': 0.5,
        'rot_type': 'rot6d',
        'normalize_translation': False,  # Disable for synthetic data
    }
    
    # Compute distance matrix
    try:
        distance_matrix = compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        logger.info(f"✓ Distance matrix computed: shape={distance_matrix.shape}")
        logger.info(f"  Mean distance: {distance_matrix.mean().item():.4f}")
        logger.info(f"  Min distance: {distance_matrix.min().item():.4f}")
        logger.info(f"  Max distance: {distance_matrix.max().item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Distance computation failed: {e}", exc_info=True)
        return False


def test_ot_loss():
    """Test Sinkhorn Optimal Transport loss."""
    logger.info("=" * 80)
    logger.info("Testing Sinkhorn OT Loss")
    logger.info("=" * 80)
    
    # Create synthetic cost matrix
    B, N, M = 2, 10, 15
    cost_matrix = torch.rand(B, N, M) * 10
    
    # Configure OT loss
    ot_cfg = {
        'epsilon': 0.1,
        'tau': 1.0,
        'max_iter': 50,
        'max_samples': 100
    }
    
    try:
        ot_loss_fn = SinkhornOTLoss(ot_cfg)
        loss = ot_loss_fn(cost_matrix)
        logger.info(f"✓ OT loss computed: {loss.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ OT loss computation failed: {e}", exc_info=True)
        return False


def test_chamfer_loss():
    """Test Chamfer Distance loss."""
    logger.info("=" * 80)
    logger.info("Testing Chamfer Distance Loss")
    logger.info("=" * 80)
    
    # Create synthetic cost matrix
    B, N, M = 2, 10, 15
    cost_matrix = torch.rand(B, N, M) * 10
    
    try:
        cd_loss_fn = ChamferDistanceLoss({})
        loss = cd_loss_fn(cost_matrix)
        logger.info(f"✓ Chamfer loss computed: {loss.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Chamfer loss computation failed: {e}", exc_info=True)
        return False


def test_repulsion_loss():
    """Test Repulsion (diversity) loss."""
    logger.info("=" * 80)
    logger.info("Testing Repulsion Loss")
    logger.info("=" * 80)
    
    # Create synthetic predictions
    B, N, D = 2, 10, 22
    pred_poses = torch.randn(B, N, D)
    
    # Configure repulsion loss
    rep_cfg = {
        'k': 3,
        'lambda_repulsion': 1.0,
        'delta': 0.01,
        's': 2.0
    }
    
    try:
        rep_loss_fn = RepulsionLoss(rep_cfg)
        loss = rep_loss_fn(pred_poses)
        logger.info(f"✓ Repulsion loss computed: {loss.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Repulsion loss computation failed: {e}", exc_info=True)
        return False


def test_coverage_metric():
    """Test Coverage metric."""
    logger.info("=" * 80)
    logger.info("Testing Coverage Metric")
    logger.info("=" * 80)
    
    # Create synthetic distance matrix
    B, N, M = 2, 10, 15
    distance_matrix = torch.rand(B, N, M) * 0.3  # Small distances
    
    # Configure coverage metric
    cov_cfg = {
        'thresholds': [0.05, 0.1, 0.2]
    }
    
    try:
        cov_fn = CoverageMetric(cov_cfg)
        metrics = cov_fn(distance_matrix)
        logger.info(f"✓ Coverage metrics computed:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Coverage metric computation failed: {e}", exc_info=True)
        return False


def test_mmd_metric():
    """Test Minimum Matching Distance metric."""
    logger.info("=" * 80)
    logger.info("Testing MMD Metric")
    logger.info("=" * 80)
    
    # Create synthetic distance matrix
    B, N, M = 2, 10, 15
    distance_matrix = torch.rand(B, N, M) * 10
    
    try:
        mmd_fn = MinimumMatchingDistanceMetric({})
        metrics = mmd_fn(distance_matrix)
        logger.info(f"✓ MMD metric computed: {metrics['mmd']:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ MMD metric computation failed: {e}", exc_info=True)
        return False


def test_diversity_metric():
    """Test Diversity metric (NND stats)."""
    logger.info("=" * 80)
    logger.info("Testing Diversity Metric")
    logger.info("=" * 80)
    
    # Create synthetic predictions
    B, N, D = 2, 10, 22
    pred_poses = torch.randn(B, N, D)
    
    try:
        div_fn = DiversityMetric({})
        metrics = div_fn(pred_poses)
        logger.info(f"✓ Diversity metrics computed:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Diversity metric computation failed: {e}", exc_info=True)
        return False


def test_integrated_losses():
    """Test integrated set losses computation."""
    logger.info("=" * 80)
    logger.info("Testing Integrated Set Losses")
    logger.info("=" * 80)
    
    # Create synthetic data
    B, N, M = 2, 10, 15
    pred_poses = torch.randn(B, N, 22)
    target_poses = torch.randn(B, M, 22)
    
    # Mock outputs and targets
    outputs = {
        'matched': {
            'norm_pose': pred_poses
        }
    }
    targets = {
        'matched': {
            'norm_pose': target_poses
        }
    }
    
    # Configure set losses
    distance_cfg = {
        'alpha_translation': 1.0,
        'alpha_rotation': 1.0,
        'alpha_qpos': 0.5,
        'rot_type': 'rot6d',
    }
    
    set_loss_cfg = {
        'lambda_ot': 1.0,
        'gamma_cd': 1.0,
        'eta_repulsion': 0.1,
        'zeta_physics': 0.0,
        'schedule_type': 'constant',
        'ot_config': {'epsilon': 0.1, 'max_iter': 50},
        'cd_config': {},
        'repulsion_config': {'k': 3},
        'physics_config': {}
    }
    
    # Compute distance matrix
    try:
        distance_matrix = compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        
        # Compute set losses
        losses = compute_set_losses(
            outputs, targets, distance_matrix, set_loss_cfg
        )
        
        logger.info(f"✓ Integrated set losses computed:")
        for k, v in losses.items():
            logger.info(f"  {k}: {v.item():.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Integrated set losses computation failed: {e}", exc_info=True)
        return False


def test_integrated_metrics():
    """Test integrated set metrics computation."""
    logger.info("=" * 80)
    logger.info("Testing Integrated Set Metrics")
    logger.info("=" * 80)
    
    # Create synthetic data
    B, N, M = 2, 10, 15
    pred_poses = torch.randn(B, N, 22)
    target_poses = torch.randn(B, M, 22)
    
    # Configure
    distance_cfg = {
        'alpha_translation': 1.0,
        'alpha_rotation': 1.0,
        'alpha_qpos': 0.5,
        'rot_type': 'rot6d',
    }
    
    metric_cfg = {
        'compute_coverage': True,
        'compute_mmd': True,
        'compute_diversity': True,
        'coverage_config': {'thresholds': [0.1, 0.2]},
        'mmd_config': {},
        'diversity_config': {}
    }
    
    try:
        # Compute distance matrix
        distance_matrix = compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        
        # Compute metrics
        metrics = compute_set_metrics(
            pred_poses, target_poses, distance_matrix, metric_cfg
        )
        
        logger.info(f"✓ Integrated set metrics computed:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        return True
    except Exception as e:
        logger.error(f"✗ Integrated set metrics computation failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING SET-BASED GRASP LEARNING TESTS")
    logger.info("=" * 80 + "\n")
    
    results = {}
    
    # Test distance computation
    results['distance'] = test_distance_computation()
    
    # Test losses
    results['ot_loss'] = test_ot_loss()
    results['chamfer_loss'] = test_chamfer_loss()
    results['repulsion_loss'] = test_repulsion_loss()
    
    # Test metrics
    results['coverage'] = test_coverage_metric()
    results['mmd'] = test_mmd_metric()
    results['diversity'] = test_diversity_metric()
    
    # Test integrated
    results['integrated_losses'] = test_integrated_losses()
    results['integrated_metrics'] = test_integrated_metrics()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 80 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
