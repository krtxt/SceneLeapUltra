"""
Simple test script for set-based grasp learning modules (no project dependencies).

Tests the core modules directly without loading the full project.
"""

import logging
import sys
sys.path.insert(0, '/home/engine/project')

import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_distance_module():
    """Test set_distance module."""
    logger.info("=" * 80)
    logger.info("Testing set_distance module")
    logger.info("=" * 80)
    
    try:
        from models.loss.set_distance import (
            GraspSetDistance,
            extract_grasp_components,
            compute_pairwise_grasp_distance
        )
        logger.info("✓ Module imports successful")
        
        # Test distance computation
        distance_cfg = {
            'alpha_translation': 1.0,
            'alpha_rotation': 1.0,
            'alpha_qpos': 0.5,
            'rot_type': 'rot6d',
            'normalize_translation': False,
        }
        
        B, N, M = 2, 10, 15
        pred_poses = torch.randn(B, N, 22)
        target_poses = torch.randn(B, M, 22)
        
        distance_matrix = compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        
        logger.info(f"✓ Distance matrix computed: shape={distance_matrix.shape}")
        logger.info(f"  Mean: {distance_matrix.mean().item():.4f}")
        logger.info(f"  Min: {distance_matrix.min().item():.4f}")
        logger.info(f"  Max: {distance_matrix.max().item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_losses_module():
    """Test set_losses module."""
    logger.info("=" * 80)
    logger.info("Testing set_losses module")
    logger.info("=" * 80)
    
    try:
        from models.loss.set_losses import (
            SinkhornOTLoss,
            ChamferDistanceLoss,
            RepulsionLoss,
            PhysicsFeasibilityLoss,
            compute_set_losses
        )
        logger.info("✓ Module imports successful")
        
        # Test Sinkhorn OT Loss
        B, N, M = 2, 10, 15
        cost_matrix = torch.rand(B, N, M) * 10
        
        ot_loss_fn = SinkhornOTLoss({'epsilon': 0.1, 'max_iter': 50})
        ot_loss = ot_loss_fn(cost_matrix)
        logger.info(f"✓ OT loss: {ot_loss.item():.4f}")
        
        # Test Chamfer Distance Loss
        cd_loss_fn = ChamferDistanceLoss({})
        cd_loss = cd_loss_fn(cost_matrix)
        logger.info(f"✓ Chamfer loss: {cd_loss.item():.4f}")
        
        # Test Repulsion Loss
        pred_poses = torch.randn(B, N, 22)
        rep_loss_fn = RepulsionLoss({'k': 3})
        rep_loss = rep_loss_fn(pred_poses)
        logger.info(f"✓ Repulsion loss: {rep_loss.item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_metrics_module():
    """Test set_metrics module."""
    logger.info("=" * 80)
    logger.info("Testing set_metrics module")
    logger.info("=" * 80)
    
    try:
        from models.loss.set_metrics import (
            CoverageMetric,
            MinimumMatchingDistanceMetric,
            DiversityMetric,
            PrecisionRecallMetric,
            compute_set_metrics
        )
        logger.info("✓ Module imports successful")
        
        B, N, M = 2, 10, 15
        distance_matrix = torch.rand(B, N, M) * 0.3
        pred_poses = torch.randn(B, N, 22)
        
        # Test Coverage
        cov_fn = CoverageMetric({'thresholds': [0.1, 0.2]})
        cov_metrics = cov_fn(distance_matrix)
        logger.info(f"✓ Coverage metrics: {cov_metrics}")
        
        # Test MMD
        mmd_fn = MinimumMatchingDistanceMetric({})
        mmd_metrics = mmd_fn(distance_matrix)
        logger.info(f"✓ MMD: {mmd_metrics['mmd']:.4f}")
        
        # Test Diversity
        div_fn = DiversityMetric({})
        div_metrics = div_fn(pred_poses)
        logger.info(f"✓ Diversity metrics: {div_metrics}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_integration():
    """Test integration of all modules."""
    logger.info("=" * 80)
    logger.info("Testing Integration")
    logger.info("=" * 80)
    
    try:
        from models.loss.set_distance import compute_pairwise_grasp_distance
        from models.loss.set_losses import compute_set_losses
        from models.loss.set_metrics import compute_set_metrics
        
        B, N, M = 2, 10, 15
        pred_poses = torch.randn(B, N, 22)
        target_poses = torch.randn(B, M, 22)
        
        # Mock outputs and targets
        outputs = {'matched': {'norm_pose': pred_poses}}
        targets = {'matched': {'norm_pose': target_poses}}
        
        # Configure
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
            'ot_config': {'epsilon': 0.1, 'max_iter': 50},
            'cd_config': {},
            'repulsion_config': {'k': 3},
            'physics_config': {}
        }
        
        metric_cfg = {
            'compute_coverage': True,
            'compute_mmd': True,
            'compute_diversity': True,
            'coverage_config': {'thresholds': [0.1, 0.2]},
        }
        
        # Compute distance matrix
        distance_matrix = compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        logger.info(f"✓ Distance matrix: {distance_matrix.shape}")
        
        # Compute losses
        losses = compute_set_losses(
            outputs, targets, distance_matrix, set_loss_cfg
        )
        logger.info(f"✓ Set losses: {list(losses.keys())}")
        for k, v in losses.items():
            logger.info(f"  {k}: {v.item():.4f}")
        
        # Compute metrics
        metrics = compute_set_metrics(
            pred_poses, target_poses, distance_matrix, metric_cfg
        )
        logger.info(f"✓ Set metrics: {list(metrics.keys())}")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING SIMPLE SET-BASED GRASP LEARNING TESTS")
    logger.info("=" * 80 + "\n")
    
    results = {}
    
    results['distance'] = test_distance_module()
    results['losses'] = test_losses_module()
    results['metrics'] = test_metrics_module()
    results['integration'] = test_integration()
    
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
