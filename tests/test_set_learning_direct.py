"""
Direct test of set-based grasp learning modules (bypassing __init__.py).
"""

import logging
import sys
sys.path.insert(0, '/home/engine/project')

import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("TESTING SET-BASED GRASP LEARNING MODULES")
    logger.info("=" * 80)
    
    # Test 1: Distance Module
    logger.info("\n[1/3] Testing set_distance module...")
    try:
        import models.loss.set_distance as sd
        
        distance_cfg = {
            'alpha_translation': 1.0,
            'alpha_rotation': 1.0,
            'alpha_qpos': 0.5,
            'rot_type': 'rot6d',
        }
        
        # For rot6d: pose_dim = 3 (trans) + 16 (qpos) + 6 (rot6d) = 25
        B, N, M = 2, 8, 12
        pred_poses = torch.randn(B, N, 25)
        target_poses = torch.randn(B, M, 25)
        
        distance_matrix = sd.compute_pairwise_grasp_distance(
            pred_poses, target_poses, distance_cfg, rot_type='rot6d'
        )
        
        logger.info(f"✓ Distance matrix: {distance_matrix.shape}")
        logger.info(f"  Stats: mean={distance_matrix.mean():.3f}, "
                   f"min={distance_matrix.min():.3f}, max={distance_matrix.max():.3f}")
    except Exception as e:
        logger.error(f"✗ Distance test failed: {e}", exc_info=True)
        return False
    
    # Test 2: Losses Module
    logger.info("\n[2/3] Testing set_losses module...")
    try:
        import models.loss.set_losses as sl
        
        # OT Loss
        ot_fn = sl.SinkhornOTLoss({'epsilon': 0.1, 'max_iter': 50})
        ot_loss = ot_fn(distance_matrix)
        logger.info(f"✓ OT Loss: {ot_loss.item():.4f}")
        
        # Chamfer Loss
        cd_fn = sl.ChamferDistanceLoss({})
        cd_loss = cd_fn(distance_matrix)
        logger.info(f"✓ Chamfer Loss: {cd_loss.item():.4f}")
        
        # Repulsion Loss
        rep_fn = sl.RepulsionLoss({'k': 3})
        rep_loss = rep_fn(pred_poses)
        logger.info(f"✓ Repulsion Loss: {rep_loss.item():.4f}")
        
    except Exception as e:
        logger.error(f"✗ Losses test failed: {e}", exc_info=True)
        return False
    
    # Test 3: Metrics Module
    logger.info("\n[3/3] Testing set_metrics module...")
    try:
        import models.loss.set_metrics as sm
        
        # Coverage
        cov_fn = sm.CoverageMetric({'thresholds': [0.1, 0.2]})
        cov = cov_fn(distance_matrix)
        logger.info(f"✓ Coverage: {cov}")
        
        # MMD
        mmd_fn = sm.MinimumMatchingDistanceMetric({})
        mmd = mmd_fn(distance_matrix)
        logger.info(f"✓ MMD: {mmd}")
        
        # Diversity
        div_fn = sm.DiversityMetric({})
        div = div_fn(pred_poses)
        logger.info(f"✓ Diversity: {div}")
        
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}", exc_info=True)
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
