# Lazy imports to avoid dependency issues
def __getattr__(name):
    if name == 'GraspLossPose':
        from .grasp_loss_pose import GraspLossPose
        return GraspLossPose
    elif name == 'GraspSetDistance':
        from .set_distance import GraspSetDistance
        return GraspSetDistance
    elif name == 'extract_grasp_components':
        from .set_distance import extract_grasp_components
        return extract_grasp_components
    elif name == 'compute_pairwise_grasp_distance':
        from .set_distance import compute_pairwise_grasp_distance
        return compute_pairwise_grasp_distance
    elif name == 'SinkhornOTLoss':
        from .set_losses import SinkhornOTLoss
        return SinkhornOTLoss
    elif name == 'ChamferDistanceLoss':
        from .set_losses import ChamferDistanceLoss
        return ChamferDistanceLoss
    elif name == 'RepulsionLoss':
        from .set_losses import RepulsionLoss
        return RepulsionLoss
    elif name == 'PhysicsFeasibilityLoss':
        from .set_losses import PhysicsFeasibilityLoss
        return PhysicsFeasibilityLoss
    elif name == 'compute_set_losses':
        from .set_losses import compute_set_losses
        return compute_set_losses
    elif name == 'compute_timestep_weight':
        from .set_losses import compute_timestep_weight
        return compute_timestep_weight
    elif name == 'CoverageMetric':
        from .set_metrics import CoverageMetric
        return CoverageMetric
    elif name == 'MinimumMatchingDistanceMetric':
        from .set_metrics import MinimumMatchingDistanceMetric
        return MinimumMatchingDistanceMetric
    elif name == 'DiversityMetric':
        from .set_metrics import DiversityMetric
        return DiversityMetric
    elif name == 'PrecisionRecallMetric':
        from .set_metrics import PrecisionRecallMetric
        return PrecisionRecallMetric
    elif name == 'compute_set_metrics':
        from .set_metrics import compute_set_metrics
        return compute_set_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Keep backward compatibility
__all__ = [
    'GraspLossPose',
    'GraspSetDistance',
    'extract_grasp_components',
    'compute_pairwise_grasp_distance',
    'SinkhornOTLoss',
    'ChamferDistanceLoss',
    'RepulsionLoss',
    'PhysicsFeasibilityLoss',
    'compute_set_losses',
    'compute_timestep_weight',
    'CoverageMetric',
    'MinimumMatchingDistanceMetric',
    'DiversityMetric',
    'PrecisionRecallMetric',
    'compute_set_metrics',
]
