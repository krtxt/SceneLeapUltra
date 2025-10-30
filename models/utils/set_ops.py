from typing import Dict, Optional

import torch
import torch.nn.functional as F


# Slices for pose components
TRANSLATION_SLICE = slice(0, 3)
QPOS_SLICE = slice(3, 19)
ROTATION_SLICE = slice(19, None)

def _extract_component(data: Dict[str, torch.Tensor], primary_key: str, fallback_key: str, value_slice: slice) -> Optional[torch.Tensor]:
    if not isinstance(data, dict):
        return None

    # If the specific component is already provided, return it directly
    tensor = data.get(primary_key)
    if tensor is not None:
        return tensor

    # Otherwise, fall back to slicing from the full pose tensor
    tensor = data.get(fallback_key)
    if tensor is None:
        return None

    # Perform slicing along the last dimension using the provided slice
    # value_slice.stop may be None (e.g., rotation), which is valid for slicing
    return tensor[..., value_slice]


def _prepare_prediction_components(prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    components = {}
    pred_pose = prediction.get('pred_pose_norm')
    if pred_pose is not None and pred_pose.dim() == 2:
        pred_pose = pred_pose.unsqueeze(1)

    components['translation'] = _extract_component(prediction, 'translation_norm', 'pred_pose_norm', TRANSLATION_SLICE)
    components['qpos'] = _extract_component(prediction, 'qpos_norm', 'pred_pose_norm', QPOS_SLICE)
    components['rotation'] = prediction.get('rotation')
    if components['rotation'] is None and pred_pose is not None:
        components['rotation'] = pred_pose[..., ROTATION_SLICE]

    if components['translation'] is None and pred_pose is not None:
        components['translation'] = pred_pose[..., TRANSLATION_SLICE]
    if components['qpos'] is None and pred_pose is not None:
        components['qpos'] = pred_pose[..., QPOS_SLICE]

    return components


def _prepare_target_components(targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    components = {}
    target_pose = targets.get('norm_pose') if isinstance(targets, dict) else None
    if target_pose is not None and target_pose.dim() == 2:
        target_pose = target_pose.unsqueeze(1)

    components['translation'] = _extract_component(targets, 'translation_norm', 'norm_pose', TRANSLATION_SLICE)
    components['qpos'] = _extract_component(targets, 'qpos_norm', 'norm_pose', QPOS_SLICE)
    components['rotation'] = targets.get('rotation')
    if components['rotation'] is None and target_pose is not None:
        components['rotation'] = target_pose[..., ROTATION_SLICE]

    if components['translation'] is None and target_pose is not None:
        components['translation'] = target_pose[..., TRANSLATION_SLICE]
    if components['qpos'] is None and target_pose is not None:
        components['qpos'] = target_pose[..., QPOS_SLICE]

    return components


def _ensure_three_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    return tensor


def _cdist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L1 distances [B, N, M] without materializing [B, N, M, D] when possible.

    Falls back to a vectorized implementation if torch.cdist with p=1 is not available.
    """
    try:
        return torch.cdist(x, y, p=1)
    except (TypeError, RuntimeError):
        # Fallback: vectorized absolute difference (may allocate [B, N, M, D])
        return (x.unsqueeze(2) - y.unsqueeze(1)).abs().sum(dim=-1)


def pairwise_cost(
    prediction: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Dict[str, float],
    rot_type: str,
    valid_mask: Optional[torch.Tensor] = None,
    return_components: bool = False,
):
    """
    Compute differentiable pairwise cost matrix between predicted and target grasp sets.

    Args:
        prediction: Dictionary containing prediction tensors in normalized pose space.
        targets: Dictionary containing target tensors in normalized pose space.
        weights: Component weights for translation, rotation, and qpos.
        rot_type: Rotation representation ("r6d" or "quat").
        valid_mask: Optional mask marking valid target grasps (B, M).

    Returns:
        torch.Tensor: Pairwise cost matrix of shape [B, N_pred, N_gt].
        When return_components=True, also returns a dict containing the weighted
        per-component cost matrices.
    """
    pred_components = _prepare_prediction_components(prediction)
    target_components = _prepare_target_components(targets)

    translation_weight = float(weights.get('translation', 0.0))
    rotation_weight = float(weights.get('rotation', 0.0))
    qpos_weight = float(weights.get('qpos', 0.0))

    pred_translation = _ensure_three_dim(pred_components.get('translation'))
    pred_qpos = _ensure_three_dim(pred_components.get('qpos'))
    pred_rotation = _ensure_three_dim(pred_components.get('rotation'))

    target_translation = _ensure_three_dim(target_components.get('translation'))
    target_qpos = _ensure_three_dim(target_components.get('qpos'))
    target_rotation = _ensure_three_dim(target_components.get('rotation'))

    if pred_translation is None or target_translation is None:
        raise ValueError("Translation components are required for pairwise cost computation.")

    device = pred_translation.device

    base_shape = pred_translation.shape[:-1]
    cost = pred_translation.new_zeros((base_shape[0], base_shape[1], target_translation.shape[1]))
    component_costs = {} if return_components else None

    if translation_weight > 0 and target_translation is not None:
        translation_cost = _cdist_l1(pred_translation, target_translation)
        weighted_translation_cost = translation_weight * translation_cost
        cost = cost + weighted_translation_cost
        if return_components:
            component_costs['translation'] = weighted_translation_cost
    elif return_components:
        component_costs['translation'] = cost.new_zeros(cost.shape)

    if qpos_weight > 0 and pred_qpos is not None and target_qpos is not None:
        qpos_cost = _cdist_l1(pred_qpos, target_qpos)
        weighted_qpos_cost = qpos_weight * qpos_cost
        cost = cost + weighted_qpos_cost
        if return_components:
            component_costs['qpos'] = weighted_qpos_cost
    elif return_components:
        component_costs['qpos'] = cost.new_zeros(cost.shape)

    if rotation_weight > 0 and pred_rotation is not None and target_rotation is not None:
        if rot_type == 'quat':
            normalized_pred = F.normalize(pred_rotation, dim=-1)
            normalized_target = F.normalize(target_rotation, dim=-1)
            rotation_cost = 1.0 - torch.abs(
                torch.matmul(normalized_pred, normalized_target.transpose(-2, -1))
            )
        else:
            rotation_cost = _cdist_l1(pred_rotation, target_rotation)
        weighted_rotation_cost = rotation_weight * rotation_cost
        cost = cost + weighted_rotation_cost
        if return_components:
            component_costs['rotation'] = weighted_rotation_cost
    elif return_components:
        component_costs['rotation'] = cost.new_zeros(cost.shape)

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        broadcast_mask = valid_mask.unsqueeze(1)
        inf_value = torch.finfo(cost.dtype).max
        cost = torch.where(broadcast_mask, cost, cost.new_full((), inf_value))
        if return_components:
            for key, value in component_costs.items():
                component_costs[key] = torch.where(
                    broadcast_mask,
                    value,
                    value.new_full((), inf_value)
                )
    elif return_components:
        for key, value in component_costs.items():
            component_costs[key] = value

    if return_components:
        return cost, component_costs

    return cost


def chamfer_set_loss(cost_matrix: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute symmetric Chamfer loss over a precomputed cost matrix."""
    if cost_matrix.numel() == 0:
        return cost_matrix.new_tensor(0.0)

    # Forward direction: prediction to ground truth
    forward_min = cost_matrix.min(dim=2).values
    forward_min = torch.where(torch.isfinite(forward_min), forward_min, forward_min.new_zeros(forward_min.shape))
    loss_forward = forward_min.mean()

    # Backward direction: ground truth to prediction
    backward_min = cost_matrix.min(dim=1).values
    if valid_mask is not None:
        valid_mask = valid_mask.to(cost_matrix.device)
        backward_min = torch.where(valid_mask, backward_min, backward_min.new_zeros(backward_min.shape))
        denom = valid_mask.sum(dim=1).clamp(min=1)
        loss_backward = (backward_min.sum(dim=1) / denom).mean()
    else:
        backward_min = torch.where(torch.isfinite(backward_min), backward_min, backward_min.new_zeros(backward_min.shape))
        loss_backward = backward_min.mean()

    return loss_forward + loss_backward


def sinkhorn_ot_loss(
    cost_matrix: torch.Tensor,
    *,
    eps: float = 0.05,
    tau: float = 0.5,
    iters: int = 50,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute an entropic OT loss using Sinkhorn iterations in log space."""
    _ = tau  # Placeholder for future unbalanced OT support
    if cost_matrix.numel() == 0:
        return cost_matrix.new_tensor(0.0)

    B, N, M = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    a = cost_matrix.new_full((B, N), 1.0 / max(N, 1))
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        b = valid_mask.float()
        counts = b.sum(dim=1, keepdim=True).clamp(min=1.0)
        b = b / counts
    else:
        b = cost_matrix.new_full((B, M), 1.0 / max(M, 1))

    log_a = torch.log(a)
    log_b = torch.where(b > 0, torch.log(b), b.new_full(b.shape, -float('inf')))

    log_K = -cost_matrix / eps
    if valid_mask is not None:
        broadcast_mask = valid_mask.unsqueeze(1)
        log_K = torch.where(broadcast_mask, log_K, cost_matrix.new_full(log_K.shape, -float('inf')))

    log_u = cost_matrix.new_zeros(B, N)
    log_v = cost_matrix.new_zeros(B, M)
    if valid_mask is not None:
        log_v = torch.where(valid_mask, log_v, log_v.new_full(log_v.shape, -float('inf')))

    for _ in range(iters):
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        log_v = log_b - torch.logsumexp(log_K.transpose(1, 2) + log_u.unsqueeze(2), dim=2)
        if valid_mask is not None:
            log_v = torch.where(valid_mask, log_v, log_v.new_full(log_v.shape, -float('inf')))

    plan_log = log_u.unsqueeze(2) + log_v.unsqueeze(1) + log_K
    transport_plan = torch.exp(plan_log)
    transport_plan = torch.nan_to_num(transport_plan, nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        transport_plan = transport_plan * valid_mask.unsqueeze(1).float()

    ot_cost = (transport_plan * cost_matrix).sum(dim=(1, 2))
    return ot_cost.mean()
