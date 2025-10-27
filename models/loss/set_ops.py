from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _extract_component(data: Dict[str, torch.Tensor], primary_key: str, fallback_key: str, value_slice: slice) -> Optional[torch.Tensor]:
    if not isinstance(data, dict):
        return None

    tensor = data.get(primary_key)
    if tensor is None:
        tensor = data.get(fallback_key)
        if tensor is not None:
            tensor = tensor[..., value_slice]
    elif tensor.dim() >= value_slice.stop:
        tensor = tensor[..., value_slice]
    return tensor


def _prepare_prediction_components(prediction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    components = {}
    pred_pose = prediction.get('pred_pose_norm')
    if pred_pose is not None and pred_pose.dim() == 2:
        pred_pose = pred_pose.unsqueeze(1)

    components['translation'] = _extract_component(prediction, 'translation_norm', 'pred_pose_norm', slice(0, 3))
    components['qpos'] = _extract_component(prediction, 'qpos_norm', 'pred_pose_norm', slice(3, 19))
    components['rotation'] = prediction.get('rotation')
    if components['rotation'] is None and pred_pose is not None:
        components['rotation'] = pred_pose[..., 19:]

    if components['translation'] is None and pred_pose is not None:
        components['translation'] = pred_pose[..., 0:3]
    if components['qpos'] is None and pred_pose is not None:
        components['qpos'] = pred_pose[..., 3:19]

    return components


def _prepare_target_components(targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    components = {}
    target_pose = targets.get('norm_pose') if isinstance(targets, dict) else None
    if target_pose is not None and target_pose.dim() == 2:
        target_pose = target_pose.unsqueeze(1)

    components['translation'] = _extract_component(targets, 'translation_norm', 'norm_pose', slice(0, 3))
    components['qpos'] = _extract_component(targets, 'qpos_norm', 'norm_pose', slice(3, 19))
    components['rotation'] = targets.get('rotation')
    if components['rotation'] is None and target_pose is not None:
        components['rotation'] = target_pose[..., 19:]

    if components['translation'] is None and target_pose is not None:
        components['translation'] = target_pose[..., 0:3]
    if components['qpos'] is None and target_pose is not None:
        components['qpos'] = target_pose[..., 3:19]

    return components


def _ensure_three_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor is None:
        return tensor
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    return tensor


def pairwise_cost(
    prediction: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Dict[str, float],
    rot_type: str,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    if translation_weight > 0 and target_translation is not None:
        translation_cost = F.l1_loss(
            pred_translation.unsqueeze(2).expand(-1, -1, target_translation.shape[1], -1),
            target_translation.unsqueeze(1).expand(-1, pred_translation.shape[1], -1, -1),
            reduction='none'
        ).sum(dim=-1)
        cost = cost + translation_weight * translation_cost

    if qpos_weight > 0 and pred_qpos is not None and target_qpos is not None:
        qpos_cost = F.l1_loss(
            pred_qpos.unsqueeze(2).expand(-1, -1, target_qpos.shape[1], -1),
            target_qpos.unsqueeze(1).expand(-1, pred_qpos.shape[1], -1, -1),
            reduction='none'
        ).sum(dim=-1)
        cost = cost + qpos_weight * qpos_cost

    if rotation_weight > 0 and pred_rotation is not None and target_rotation is not None:
        if rot_type == 'quat':
            normalized_pred = F.normalize(pred_rotation, dim=-1)
            normalized_target = F.normalize(target_rotation, dim=-1)
            rotation_cost = 1.0 - torch.abs(
                torch.matmul(normalized_pred, normalized_target.transpose(-2, -1))
            )
        else:
            rotation_cost = F.l1_loss(
                pred_rotation.unsqueeze(2).expand(-1, -1, target_rotation.shape[1], -1),
                target_rotation.unsqueeze(1).expand(-1, pred_rotation.shape[1], -1, -1),
                reduction='none'
            ).sum(dim=-1)
        cost = cost + rotation_weight * rotation_cost

    if valid_mask is not None:
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        broadcast_mask = valid_mask.unsqueeze(1)
        inf_value = torch.finfo(cost.dtype).max
        cost = torch.where(broadcast_mask, cost, cost.new_full((), inf_value))

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
