import logging
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from torch.functional import Tensor

# ==================== Utility Functions ====================

def huber_loss(error, delta=1.0):
    """
    Huber loss function for robust regression

    Reference: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py

    Formula:
    - 0.5 * |x|^2                 if |x| <= delta
    - 0.5 * delta^2 + delta * (|x| - delta)     if |x| > delta

    Args:
        error: Error tensor (pred - gt or dist(pred, gt))
        delta: Huber loss threshold

    Returns:
        Huber loss tensor
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss

def _get_regression_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Calculate regression loss using Huber loss"""
    loss = huber_loss(prediction - target).sum(-1).mean()
    return loss

# ==================== Rotation-Specific Loss Functions ====================

def calculate_euler_loss(prediction, target) -> Tensor:
    """Calculate Euler angle rotation loss"""
    error = (prediction - target).abs().sum(-1).mean()
    return error

def calculate_quat_loss(prediction_raw: Tensor, target_unit: Tensor) -> Tensor:
    """
    Calculate quaternion rotation loss measuring angular difference
    
    Args:
        prediction_raw: Raw quaternion predictions of size (K, 4) - NOT unit quaternions
        target_unit: Target unit quaternions of size (K, 4) - ARE unit quaternions
    """
    # Normalize raw predictions to unit quaternions
    epsilon = 1e-8  # Numerical stability
    prediction_norm = torch.linalg.norm(prediction_raw, dim=-1, keepdim=True)
    normalized_prediction = prediction_raw / (prediction_norm + epsilon)

    # Calculate dot product between normalized prediction and unit target
    # For unit quaternions p and q, p·q = cos(theta/2) where theta is rotation angle
    dot_product = (normalized_prediction * target_unit).sum(dim=-1)

    # Calculate loss: 1 - |dot_product|
    # abs() handles q and -q representing same rotation
    # Loss range [0, 1]: 0 = perfect alignment, 1 = 90° difference in 4D space
    rotation_loss = 1.0 - dot_product.abs()
    mean_rotation_loss = rotation_loss.mean()

    return mean_rotation_loss

def calculate_axis_loss(prediction, target) -> Tensor:
    """Calculate axis-angle rotation loss"""
    return _get_regression_loss(prediction, target)

def calculate_r6d_loss(prediction, target) -> Tensor:
    """Calculate 6D rotation representation loss"""
    return _get_regression_loss(prediction, target)

# ==================== Main Loss Functions ====================

def calculate_latent_loss(prediction, target) -> Tensor:
    """Calculate latent space KL divergence loss for variational models"""
    log_var_tensor, means_tensor = None, None

    # Try to get log_var and means from prediction['matched'] (training path)
    if 'matched' in prediction and isinstance(prediction['matched'], dict):
        log_var_tensor = prediction['matched'].get('log_var')
        means_tensor = prediction['matched'].get('means')

    # Fallback to top-level prediction (CVAE validation path)
    if log_var_tensor is None and 'log_var' in prediction:
        log_var_tensor = prediction.get('log_var')
    if means_tensor is None and 'means' in prediction:
        means_tensor = prediction.get('means')

    # Determine device for zero tensor fallback
    tensor_for_device = None
    if 'matched' in target and isinstance(target['matched'], dict) and 'norm_pose' in target['matched']:
        tensor_for_device = target['matched']['norm_pose']
    elif 'norm_pose' in target:
        tensor_for_device = target['norm_pose']

    if log_var_tensor is None or means_tensor is None:
        expected_device = torch.device('cpu')
        if tensor_for_device is not None and hasattr(tensor_for_device, 'device'):
            expected_device = tensor_for_device.device
        
        logging.warning(f"Missing 'log_var' or 'means' for latent loss calculation. Returning 0 KLD loss on device {expected_device}.")
        return torch.tensor(0.0, device=expected_device)

    # Calculate KL divergence loss
    kld = -0.5 * torch.sum(1 + log_var_tensor - means_tensor.pow(2) - log_var_tensor.exp()) / log_var_tensor.shape[0]
    return kld

def calculate_para_loss(prediction, target, loss_aggregation='mean') -> Tensor:
    """Calculate parameter reconstruction loss with multi-grasp support"""
    pred_para = prediction['matched']['pred_pose_norm']
    para = target['matched']['norm_pose']

    # Handle multi-grasp format
    if para.dim() == 3:
        # Multi-grasp format: [B, num_grasps, pose_dim]
        B, num_grasps, pose_dim = para.shape
        mse_loss = F.mse_loss(pred_para, para, reduction='none')  # [B, num_grasps, pose_dim]

        # Apply aggregation strategy
        if loss_aggregation == 'mean':
            para_loss = mse_loss.mean()
        elif loss_aggregation == 'sum':
            para_loss = mse_loss.sum() / B  # Normalize by batch size
        elif loss_aggregation == 'weighted':
            # Use grasp weights if available
            weights = target.get('grasp_weights', torch.ones(B, num_grasps, device=para.device))
            weighted_loss = mse_loss.mean(dim=-1) * weights  # [B, num_grasps]
            para_loss = weighted_loss.sum() / weights.sum()
        else:
            para_loss = mse_loss.mean()
    else:
        # Single grasp format (backward compatibility)
        para_loss = F.mse_loss(pred_para, para)

    return para_loss

def calculate_noise_loss(prediction, target, loss_aggregation='mean') -> Tensor:
    """Calculate noise prediction loss for denoising models with multi-grasp support"""
    pred_noise = prediction['matched']['pred_noise']
    noise = prediction['matched']['noise']

    # Handle multi-grasp format
    if noise.dim() == 3:
        # Multi-grasp format: [B, num_grasps, pose_dim]
        B, num_grasps, pose_dim = noise.shape
        mse_loss = F.mse_loss(pred_noise, noise, reduction='none')  # [B, num_grasps, pose_dim]

        # Apply aggregation strategy
        if loss_aggregation == 'mean':
            noise_loss = mse_loss.mean()
        elif loss_aggregation == 'sum':
            noise_loss = mse_loss.sum() / B  # Normalize by batch size
        elif loss_aggregation == 'weighted':
            # Use grasp weights if available
            weights = target.get('grasp_weights', torch.ones(B, num_grasps, device=noise.device))
            weighted_loss = mse_loss.mean(dim=-1) * weights  # [B, num_grasps]
            noise_loss = weighted_loss.sum() / weights.sum()
        else:
            noise_loss = mse_loss.mean()
    else:
        # Single grasp format (backward compatibility)
        noise_loss = F.mse_loss(pred_noise, noise)

    return noise_loss

def calculate_translation_loss(prediction, target, loss_aggregation='mean') -> Tensor:
    """Calculate translation parameter reconstruction loss with multi-grasp support"""
    pred = prediction['matched']['pred_pose_norm'][...,:3]
    target_trans = target['matched']['norm_pose'][...,:3]

    # Handle multi-grasp format
    if target_trans.dim() == 3:
        # Multi-grasp format: [B, num_grasps, 3]
        B, num_grasps, _ = target_trans.shape
        mse_loss = F.mse_loss(pred, target_trans, reduction='none')  # [B, num_grasps, 3]

        # Apply aggregation strategy
        if loss_aggregation == 'mean':
            translation_loss = mse_loss.mean()
        elif loss_aggregation == 'sum':
            translation_loss = mse_loss.sum() / B
        elif loss_aggregation == 'weighted':
            weights = target.get('grasp_weights', torch.ones(B, num_grasps, device=target_trans.device))
            weighted_loss = mse_loss.mean(dim=-1) * weights  # [B, num_grasps]
            translation_loss = weighted_loss.sum() / weights.sum()
        else:
            translation_loss = mse_loss.mean()
    else:
        # Single grasp format (backward compatibility)
        translation_loss = _get_regression_loss(pred, target_trans)

    return translation_loss

def calculate_qpos_loss(prediction, target, loss_aggregation='mean') -> Tensor:
    """Calculate joint position reconstruction loss with multi-grasp support"""
    pred = prediction['matched']['pred_pose_norm'][...,3:19]
    target_qpos = target['matched']['norm_pose'][...,3:19]

    # Handle multi-grasp format
    if target_qpos.dim() == 3:
        # Multi-grasp format: [B, num_grasps, 16]
        B, num_grasps, _ = target_qpos.shape
        mse_loss = F.mse_loss(pred, target_qpos, reduction='none')  # [B, num_grasps, 16]

        # Apply aggregation strategy
        if loss_aggregation == 'mean':
            qpos_loss = mse_loss.mean()
        elif loss_aggregation == 'sum':
            qpos_loss = mse_loss.sum() / B
        elif loss_aggregation == 'weighted':
            weights = target.get('grasp_weights', torch.ones(B, num_grasps, device=target_qpos.device))
            weighted_loss = mse_loss.mean(dim=-1) * weights  # [B, num_grasps]
            qpos_loss = weighted_loss.sum() / weights.sum()
        else:
            qpos_loss = mse_loss.mean()
    else:
        # Single grasp format (backward compatibility)
        qpos_loss = _get_regression_loss(pred, target_qpos)

    return qpos_loss

def calculate_rotation_loss(prediction, target, loss_aggregation='mean') -> Tensor:
    """Calculate rotation loss based on rotation type with multi-grasp support"""
    rot_type = prediction['rot_type']
    pred = prediction['matched']['pred_pose_norm'][...,19:]
    target_rot = target['matched']['norm_pose'][...,19:]

    loss_method_map = {
        'euler': calculate_euler_loss,
        'quat': calculate_quat_loss,
        'axis': calculate_axis_loss,
        'r6d': calculate_r6d_loss,
    }

    if rot_type in loss_method_map:
        loss_method = loss_method_map[rot_type]

        # Handle multi-grasp format
        if target_rot.dim() == 3:
            # Multi-grasp format: [B, num_grasps, rot_dim]
            B, num_grasps, rot_dim = target_rot.shape

            # Flatten for rotation loss calculation
            pred_flat = pred.reshape(B * num_grasps, rot_dim)
            target_flat = target_rot.reshape(B * num_grasps, rot_dim)

            # Calculate rotation loss
            rotation_loss = loss_method(pred_flat, target_flat)

            # Apply aggregation strategy
            if loss_aggregation == 'mean':
                # Loss is already averaged over flattened dimension
                pass
            elif loss_aggregation == 'sum':
                rotation_loss = rotation_loss * num_grasps / B  # Scale up并除以批大小，保持与其他 loss 一致
            elif loss_aggregation == 'weighted':
                # For rotation loss, we'll use mean aggregation as weights are complex to apply
                pass

            return rotation_loss
        else:
            # Single grasp format (backward compatibility)
            loss = loss_method(pred, target_rot)
            return loss
    else:
        raise NotImplementedError(f"Unable to calculate {rot_type} loss.")

def calculate_hand_chamfer_loss(prediction, target) -> Tensor:
    """Calculate Chamfer distance loss between predicted and target hand point clouds"""
    pred_hand_pc = prediction['hand']['surface_points']
    target_hand_pc = target['hand']['surface_points']
    chamfer_loss = chamfer_distance(
        pred_hand_pc, 
        target_hand_pc, 
        point_reduction="sum", 
        batch_reduction="mean"
    )[0]
    return chamfer_loss

def calculate_obj_penetration_loss(prediction, target) -> Tensor:
    """Calculate object penetration loss"""
    batch_size = prediction['hand']['penetration_keypoints'].size(0)
    # Signed squared distances from object_pc to hand, inside positive, outside negative
    distances = prediction['hand']['penetration']
    # Penetration loss - only penalize positive distances (inside object)
    loss_pen = distances[distances > 0].sum() / batch_size
    return loss_pen

def calculate_self_penetration_loss(prediction, target) -> Tensor:
    """Calculate self-penetration loss to prevent hand self-intersection"""
    batch_size = prediction['hand']['penetration_keypoints'].size(0)
    penetration_keypoints = prediction['hand']['penetration_keypoints']
    
    # Calculate pairwise distances between penetration keypoints
    dis_spen = (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    # Avoid self-distance (set very large value for same points)
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    # Penalize distances smaller than threshold (0.02)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    loss_spen = dis_spen.sum() / batch_size
    return loss_spen

def calculate_distance_loss(prediction, target, thres_dis=0.01) -> Tensor:
    """Calculate contact distance loss"""
    dis_pred = prediction['hand']['contact_candidates_dis']
    small_dis_pred = dis_pred < thres_dis ** 2
    loss_dis = dis_pred[small_dis_pred].sum() / dis_pred.size(0)
    return loss_dis

def calculate_cmap_loss(prediction, target, normalize_factor=200) -> Tensor:
    """Calculate contact map loss"""
    # Calculate contact maps using sigmoid normalization
    cmap = 2 - 2 * torch.sigmoid(normalize_factor * (target["hand"]['distances'].abs() + 1e-4).sqrt())
    cmap_pred = 2 - 2 * torch.sigmoid(normalize_factor * (prediction["hand"]['distances'].abs() + 1e-4).sqrt())
    loss_cmap = torch.nn.functional.mse_loss(cmap, cmap_pred, reduction='sum') / cmap.size(0)
    return loss_cmap

# ==================== Multi-Grasp Loss Functions ====================

def calculate_consistency_loss(prediction, target) -> Tensor:
    """
    Calculate grasp consistency loss to encourage similar grasps for similar scene regions
    """
    pred_poses = prediction['matched']['pred_pose_norm']
        
    # Only apply to multi-grasp format
    if pred_poses.dim() != 3:
        logging.warning(f'[Consistency Loss] Expected 3D tensor, got {pred_poses.dim()}D tensor')
        return torch.tensor(0.0, device=pred_poses.device)

    B, num_grasps, pose_dim = pred_poses.shape

    # Calculate pairwise similarities between grasps
    consistency_loss = torch.tensor(0.0, device=pred_poses.device)
    for b in range(B):
        pred_b = pred_poses[b]  # [num_grasps, pose_dim]

        # Calculate pairwise distances between grasps
        distances = torch.cdist(pred_b, pred_b, p=2)  # [num_grasps, num_grasps]

        # Exclude diagonal elements (self-distance)
        mask = ~torch.eye(num_grasps, dtype=torch.bool, device=distances.device)
        distances_masked = distances[mask]

        # Consistency loss: encourage moderate diversity (not too similar, not too different)
        # Target distance around 0.5 (adjust based on pose normalization)
        target_distance = 0.5
        consistency_loss += F.mse_loss(distances_masked,
                                     torch.full_like(distances_masked, target_distance))

    consistency_loss = consistency_loss / B
    return consistency_loss

def calculate_diversity_loss(prediction, target) -> Tensor:
    """
    Calculate diversity loss to encourage diverse grasp predictions
    """
    pred_poses = prediction['matched']['pred_pose_norm']

    # Only apply to multi-grasp format
    if pred_poses.dim() != 3:
        logging.warning(f'[Diversity Loss] Expected 3D tensor, got {pred_poses.dim()}D tensor')
        return torch.tensor(0.0, device=pred_poses.device)

    B, num_grasps, pose_dim = pred_poses.shape

    diversity_scores = []
    for b in range(B):
        pred_b = pred_poses[b]  # [num_grasps, pose_dim]

        # Calculate pairwise distances
        distances = torch.cdist(pred_b, pred_b, p=2)  # [num_grasps, num_grasps]

        # Exclude diagonal elements
        mask = ~torch.eye(num_grasps, dtype=torch.bool, device=distances.device)
        distances_masked = distances[mask]

        # Diversity score: negative of mean distance (we want to maximize diversity)
        diversity_score = -distances_masked.mean()
        diversity_scores.append(diversity_score)

    diversity_loss = torch.stack(diversity_scores).mean()
    return diversity_loss 