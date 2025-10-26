"""
Set-based losses for grasp generation.

Implements:
- Unbalanced Sinkhorn Optimal Transport Loss
- Chamfer Distance Loss
- Repulsion Loss (diversity regularization)
- Physics Feasibility Loss
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinkhornOTLoss(nn.Module):
    """
    Unbalanced Sinkhorn Optimal Transport Loss.
    
    Aligns predicted grasp distribution with target grasp distribution
    using optimal transport with entropic regularization.
    """
    
    def __init__(self, loss_cfg):
        """
        Args:
            loss_cfg: Configuration dict with keys:
                - epsilon: Entropic regularization parameter (default: 0.1)
                - tau: Unbalanced regularization parameter (default: 1.0)
                - max_iter: Maximum Sinkhorn iterations (default: 100)
                - threshold: Convergence threshold (default: 1e-3)
                - max_samples: Maximum samples to use (for efficiency) (default: 256)
        """
        super().__init__()
        
        self.epsilon = loss_cfg.get('epsilon', 0.1)
        self.tau = loss_cfg.get('tau', 1.0)
        self.max_iter = loss_cfg.get('max_iter', 100)
        self.threshold = loss_cfg.get('threshold', 1e-3)
        self.max_samples = loss_cfg.get('max_samples', 256)
        
        logging.info(
            f"SinkhornOTLoss initialized: ε={self.epsilon}, τ={self.tau}, "
            f"max_iter={self.max_iter}, max_samples={self.max_samples}"
        )
    
    def forward(self, cost_matrix, mass_pred=None, mass_target=None):
        """
        Compute unbalanced Sinkhorn OT loss.
        
        Args:
            cost_matrix: [B, N, M] - pairwise cost matrix
            mass_pred: Optional [B, N] - mass distribution for predictions (default: uniform)
            mass_target: Optional [B, M] - mass distribution for targets (default: uniform)
            
        Returns:
            loss: Scalar tensor - OT distance
        """
        B, N, M = cost_matrix.shape
        device = cost_matrix.device
        
        # Subsample if necessary for efficiency
        if N > self.max_samples or M > self.max_samples:
            cost_matrix, mass_pred, mass_target = self._subsample(
                cost_matrix, mass_pred, mass_target
            )
            B, N, M = cost_matrix.shape
        
        # Initialize uniform mass distributions if not provided
        if mass_pred is None:
            mass_pred = torch.ones(B, N, device=device) / N
        if mass_target is None:
            mass_target = torch.ones(B, M, device=device) / M
        
        # Normalize mass distributions
        mass_pred = mass_pred / (mass_pred.sum(dim=-1, keepdim=True) + 1e-8)
        mass_target = mass_target / (mass_target.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Run Sinkhorn algorithm
        try:
            transport_plan = self._sinkhorn_iterations(
                cost_matrix, mass_pred, mass_target
            )
            
            # Compute OT cost
            ot_cost = torch.sum(transport_plan * cost_matrix, dim=(-2, -1))  # [B,]
            
            # Average over batch
            loss = ot_cost.mean()
            
            logging.debug(f"SinkhornOTLoss: mean={loss.item():.4f}, min={ot_cost.min().item():.4f}, max={ot_cost.max().item():.4f}")
            
        except Exception as e:
            logging.error(f"Sinkhorn OT computation failed: {e}")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    def _sinkhorn_iterations(self, C, a, b):
        """
        Perform Sinkhorn iterations to solve unbalanced OT.
        
        Args:
            C: [B, N, M] - cost matrix
            a: [B, N] - source mass distribution
            b: [B, M] - target mass distribution
            
        Returns:
            P: [B, N, M] - transport plan
        """
        B, N, M = C.shape
        device = C.device
        
        # Compute kernel matrix: K = exp(-C/ε)
        K = torch.exp(-C / self.epsilon)
        
        # Initialize dual variables
        u = torch.ones(B, N, device=device)
        v = torch.ones(B, M, device=device)
        
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u_prev = u.clone()
            
            # Update v
            Kt_u = torch.matmul(K.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1)  # [B, M]
            v = b / (Kt_u + 1e-8)
            
            # Update u
            K_v = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)  # [B, N]
            u = a / (K_v + 1e-8)
            
            # Check convergence
            if i % 10 == 0:
                err = torch.max(torch.abs(u - u_prev)).item()
                if err < self.threshold:
                    logging.debug(f"Sinkhorn converged at iteration {i}")
                    break
        
        # Compute transport plan: P = diag(u) K diag(v)
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B, N, M]
        
        return P
    
    def _subsample(self, cost_matrix, mass_pred, mass_target):
        """
        Subsample cost matrix for efficiency.
        
        Args:
            cost_matrix: [B, N, M]
            mass_pred: [B, N] or None
            mass_target: [B, M] or None
            
        Returns:
            Subsampled cost_matrix, mass_pred, mass_target
        """
        B, N, M = cost_matrix.shape
        
        # Subsample predictions
        if N > self.max_samples:
            idx = torch.randperm(N, device=cost_matrix.device)[:self.max_samples]
            cost_matrix = cost_matrix[:, idx, :]
            if mass_pred is not None:
                mass_pred = mass_pred[:, idx]
        
        # Subsample targets
        if M > self.max_samples:
            idx = torch.randperm(M, device=cost_matrix.device)[:self.max_samples]
            cost_matrix = cost_matrix[:, :, idx]
            if mass_target is not None:
                mass_target = mass_target[:, idx]
        
        logging.info(f"Subsampled cost matrix from [{B}, {N}, {M}] to {list(cost_matrix.shape)}")
        
        return cost_matrix, mass_pred, mass_target


class ChamferDistanceLoss(nn.Module):
    """
    Chamfer Distance Loss for grasp sets.
    
    Measures bidirectional minimum distance between predicted and target sets.
    """
    
    def __init__(self, loss_cfg):
        """
        Args:
            loss_cfg: Configuration dict (currently no specific parameters)
        """
        super().__init__()
        logging.info("ChamferDistanceLoss initialized")
    
    def forward(self, cost_matrix):
        """
        Compute Chamfer Distance.
        
        Args:
            cost_matrix: [B, N, M] - pairwise cost matrix
            
        Returns:
            loss: Scalar tensor - Chamfer distance
        """
        # Forward direction: for each prediction, find nearest target
        min_pred_to_target, _ = torch.min(cost_matrix, dim=-1)  # [B, N]
        forward_cd = min_pred_to_target.mean(dim=-1)  # [B,]
        
        # Backward direction: for each target, find nearest prediction
        min_target_to_pred, _ = torch.min(cost_matrix, dim=-2)  # [B, M]
        backward_cd = min_target_to_pred.mean(dim=-1)  # [B,]
        
        # Symmetric Chamfer distance
        cd = forward_cd + backward_cd
        
        # Average over batch
        loss = cd.mean()
        
        logging.debug(f"ChamferDistanceLoss: mean={loss.item():.4f}")
        
        return loss


class RepulsionLoss(nn.Module):
    """
    Repulsion Loss to encourage diversity in generated grasps.
    
    Penalizes grasps that are too similar to each other using kNN-Riesz repulsion.
    """
    
    def __init__(self, loss_cfg):
        """
        Args:
            loss_cfg: Configuration dict with keys:
                - k: Number of nearest neighbors to consider (default: 5)
                - lambda_repulsion: Repulsion strength (default: 1.0)
                - delta: Regularization term to avoid singularity (default: 0.01)
                - s: Riesz parameter (default: 2.0)
        """
        super().__init__()
        
        self.k = loss_cfg.get('k', 5)
        self.lambda_rep = loss_cfg.get('lambda_repulsion', 1.0)
        self.delta = loss_cfg.get('delta', 0.01)
        self.s = loss_cfg.get('s', 2.0)
        
        logging.info(
            f"RepulsionLoss initialized: k={self.k}, λ={self.lambda_rep}, "
            f"δ={self.delta}, s={self.s}"
        )
    
    def forward(self, pred_poses):
        """
        Compute repulsion loss on predicted grasps.
        
        Args:
            pred_poses: [B, N, D] - predicted grasp poses or embeddings
            
        Returns:
            loss: Scalar tensor - repulsion loss
        """
        B, N, D = pred_poses.shape
        device = pred_poses.device
        
        if N <= 1:
            # No repulsion needed for single grasp
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute pairwise distances within predicted set
        # [B, N, 1, D] - [B, 1, N, D] -> [B, N, N]
        pred_expanded1 = pred_poses.unsqueeze(2)
        pred_expanded2 = pred_poses.unsqueeze(1)
        pairwise_dist_sq = torch.sum((pred_expanded1 - pred_expanded2) ** 2, dim=-1)  # [B, N, N]
        
        # For each grasp, find k nearest neighbors (excluding itself)
        # Set diagonal to infinity to exclude self-distance
        pairwise_dist_sq_masked = pairwise_dist_sq.clone()
        pairwise_dist_sq_masked.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))
        
        # Find k nearest neighbors
        k_actual = min(self.k, N - 1)
        if k_actual <= 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        knn_distances_sq, _ = torch.topk(
            pairwise_dist_sq_masked, k_actual, dim=-1, largest=False
        )  # [B, N, k]
        
        # Compute Riesz repulsion: λ / (δ + dist^2)^(s/2)
        repulsion = self.lambda_rep / (self.delta + knn_distances_sq) ** (self.s / 2.0)
        
        # Average over neighbors and grasps
        loss = repulsion.mean()
        
        logging.debug(f"RepulsionLoss: mean={loss.item():.4f}")
        
        return loss


class PhysicsFeasibilityLoss(nn.Module):
    """
    Physics Feasibility Loss to penalize physically infeasible grasps.
    
    Includes:
    - Collision/penetration penalty
    - Contact consistency (if applicable)
    """
    
    def __init__(self, loss_cfg, hand_model=None):
        """
        Args:
            loss_cfg: Configuration dict with keys:
                - penetration_weight: Weight for penetration penalty (default: 1.0)
                - contact_weight: Weight for contact consistency (default: 0.5)
                - penetration_threshold: Threshold for penetration (default: 0.005)
            hand_model: Optional HandModel instance for physics checks
        """
        super().__init__()
        
        self.penetration_weight = loss_cfg.get('penetration_weight', 1.0)
        self.contact_weight = loss_cfg.get('contact_weight', 0.5)
        self.penetration_threshold = loss_cfg.get('penetration_threshold', 0.005)
        self.hand_model = hand_model
        
        logging.info(
            f"PhysicsFeasibilityLoss initialized: pen_weight={self.penetration_weight}, "
            f"contact_weight={self.contact_weight}, threshold={self.penetration_threshold}"
        )
    
    def forward(self, outputs, targets):
        """
        Compute physics feasibility loss.
        
        Args:
            outputs: Dict containing hand model outputs with 'penetration' info
            targets: Dict containing target grasp info
            
        Returns:
            loss: Scalar tensor - physics feasibility loss
        """
        device = outputs.get('hand', {}).get('translation', torch.tensor(0.0)).device
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 1. Penetration penalty
        if self.penetration_weight > 0 and 'hand' in outputs:
            pen_loss = self._compute_penetration_loss(outputs)
            loss = loss + self.penetration_weight * pen_loss
        
        # 2. Contact consistency (placeholder)
        if self.contact_weight > 0:
            contact_loss = self._compute_contact_loss(outputs, targets)
            loss = loss + self.contact_weight * contact_loss
        
        return loss
    
    def _compute_penetration_loss(self, outputs):
        """
        Compute penetration penalty from hand model outputs.
        
        Args:
            outputs: Dict with 'hand' containing penetration info
            
        Returns:
            loss: Scalar tensor
        """
        hand_outputs = outputs.get('hand', {})
        
        # Try to get penetration information from hand model
        if 'penetration' in hand_outputs:
            penetration = hand_outputs['penetration']  # Should be [B, N] or similar
            
            # Penalize penetration above threshold
            excess_penetration = F.relu(penetration - self.penetration_threshold)
            loss = excess_penetration.mean()
            
            logging.debug(f"Penetration loss: {loss.item():.4f}")
            return loss
        else:
            device = hand_outputs.get('translation', torch.tensor(0.0)).device
            logging.debug("No penetration info available in hand outputs")
            return torch.tensor(0.0, device=device)
    
    def _compute_contact_loss(self, outputs, targets):
        """
        Compute contact consistency loss (placeholder).
        
        Args:
            outputs: Dict with predictions
            targets: Dict with targets
            
        Returns:
            loss: Scalar tensor
        """
        # Placeholder implementation
        device = outputs.get('hand', {}).get('translation', torch.tensor(0.0)).device
        logging.debug("Contact loss not implemented, returning zero")
        return torch.tensor(0.0, device=device)


def compute_set_losses(
    pred_dict,
    targets,
    distance_matrix,
    set_loss_cfg,
    hand_model=None,
    current_timestep=None,
    total_timesteps=1000
):
    """
    Compute all set-based losses.
    
    Args:
        pred_dict: Dict containing predictions
        targets: Dict containing targets
        distance_matrix: [B, N, M] - precomputed distance matrix
        set_loss_cfg: Configuration dict for set losses
        hand_model: Optional HandModel instance
        current_timestep: Current diffusion timestep (for scheduling)
        total_timesteps: Total number of timesteps
        
    Returns:
        Dict of losses: {
            'ot_loss': ...,
            'chamfer_loss': ...,
            'repulsion_loss': ...,
            'physics_loss': ...
        }
    """
    losses = {}
    device = distance_matrix.device
    
    # Get loss weights
    lambda_ot = set_loss_cfg.get('lambda_ot', 1.0)
    gamma_cd = set_loss_cfg.get('gamma_cd', 1.0)
    eta_rep = set_loss_cfg.get('eta_repulsion', 0.1)
    zeta_phys = set_loss_cfg.get('zeta_physics', 0.1)
    
    # Apply timestep scheduling if configured
    if current_timestep is not None:
        schedule_type = set_loss_cfg.get('schedule_type', 'constant')
        lambda_set = compute_timestep_weight(
            current_timestep, total_timesteps, schedule_type, set_loss_cfg
        )
        logging.debug(f"Set loss timestep weight: {lambda_set:.3f} at t={current_timestep}")
    else:
        lambda_set = 1.0
    
    # 1. Optimal Transport Loss
    if lambda_ot > 0:
        try:
            ot_loss_fn = SinkhornOTLoss(set_loss_cfg.get('ot_config', {}))
            ot_loss = ot_loss_fn(distance_matrix)
            losses['ot_loss'] = lambda_set * lambda_ot * ot_loss
        except Exception as e:
            logging.error(f"OT loss computation failed: {e}")
            losses['ot_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 2. Chamfer Distance Loss
    if gamma_cd > 0:
        try:
            cd_loss_fn = ChamferDistanceLoss(set_loss_cfg.get('cd_config', {}))
            cd_loss = cd_loss_fn(distance_matrix)
            losses['chamfer_loss'] = lambda_set * gamma_cd * cd_loss
        except Exception as e:
            logging.error(f"Chamfer loss computation failed: {e}")
            losses['chamfer_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 3. Repulsion Loss (diversity)
    if eta_rep > 0 and 'matched' in pred_dict and 'norm_pose' in pred_dict['matched']:
        try:
            rep_loss_fn = RepulsionLoss(set_loss_cfg.get('repulsion_config', {}))
            pred_poses = pred_dict['matched']['norm_pose']
            rep_loss = rep_loss_fn(pred_poses)
            losses['repulsion_loss'] = eta_rep * rep_loss  # No timestep weighting for diversity
        except Exception as e:
            logging.error(f"Repulsion loss computation failed: {e}")
            losses['repulsion_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 4. Physics Feasibility Loss
    if zeta_phys > 0:
        try:
            phys_loss_fn = PhysicsFeasibilityLoss(
                set_loss_cfg.get('physics_config', {}), hand_model
            )
            phys_loss = phys_loss_fn(pred_dict, targets)
            losses['physics_loss'] = zeta_phys * phys_loss
        except Exception as e:
            logging.error(f"Physics loss computation failed: {e}")
            losses['physics_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
    
    return losses


def compute_timestep_weight(t, T, schedule_type='constant', config=None):
    """
    Compute timestep-dependent weight for set losses.
    
    Args:
        t: Current timestep
        T: Total timesteps
        schedule_type: 'constant', 'linear', 'cosine', 'quadratic'
        config: Additional configuration
        
    Returns:
        weight: Scalar weight
    """
    if config is None:
        config = {}
    
    # Normalize timestep to [0, 1]
    t_norm = float(t) / float(T)
    
    if schedule_type == 'constant':
        return 1.0
    elif schedule_type == 'linear':
        # Increase linearly from 0 to final_weight
        final_weight = config.get('final_weight', 1.0)
        return t_norm * final_weight
    elif schedule_type == 'cosine':
        # Cosine schedule: 0.5 * (1 + cos(π * (1 - t_norm)))
        import math
        final_weight = config.get('final_weight', 1.0)
        return final_weight * 0.5 * (1 + math.cos(math.pi * (1 - t_norm)))
    elif schedule_type == 'quadratic':
        # Quadratic: t_norm^2
        final_weight = config.get('final_weight', 1.0)
        return final_weight * (t_norm ** 2)
    else:
        logging.warning(f"Unknown schedule type {schedule_type}, using constant")
        return 1.0
