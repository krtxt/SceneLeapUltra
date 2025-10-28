"""
Set-based grasp distance computation for set learning losses and metrics.

This module implements the core grasp element distance function c(g, g_tilde)
used in all set-based losses and metrics.

A single grasp g = (R, t, q) consists of:
- R ∈ SO(3): rotation
- t ∈ R^3: translation
- q ∈ R^{n_q}: joint angles
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraspSetDistance(nn.Module):
    """
    Computes distance between grasps for set-based learning.
    
    Distance formula:
    c(g, g') = α_t * ||t - t'||_2 / s_obj
               + α_R * d_SO(3)(R, R')
               + α_q * ||(q - q') / range(q)||_2
               + α_sym * min_{S∈S} d_SO(3)(R, R'S)
               + α_cnt * d_contact(g, g')
    """
    
    def __init__(self, distance_cfg):
        """
        Args:
            distance_cfg: Configuration dict with the following keys:
                - alpha_translation: Weight for translation distance (default: 1.0)
                - alpha_rotation: Weight for rotation distance (default: 1.0)
                - alpha_qpos: Weight for joint angle distance (default: 0.5)
                - alpha_symmetry: Weight for symmetry-aware rotation (default: 0.0)
                - alpha_contact: Weight for contact distance (default: 0.0)
                - rot_type: Rotation representation type (rot6d, quat, etc.)
                - normalize_translation: Whether to normalize by object scale (default: True)
                - qpos_range: Range for joint normalization (default: [0, 1])
                - use_symmetry: Whether to consider object symmetries (default: False)
                - symmetry_group: List of symmetry matrices if available (default: None)
        """
        super().__init__()
        
        self.alpha_t = distance_cfg.get('alpha_translation', 1.0)
        self.alpha_R = distance_cfg.get('alpha_rotation', 1.0)
        self.alpha_q = distance_cfg.get('alpha_qpos', 0.5)
        self.alpha_sym = distance_cfg.get('alpha_symmetry', 0.0)
        self.alpha_cnt = distance_cfg.get('alpha_contact', 0.0)
        
        self.rot_type = distance_cfg.get('rot_type', 'rot6d')
        self.normalize_translation = distance_cfg.get('normalize_translation', True)
        self.qpos_range = distance_cfg.get('qpos_range', [0.0, 1.0])
        self.use_symmetry = distance_cfg.get('use_symmetry', False)
        
        logging.info(
            f"GraspSetDistance initialized: α_t={self.alpha_t}, α_R={self.alpha_R}, "
            f"α_q={self.alpha_q}, α_sym={self.alpha_sym}, α_cnt={self.alpha_cnt}, "
            f"rot_type={self.rot_type}"
        )
        
    def forward(self, grasps1, grasps2, object_scale=None):
        """
        Compute pairwise distance matrix between two sets of grasps.
        
        Args:
            grasps1: Dict with keys 'translation', 'rotation', 'qpos'
                     Each tensor has shape [B, N1, D] or [N1, D]
            grasps2: Dict with keys 'translation', 'rotation', 'qpos'
                     Each tensor has shape [B, N2, D] or [N2, D]
            object_scale: Optional tensor [B,] or [B, 1] for translation normalization
            
        Returns:
            distance_matrix: Tensor of shape [B, N1, N2] or [N1, N2]
        """
        # Handle batch dimension
        has_batch = grasps1['translation'].dim() == 3
        
        if not has_batch:
            # Add batch dimension for unified processing
            grasps1 = {k: v.unsqueeze(0) for k, v in grasps1.items()}
            grasps2 = {k: v.unsqueeze(0) for k, v in grasps2.items()}
            if object_scale is not None and object_scale.dim() == 0:
                object_scale = object_scale.unsqueeze(0)
        
        B = grasps1['translation'].shape[0]
        N1 = grasps1['translation'].shape[1]
        N2 = grasps2['translation'].shape[1]
        device = grasps1['translation'].device
        
        # Initialize distance matrix
        distance_matrix = torch.zeros(B, N1, N2, device=device)
        
        # 1. Translation distance
        if self.alpha_t > 0:
            trans_dist = self._translation_distance(
                grasps1['translation'], grasps2['translation'], object_scale
            )
            distance_matrix += self.alpha_t * trans_dist
            
        # 2. Rotation distance
        if self.alpha_R > 0:
            rot_dist = self._rotation_distance(
                grasps1['rotation'], grasps2['rotation']
            )
            distance_matrix += self.alpha_R * rot_dist
            
        # 3. Joint angle distance
        if self.alpha_q > 0 and 'qpos' in grasps1 and 'qpos' in grasps2:
            qpos_dist = self._qpos_distance(
                grasps1['qpos'], grasps2['qpos']
            )
            distance_matrix += self.alpha_q * qpos_dist
            
        # 4. Symmetry-aware rotation (if enabled)
        if self.alpha_sym > 0 and self.use_symmetry:
            sym_dist = self._symmetry_distance(
                grasps1['rotation'], grasps2['rotation']
            )
            distance_matrix += self.alpha_sym * sym_dist
            
        # 5. Contact distance (if enabled)
        if self.alpha_cnt > 0:
            contact_dist = self._contact_distance(grasps1, grasps2)
            distance_matrix += self.alpha_cnt * contact_dist
        
        # Remove batch dimension if input didn't have it
        if not has_batch:
            distance_matrix = distance_matrix.squeeze(0)
            
        return distance_matrix
    
    def _translation_distance(self, trans1, trans2, object_scale=None):
        """
        Compute pairwise translation distance.
        
        Args:
            trans1: [B, N1, 3]
            trans2: [B, N2, 3]
            object_scale: Optional [B,] or [B, 1] for normalization
            
        Returns:
            dist: [B, N1, N2]
        """
        # Expand dimensions for broadcasting: [B, N1, 1, 3] and [B, 1, N2, 3]
        trans1_expanded = trans1.unsqueeze(2)  # [B, N1, 1, 3]
        trans2_expanded = trans2.unsqueeze(1)  # [B, 1, N2, 3]
        
        # Compute L2 distance
        dist = torch.norm(trans1_expanded - trans2_expanded, dim=-1)  # [B, N1, N2]
        
        # Normalize by object scale if provided
        if self.normalize_translation and object_scale is not None:
            # Reshape object_scale to [B, 1, 1] for broadcasting
            scale = object_scale.view(-1, 1, 1)
            dist = dist / (scale + 1e-8)
            
        return dist
    
    def _rotation_distance(self, rot1, rot2):
        """
        Compute geodesic distance on SO(3) between rotations.
        
        Args:
            rot1: [B, N1, rot_dim] - rotation representation
            rot2: [B, N2, rot_dim] - rotation representation
            
        Returns:
            dist: [B, N1, N2] - geodesic distances
        """
        if self.rot_type == 'quat':
            return self._quaternion_distance(rot1, rot2)
        elif self.rot_type == 'rot6d':
            return self._rot6d_distance(rot1, rot2)
        else:
            # Fallback to simple L2 distance
            logging.warning(f"Rotation type {self.rot_type} not specifically handled, using L2 distance")
            return self._generic_pairwise_distance(rot1, rot2)
    
    def _quaternion_distance(self, quat1, quat2):
        """
        Compute geodesic distance between quaternions.
        
        Distance formula: d = 2 * arccos(|<q1, q2>|)
        
        Args:
            quat1: [B, N1, 4]
            quat2: [B, N2, 4]
            
        Returns:
            dist: [B, N1, N2]
        """
        # Normalize quaternions
        quat1_norm = F.normalize(quat1, dim=-1)  # [B, N1, 4]
        quat2_norm = F.normalize(quat2, dim=-1)  # [B, N2, 4]
        
        # Compute dot products: [B, N1, N2]
        dot_products = torch.matmul(quat1_norm, quat2_norm.transpose(-2, -1))
        
        # Take absolute value (quaternions q and -q represent the same rotation)
        dot_products = torch.abs(dot_products)
        
        # Clamp to avoid numerical issues with arccos
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        
        # Geodesic distance
        dist = 2.0 * torch.acos(dot_products)
        
        return dist
    
    def _rot6d_distance(self, rot6d1, rot6d2):
        """
        Compute distance between 6D rotation representations.
        
        For now, we use Frobenius norm of the difference after converting to matrices.
        This could be improved to true geodesic distance.
        
        Args:
            rot6d1: [B, N1, 6]
            rot6d2: [B, N2, 6]
            
        Returns:
            dist: [B, N1, N2]
        """
        # Convert to rotation matrices
        from utils.rot6d import compute_rotation_matrix_from_ortho6d
        
        B, N1, _ = rot6d1.shape
        N2 = rot6d2.shape[1]
        
        # Flatten batch and num_grasps for conversion
        rot6d1_flat = rot6d1.view(B * N1, 6)
        rot6d2_flat = rot6d2.view(B * N2, 6)
        
        # Convert to rotation matrices [B*N, 3, 3]
        rotmat1_flat = compute_rotation_matrix_from_ortho6d(rot6d1_flat)
        rotmat2_flat = compute_rotation_matrix_from_ortho6d(rot6d2_flat)
        
        # Reshape back to [B, N, 3, 3]
        rotmat1 = rotmat1_flat.view(B, N1, 3, 3)
        rotmat2 = rotmat2_flat.view(B, N2, 3, 3)
        
        # Compute geodesic distance using rotation matrices
        # d_SO(3) = arccos((trace(R1^T R2) - 1) / 2)
        
        # Expand for pairwise computation: [B, N1, 1, 3, 3] and [B, 1, N2, 3, 3]
        rotmat1_exp = rotmat1.unsqueeze(2)  # [B, N1, 1, 3, 3]
        rotmat2_exp = rotmat2.unsqueeze(1)  # [B, 1, N2, 3, 3]
        
        # Compute R1^T @ R2 for all pairs
        # Using batched matrix multiplication
        relative_rot = torch.matmul(
            rotmat1_exp.transpose(-2, -1),  # [B, N1, 1, 3, 3]
            rotmat2_exp                      # [B, 1, N2, 3, 3]
        )  # [B, N1, N2, 3, 3]
        
        # Compute trace
        trace = relative_rot.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B, N1, N2]
        
        # Geodesic distance formula
        # Clamp to avoid numerical issues
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        dist = torch.acos(cos_angle)
        
        return dist
    
    def _qpos_distance(self, qpos1, qpos2):
        """
        Compute normalized joint angle distance.
        
        Args:
            qpos1: [B, N1, n_q]
            qpos2: [B, N2, n_q]
            
        Returns:
            dist: [B, N1, N2]
        """
        # Normalize by joint range
        qpos_range = self.qpos_range[1] - self.qpos_range[0]
        
        # Compute pairwise L2 distance
        dist = self._generic_pairwise_distance(qpos1, qpos2)
        
        # Normalize (divide by sqrt(n_q) to make it scale-independent)
        n_q = qpos1.shape[-1]
        dist = dist / (qpos_range * torch.sqrt(torch.tensor(n_q, device=dist.device)))
        
        return dist
    
    def _symmetry_distance(self, rot1, rot2):
        """
        Compute symmetry-aware rotation distance.
        
        For now, this is a placeholder. Full implementation would require
        symmetry group information.
        
        Args:
            rot1: [B, N1, rot_dim]
            rot2: [B, N2, rot_dim]
            
        Returns:
            dist: [B, N1, N2]
        """
        # Placeholder: return zero distance
        B, N1 = rot1.shape[:2]
        N2 = rot2.shape[1]
        device = rot1.device
        
        logging.debug("Symmetry distance requested but not fully implemented, returning zeros")
        return torch.zeros(B, N1, N2, device=device)
    
    def _contact_distance(self, grasps1, grasps2):
        """
        Compute contact-based distance between grasps.
        
        Placeholder for contact point or normal consistency distance.
        
        Args:
            grasps1: Dict of grasp components
            grasps2: Dict of grasp components
            
        Returns:
            dist: [B, N1, N2]
        """
        # Placeholder: return zero distance
        B = grasps1['translation'].shape[0]
        N1 = grasps1['translation'].shape[1]
        N2 = grasps2['translation'].shape[1]
        device = grasps1['translation'].device
        
        logging.debug("Contact distance requested but not fully implemented, returning zeros")
        return torch.zeros(B, N1, N2, device=device)
    
    def _generic_pairwise_distance(self, tensor1, tensor2):
        """
        Compute generic pairwise L2 distance between tensors.
        
        Args:
            tensor1: [B, N1, D]
            tensor2: [B, N2, D]
            
        Returns:
            dist: [B, N1, N2]
        """
        # Expand dimensions for broadcasting
        tensor1_exp = tensor1.unsqueeze(2)  # [B, N1, 1, D]
        tensor2_exp = tensor2.unsqueeze(1)  # [B, 1, N2, D]
        
        # Compute L2 distance
        dist = torch.norm(tensor1_exp - tensor2_exp, dim=-1)  # [B, N1, N2]
        
        return dist


def extract_grasp_components(norm_pose, rot_type='rot6d'):
    """
    Extract grasp components (translation, rotation, qpos) from normalized pose.
    
    Args:
        norm_pose: [B, N, 22] or [B, N, pose_dim] - normalized grasp poses
        rot_type: Rotation representation type
        
    Returns:
        Dict with keys 'translation', 'rotation', 'qpos'
    """
    # Standard pose format: [t (3), q (16), R (rot_dim)]
    # rot_dim depends on rot_type: rot6d=6, quat=4, euler=3
    # For rot6d: 3 + 16 + 6 = 25 (not 22!)
    translation = norm_pose[..., :3]
    qpos = norm_pose[..., 3:19]
    rotation = norm_pose[..., 19:]  # Flexible size based on actual data
    
    return {
        'translation': translation,
        'qpos': qpos,
        'rotation': rotation
    }


def compute_pairwise_grasp_distance(
    pred_poses, 
    target_poses, 
    distance_cfg, 
    object_scale=None,
    rot_type='rot6d'
):
    """
    Convenience function to compute distance matrix between predicted and target grasps.
    
    Args:
        pred_poses: [B, N, pose_dim] - predicted grasp poses
        target_poses: [B, M, pose_dim] - target grasp poses
        distance_cfg: Configuration dict for GraspSetDistance
        object_scale: Optional [B,] for normalization
        rot_type: Rotation representation type
        
    Returns:
        distance_matrix: [B, N, M] - pairwise distances
    """
    # Ensure distance_cfg has rot_type
    if 'rot_type' not in distance_cfg:
        distance_cfg['rot_type'] = rot_type
    
    # Initialize distance calculator
    distance_fn = GraspSetDistance(distance_cfg)
    
    # Extract grasp components
    pred_grasps = extract_grasp_components(pred_poses, rot_type)
    target_grasps = extract_grasp_components(target_poses, rot_type)
    
    # Compute distance matrix
    distance_matrix = distance_fn(pred_grasps, target_grasps, object_scale)
    
    return distance_matrix
