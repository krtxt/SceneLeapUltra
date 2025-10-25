from __future__ import annotations

"""utils.hand_physics
Physics/geometry energy calculation module.
"""

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torchsdf import compute_sdf

if TYPE_CHECKING:  # Avoid runtime circular dependency
    from utils.hand_model import HandModel


class HandPhysics:
    """Hand physics and geometry calculations."""

    def __init__(self, model: "HandModel") -> None:
        self._model = model

    def cal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance from points x to hand model."""
        model = self._model
        if model.global_translation is None or model.global_rotation is None or model.current_status is None:
            raise ValueError("Hand parameters are not set. Call set_parameters() first.")
        if x.shape[0] != model.batch_size:
            raise ValueError("Input batch size mismatch.")

        dis = []
        # Unified multi-grasp format processing
        # global_rotation: (B, num_grasps, 3, 3), global_translation: (B, num_grasps, 3)
        # Flatten to match x's batch dimension
        global_rotation_flat = model.global_rotation.view(-1, 3, 3)  # (B*num_grasps, 3, 3)
        global_translation_flat = model.global_translation.view(-1, 3)  # (B*num_grasps, 3)
        x_transformed = (x - global_translation_flat.unsqueeze(1)) @ global_rotation_flat

        for link_name in model.mesh:
            matrix = model.current_status[link_name].get_matrix()
            x_local = (x_transformed - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local_flat = x_local.reshape(-1, 3)

            if "radius" not in model.mesh[link_name]:
                assert "face_verts" in model.mesh[link_name]
                face_verts = model.mesh[link_name]["face_verts"]
                dis_local_val, dis_signs, _, _ = compute_sdf(x_local_flat, face_verts)
                dis_local_val = torch.sqrt(dis_local_val + 1e-8)
                dis_local_signed = dis_local_val * (-dis_signs)
            else:
                dis_local_signed = model.mesh[link_name]["radius"] - x_local_flat.norm(dim=1)

            dis.append(dis_local_signed.reshape(x.shape[0], x.shape[1]))

        return torch.max(torch.stack(dis, dim=0), dim=0)[0]

    def compute_object_penetration(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate signed distance (penetration) from object point cloud `x` to hand model.
        Positive values indicate points inside the model.
        """
        model = self._model
        if model.global_translation is None or model.global_rotation is None or model.current_status is None:
            raise ValueError("Hand parameters are not set. Call set_parameters() first.")
        
        expected_batch_size = model.hand_pose.shape[0]

        if x.ndim == 2:
            if expected_batch_size == 1:
                x_batched = x.unsqueeze(0)
            else:
                raise ValueError("Input x batch size mismatch.")
        elif x.ndim == 3:
            if x.shape[0] != expected_batch_size:
                raise ValueError("Input x batch size mismatch.")
            x_batched = x
        else:
            raise ValueError("Input x must have ndim 2 or 3.")

        if x_batched.shape[-1] != 3:
            raise ValueError("Last dim of x must be 3 (xyz).")

        dis_list = []

        # Unified multi-grasp format processing
        # global_rotation: (B, num_grasps, 3, 3), global_translation: (B, num_grasps, 3)
        # Flatten to match x_batched's batch dimension
        global_rotation_flat = model.global_rotation.view(-1, 3, 3)  # (B*num_grasps, 3, 3)
        global_translation_flat = model.global_translation.view(-1, 3)  # (B*num_grasps, 3)
        x_in_hand_frame = (x_batched - global_translation_flat.unsqueeze(1)) @ global_rotation_flat
        
        for link_name in model.mesh:
            matrix = model.current_status[link_name].get_matrix()
            x_local = (x_in_hand_frame - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local_flat = x_local.reshape(-1, 3)
            
            if "radius" not in model.mesh[link_name]:
                assert "face_verts" in model.mesh[link_name]
                face_verts = model.mesh[link_name]["face_verts"]
                dis_local_values, dis_signs, _, _ = compute_sdf(x_local_flat, face_verts)
                dis_local_values = torch.sqrt(dis_local_values + 1e-8)
                dis_local_signed = dis_local_values * (-dis_signs)
            else:
                dis_local_signed = model.mesh[link_name]["radius"] - x_local_flat.norm(dim=1)
            
            dis_list.append(dis_local_signed.reshape(x_batched.shape[0], x_batched.shape[1]))
        
        return torch.max(torch.stack(dis_list, dim=0), dim=0)[0]

    def cal_self_penetration_energy(self) -> torch.Tensor:
        """Calculate self-collision energy. Handles unified multi-grasp format."""
        model = self._model
        # Use flattened batch size
        batch_size = model.batch_size  # B * num_grasps

        points = model.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = model.global_index_to_link_index_penetration.clone().repeat(batch_size, 1)

        transforms = torch.zeros(batch_size, model.n_keypoints, 4, 4, dtype=torch.float, device=model.device)

        for link_name in model.mesh:
            mask = link_indices == model.link_name_to_link_index[link_name]
            if not mask.any():
                continue

            current_transform = model.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, model.n_keypoints, 4, 4)
            transforms[mask] = current_transform[mask]

        points_homog = torch.cat([
            points,
            torch.ones(batch_size, model.n_keypoints, 1, dtype=torch.float, device=model.device)
        ], dim=2)

        points_transformed = (transforms @ points_homog.unsqueeze(3))[:, :, :3, 0]

        # Unified multi-grasp format processing
        # global_rotation: (B, num_grasps, 3, 3), global_translation: (B, num_grasps, 3)
        # Flatten to match points_transformed's batch dimension
        global_rotation_flat = model.global_rotation.view(-1, 3, 3)  # (B*num_grasps, 3, 3)
        global_translation_flat = model.global_translation.view(-1, 3)  # (B*num_grasps, 3)
        points_world = points_transformed @ global_rotation_flat.transpose(1, 2) + global_translation_flat.unsqueeze(1)
        
        dis = torch.cdist(points_world, points_world, p=2)
        dis.diagonal(dim1=1, dim2=2).fill_(1e6)  # Ignore self-distance

        # Import constant from hand_constants
        from utils.hand_constants import SELF_PENETRATION_POINT_RADIUS
        spen = (SELF_PENETRATION_POINT_RADIUS * 2 - dis)
        E_spen = torch.where(spen > 0, spen, torch.zeros_like(spen))
        
        return E_spen.sum((1, 2)) / 2  # Each distance pair is counted twice, so divide by 2

    def cal_joint_limit_energy(self) -> torch.Tensor:
        """Calculate joint limit energy."""
        model = self._model
        # Use slice constant
        from utils.hand_constants import QPOS_SLICE
        joint_angles = model.hand_pose[:, QPOS_SLICE]

        upper_violations = (joint_angles - model.joints_upper).clamp(min=0)
        lower_violations = (model.joints_lower - joint_angles).clamp(min=0)

        return torch.sum(upper_violations + lower_violations, dim=-1)

    def cal_finger_finger_distance_energy(self) -> torch.Tensor:
        """Calculate finger-finger distance energy (maximize inter-finger distance)."""
        model = self._model
        if model.contact_points is None:
            return torch.zeros(model.batch_size, device=model.device)
            
        # Negative sum of distances, optimization will maximize this distance
        return -torch.pdist(model.contact_points, p=2).sum(dim=-1)

    def cal_finger_palm_distance_energy(self) -> torch.Tensor:
        """Calculate finger-palm distance energy (maximize this distance)."""
        model = self._model
        if model.contact_points is None:
            return torch.zeros(model.batch_size, device=model.device)
            
        palm_position = model.global_translation.unsqueeze(1)
        # Similarly, negative sum of distances
        return -(model.contact_points - palm_position).norm(p=2, dim=-1).sum(dim=-1)

    def cal_table_penetration(self, table_pos: torch.Tensor, table_normal: torch.Tensor) -> torch.Tensor:
        """Calculate hand penetration through a plane."""
        model = self._model
        
        surface_points = model.get_surface_points()
        if surface_points.shape[1] == 0:
            return torch.zeros(model.batch_size, device=model.device)

        # Signed distance from points to plane
        signed_dist = torch.sum((surface_points - table_pos.unsqueeze(1)) * table_normal.unsqueeze(1), dim=-1)
        
        # Only penalize penetration (negative distance)
        penetration = -signed_dist.clamp(max=0)
        
        return penetration.sum(-1)

    def sample_contact_points(
        self, total_batch_size: int, n_contacts_per_finger: int
    ) -> torch.Tensor:
        """
        Sample contact point indices.
        Ensures each finger is sampled at least once.

        Args:
            total_batch_size: Batch size
            n_contacts_per_finger: Number of contact points per finger

        Returns:
            (B, n_fingers * n_contacts_per_finger) torch.LongTensor of sampled contact point indices
        """
        model = self._model
        fingertip_keywords = model.fingertip_keywords

        # Get link indices containing finger keywords
        finger_possible_link_idxs_list = [
            [
                link_idx
                for link_name, link_idx in model.link_name_to_link_index.items()
                if finger_keyword in link_name
            ]
            for finger_keyword in fingertip_keywords
        ]

        # Get possible contact point indices from these link indices
        finger_possible_contact_point_idxs_list = [
            sum(
                [model.link_index_to_global_indices[link_idx] for link_idx in link_idxs],
                [],
            )
            for link_idxs in finger_possible_link_idxs_list
        ]

        # Sample from these contact point indices
        sampled_contact_point_idxs_list = []
        for finger_possible_contact_point_idxs in finger_possible_contact_point_idxs_list:
            sampled_idxs = torch.randint(
                len(finger_possible_contact_point_idxs),
                size=[total_batch_size, n_contacts_per_finger],
                device=model.device,
            )
            sampled_contact_point_idxs = torch.tensor(
                finger_possible_contact_point_idxs, device=model.device, dtype=torch.long
            )[sampled_idxs]
            sampled_contact_point_idxs_list.append(sampled_contact_point_idxs)

        sampled_contact_point_idxs_list = torch.cat(
            sampled_contact_point_idxs_list, dim=1
        )

        assert sampled_contact_point_idxs_list.shape == (
            total_batch_size,
            len(fingertip_keywords) * n_contacts_per_finger,
        )

        return sampled_contact_point_idxs_list

    @staticmethod
    def decompose_hand_pose(
        hand_pose: torch.Tensor,
        n_dofs: int,
        rot_type: str = "quat"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose hand_pose tensor into global translation, global rotation matrix, and joint angles.

        Args:
            hand_pose: (B, 3 + n_dofs + rot_dim) torch.FloatTensor
                Concatenated tensor containing translation, joint angles, and rotation parameters
            n_dofs: Number of degrees of freedom
            rot_type: Rotation representation type ('r6d', 'quat', 'axis', 'euler')

        Returns:
            global_translation: (B, 3) torch.FloatTensor
            global_rotation: (B, 3, 3) torch.FloatTensor
            qpos: (B, n_dofs) torch.FloatTensor
        """
        if hand_pose.dim() == 1:
            hand_pose = hand_pose.unsqueeze(0)
        assert hand_pose.dim() == 2, "hand_pose must be a 1D or 2D tensor for decomposition."

        # Determine rotation parameter dimension
        if rot_type == 'r6d':
            rot_dim = 6
        elif rot_type == 'quat':
            rot_dim = 4
        elif rot_type == 'axis':  # Axis-angle
            rot_dim = 3
        elif rot_type == 'euler': # Euler angles
            rot_dim = 3
        else:
            raise ValueError(f"Unsupported rotation type for decomposition: {rot_type}")

        expected_dim = 3 + n_dofs + rot_dim
        assert hand_pose.shape[1] == expected_dim, \
            f"Hand pose shape error for decomposition (rot_type: '{rot_type}'). " \
            f"Expected last dimension to be {expected_dim} (3 trans + {n_dofs} DoFs + {rot_dim} rot), " \
            f"but got {hand_pose.shape[1]}."

        # Use slice constants
        from utils.hand_constants import (QPOS_SLICE, ROTATION_SLICE,
                                          TRANSLATION_SLICE)

        global_translation = hand_pose[:, TRANSLATION_SLICE]
        qpos = hand_pose[:, QPOS_SLICE]
        rotation_params = hand_pose[:, ROTATION_SLICE]

        # Convert rotation parameters to rotation matrix
        if rot_type == 'r6d':
            from pytorch3d.transforms import rotation_6d_to_matrix
            global_rotation = rotation_6d_to_matrix(rotation_params)
        elif rot_type == 'quat':
            from pytorch3d.transforms import quaternion_to_matrix
            global_rotation = quaternion_to_matrix(rotation_params)
        elif rot_type == 'axis':
            from pytorch3d.transforms import axis_angle_to_matrix
            global_rotation = axis_angle_to_matrix(rotation_params)
        elif rot_type == 'euler':
            from pytorch3d.transforms import euler_angles_to_matrix

            # Assume "XYZ" convention for Euler angles
            global_rotation = euler_angles_to_matrix(rotation_params, convention="XYZ")

        return global_translation, global_rotation, qpos