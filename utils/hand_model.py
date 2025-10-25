import logging
import pathlib
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import trimesh as tm

# Define slice constants for pose components
TRANSLATION_SLICE = slice(0, 3)
QPOS_SLICE = slice(3, 19)
ROTATION_SLICE = slice(19, None)

# Attempt to import PyTorch3D for specific functionalities
try:
    from pytorch3d.ops import knn_points

    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

from utils.hand_loader import HandLoader
from utils.hand_physics import HandPhysics
from utils.hand_types import HandModelType
from utils.hand_visualizer import HandVisualizer
from utils.leap_hand_info import (
    LEAP_HAND_CONTACT_POINTS_PATH,
    LEAP_HAND_DEFAULT_JOINT_ANGLES,
    LEAP_HAND_DEFAULT_ORIENTATION,
    LEAP_HAND_FINGERTIP_KEYWORDS,
    LEAP_HAND_FINGERTIP_NAMES,
    LEAP_HAND_JOINT_NAMES,
    LEAP_HAND_NUM_FINGERS,
    LEAP_HAND_NUM_JOINTS,
    LEAP_HAND_PENETRATION_POINTS_PATH,
    LEAP_HAND_URDF_PATH,
)

SELF_PENETRATION_POINT_RADIUS = 0.01


class HandStateManager:
    """Manages hand pose state and related transformations."""

    def __init__(self, device: Union[str, torch.device]):
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Reset all state variables."""
        self.hand_pose = None
        self.hand_pose_original = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        self.batch_size_original = None
        self.num_grasps = None

    def set_pose_state(
        self,
        hand_pose_original: torch.Tensor,
        hand_pose_flat: torch.Tensor,
        global_translation: torch.Tensor,
        global_rotation: torch.Tensor,
    ) -> None:
        """Set pose-related state variables."""
        self.hand_pose_original = hand_pose_original
        self.hand_pose = hand_pose_flat
        self.global_translation = global_translation
        self.global_rotation = global_rotation
        self.batch_size_original = hand_pose_original.shape[0]
        self.num_grasps = hand_pose_original.shape[1]

        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()

    def set_kinematics_state(self, current_status) -> None:
        """Set forward kinematics state."""
        self.current_status = current_status

    def set_contact_state(
        self,
        contact_point_indices: Optional[torch.Tensor],
        contact_points: Optional[torch.Tensor],
    ) -> None:
        """Set contact-related state."""
        self.contact_point_indices = contact_point_indices
        self.contact_points = contact_points

    @property
    def batch_size_flat(self) -> int:
        """Get flattened batch size (B * num_grasps)."""
        if self.hand_pose is None:
            raise ValueError("Hand pose not set.")
        return self.hand_pose.shape[0]

    def get_flattened_global_transforms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get flattened global transforms for multi-grasp processing."""
        if self.global_rotation is None or self.global_translation is None:
            raise ValueError("Global transforms not set. Call set_pose_state first.")

        # Flatten first two dimensions: [B, num_grasps, ...] -> [B*num_grasps, ...]
        global_rotation_flat = self.global_rotation.view(-1, 3, 3)
        global_translation_flat = self.global_translation.view(-1, 3)
        return global_rotation_flat, global_translation_flat

    @property
    def is_initialized(self) -> bool:
        """Check if state is properly initialized."""
        return (
            self.hand_pose is not None
            and self.global_translation is not None
            and self.global_rotation is not None
            and self.current_status is not None
        )


class HandModel:
    def __init__(
        self,
        hand_model_type: HandModelType = HandModelType.LEAP,
        n_surface_points: int = 0,
        rot_type: str = "quat",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize hand model."""
        self.hand_model_type = hand_model_type
        self.n_surface_points = n_surface_points
        self.rot_type = rot_type
        self.device = device

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)

        # Load static resources
        self.loader = HandLoader(
            urdf_path=self.urdf_path,
            contact_points_path=self.contact_points_path,
            penetration_points_path=self.penetration_points_path,
            n_surface_points=n_surface_points,
            device=device,
            joint_names=self.joint_names,
        )
        loaded_data = self.loader.load()
        for key, value in loaded_data.items():
            setattr(self, key, value)

        # Initialize submodules
        self.state = HandStateManager(device)
        self.physics = HandPhysics(self)
        self._visualizer = None  # Lazy initialization

        # Indexing setup
        self.link_name_to_link_index = {
            link_name: idx for idx, link_name in enumerate(self.mesh)
        }
        self.link_name_to_contact_candidates = {
            link_name: self.mesh[link_name]["contact_candidates"]
            for link_name in self.mesh
        }
        contact_candidates = [
            self.link_name_to_contact_candidates[link_name] for link_name in self.mesh
        ]
        self.global_index_to_link_index = sum(
            [
                [i] * len(contact_candidates)
                for i, contact_candidates in enumerate(contact_candidates)
            ],
            [],
        )
        self.link_index_to_global_indices = defaultdict(list)
        for global_idx, link_idx in enumerate(self.global_index_to_link_index):
            self.link_index_to_global_indices[link_idx].append(global_idx)

        self.contact_candidates = torch.cat(contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(
            self.global_index_to_link_index, dtype=torch.long, device=device
        )
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [
            self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh
        ]
        self.global_index_to_link_index_penetration = sum(
            [
                [i] * len(penetration_keypoints)
                for i, penetration_keypoints in enumerate(self.penetration_keypoints)
            ],
            [],
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)
        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.n_keypoints = self.penetration_keypoints.shape[0]

    @property
    def visualizer(self):
        """延迟初始化可视化器"""
        if self._visualizer is None:
            self._visualizer = HandVisualizer(self)
        return self._visualizer

    def _ensure_device_consistency(self, *tensors: torch.Tensor):
        """
        Ensures all input tensors are on the correct device (self.device).

        Args:
            *tensors: Variable number of tensors to check and convert

        Returns:
            Single tensor if one input, tuple of tensors if multiple inputs
        """
        converted_tensors = []
        for tensor in tensors:
            if tensor is not None and tensor.device != self.device:
                converted_tensors.append(tensor.to(self.device))
            else:
                converted_tensors.append(tensor)

        # Return single tensor if only one input, tuple otherwise
        if len(converted_tensors) == 1:
            return converted_tensors[0]
        else:
            return tuple(converted_tensors)

    def sample_contact_points(
        self, total_batch_size: int, n_contacts_per_finger: int
    ) -> torch.Tensor:
        """
        Sample contact point indices.

        ✅ MIGRATED: This method has been migrated to HandPhysics.
        This is now a proxy to the HandPhysics implementation.
        """
        return self.physics.sample_contact_points(
            total_batch_size, n_contacts_per_finger
        )

    def set_parameters(
        self,
        hand_pose: torch.Tensor,
        contact_point_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Set translation, joint angles, rotation, and contact points of the hand model.
        The representation of the rotation is determined by `self.rot_type`.
        The hand_pose tensor is structured as: translation (3D), joint angles (self.n_dofs D),
        and rotation parameters (rot_dim D), in that specific order.

        ⚠️ NOT MIGRATED: This method remains in HandModel as it is the core interface
        that coordinates multiple subsystems and manages the overall model state.

        Unified multi-grasp format (single grasp is treated as num_grasps=1):
        - Expected format: [B, num_grasps, pose_dim]
        - Single grasp compatibility: [B, pose_dim] -> [B, 1, pose_dim]

        Parameters
        ----------
        hand_pose: (B, num_grasps, 3 + `self.n_dofs` + rot_dim) torch.FloatTensor
            A concatenated tensor containing translation, joint angles, and rotation parameters.
            The order is: translation, then joint angles, then rotation parameters.
            For backward compatibility, (B, pose_dim) is also accepted and treated as (B, 1, pose_dim).
        contact_point_indices: (B_flat, `n_contact`) [Optional] torch.LongTensor
            Indices of contact candidates in flattened batch format (B_flat = B * num_grasps).
        """
        # Step 1: Normalize input format
        hand_pose = self._normalize_hand_pose(hand_pose)

        # Step 2: Validate and decompose pose
        global_translation_flat, global_rotation_flat, joint_angles = (
            self._validate_and_decompose_pose(hand_pose)
        )

        # Step 3: Compute forward kinematics
        self._compute_forward_kinematics(joint_angles)

        # Step 4: Process contact points
        self._process_contact_points(contact_point_indices)

    def _normalize_hand_pose(self, hand_pose: torch.Tensor) -> torch.Tensor:
        """
        Normalize input format: Convert to unified [B, num_grasps, pose_dim] format.

        Args:
            hand_pose: Input hand pose tensor

        Returns:
            Normalized hand pose tensor in [B, num_grasps, pose_dim] format
        """
        # Normalize input format: Convert to unified [B, num_grasps, pose_dim] format
        if hand_pose.dim() == 1:
            # [pose_dim] -> [1, 1, pose_dim]
            hand_pose = hand_pose.unsqueeze(0).unsqueeze(0)
        elif hand_pose.dim() == 2:
            # [B, pose_dim] -> [B, 1, pose_dim] (single grasp compatibility)
            hand_pose = hand_pose.unsqueeze(1)
        elif hand_pose.dim() == 3:
            # [B, num_grasps, pose_dim] - already in expected format
            pass
        else:
            raise ValueError(
                f"hand_pose must be 1D, 2D or 3D tensor, got {hand_pose.dim()}D tensor with shape {hand_pose.shape}"
            )

        # Unified multi-grasp format processing
        assert (
            hand_pose.dim() == 3
        ), f"Internal error: hand_pose should be 3D after normalization, got {hand_pose.dim()}D"

        return hand_pose

    def _validate_and_decompose_pose(
        self, hand_pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate pose dimensions and decompose into components.

        Args:
            hand_pose: Normalized hand pose tensor [B, num_grasps, pose_dim]

        Returns:
            Tuple of (global_translation_flat, global_rotation_flat, joint_angles)
        """
        # Flatten to 2D for forward kinematics and other calculations
        hand_pose_flat = hand_pose.view(
            -1, hand_pose.shape[-1]
        )  # [B*num_grasps, pose_dim]

        # Validate pose dimensions
        if self.rot_type == "r6d":
            rot_dim = 6
        elif self.rot_type == "quat":
            rot_dim = 4
        elif self.rot_type == "axis":  # Axis-angle
            rot_dim = 3
        elif self.rot_type == "euler":  # Euler angles
            rot_dim = 3
        else:
            raise ValueError(f"Unsupported rotation type: {self.rot_type}")

        expected_dim = 3 + self.n_dofs + rot_dim
        assert hand_pose_flat.shape[1] == expected_dim, (
            f"Hand pose shape error (rot_type: '{self.rot_type}'). "
            f"Expected last dimension to be {expected_dim} "
            f"(3 trans + {self.n_dofs} DoFs + {rot_dim} rot), "
            f"but got {hand_pose_flat.shape[1]}."
        )

        # Ensure input tensors are on the correct device
        hand_pose_flat = self._ensure_device_consistency(hand_pose_flat)

        # Decompose hand_pose: translation | joint_angles | rotation_parameters
        global_translation_flat = hand_pose_flat[
            :, TRANSLATION_SLICE
        ]  # [B*num_grasps, 3]
        joint_angles = hand_pose_flat[:, QPOS_SLICE]  # [B*num_grasps, n_dofs]
        rotation_params = hand_pose_flat[:, ROTATION_SLICE]  # [B*num_grasps, rot_dim]

        # Ensure joint_angles are on the correct device (for forward kinematics)
        joint_angles = self._ensure_device_consistency(joint_angles)

        # Convert rotation parameters to rotation matrices
        if self.rot_type == "r6d":
            from pytorch3d.transforms import rotation_6d_to_matrix

            global_rotation_flat = rotation_6d_to_matrix(rotation_params)
        elif self.rot_type == "quat":
            from pytorch3d.transforms import quaternion_to_matrix

            global_rotation_flat = quaternion_to_matrix(rotation_params)
        elif self.rot_type == "axis":
            from pytorch3d.transforms import axis_angle_to_matrix

            global_rotation_flat = axis_angle_to_matrix(rotation_params)
        elif self.rot_type == "euler":
            from pytorch3d.transforms import euler_angles_to_matrix

            # Assuming "XYZ" convention for Euler angles
            global_rotation_flat = euler_angles_to_matrix(
                rotation_params, convention="XYZ"
            )

        # Reshape to unified multi-grasp format [B, num_grasps, ...]
        batch_size_original = hand_pose.shape[0]
        num_grasps = hand_pose.shape[1]
        global_translation = global_translation_flat.view(
            batch_size_original, num_grasps, 3
        )
        global_rotation = global_rotation_flat.view(
            batch_size_original, num_grasps, 3, 3
        )

        # Save state using HandStateManager
        self.state.set_pose_state(
            hand_pose, hand_pose_flat, global_translation, global_rotation
        )

        return global_translation_flat, global_rotation_flat, joint_angles

    def _compute_forward_kinematics(self, joint_angles: torch.Tensor) -> None:
        """
        Compute forward kinematics for joint angles.

        Args:
            joint_angles: Joint angles tensor [B*num_grasps, n_dofs]
        """
        # Forward kinematics for joint angles
        current_status = self.chain.forward_kinematics(joint_angles)
        self.state.set_kinematics_state(current_status)

    def _process_contact_points(
        self,
        contact_point_indices: Optional[torch.Tensor],
    ) -> None:
        """
        Process contact points if provided.

        Args:
            contact_point_indices: Contact point indices
        """
        if contact_point_indices is not None:
            contact_point_indices = self._ensure_device_consistency(
                contact_point_indices
            )
            if contact_point_indices.dim() != 2:
                raise ValueError(
                    "contact_point_indices must have shape (B_flat, n_contact). "
                    f"Received tensor with shape {tuple(contact_point_indices.shape)}."
                )

            batch_size, n_contact = contact_point_indices.shape
            expected_batch = self.state.batch_size_flat
            if batch_size != expected_batch:
                raise ValueError(
                    f"contact_point_indices batch ({batch_size}) does not match flattened batch size ({expected_batch})."
                )

            if n_contact == 0:
                empty_contacts = torch.empty(
                    batch_size, 0, 3, dtype=torch.float, device=self.device
                )
                self.state.set_contact_state(contact_point_indices, empty_contacts)
                return

            # Local contact points for each index (B_flat, n_contact, 3)
            local_contact_points = self.contact_candidates[contact_point_indices]

            # Link index for each sampled contact (B_flat, n_contact)
            link_indices = self.global_index_to_link_index[contact_point_indices]

            # Stack all link transforms once and gather the ones we need
            link_matrices = []
            for link_name in self.mesh:
                matrix = self.state.current_status[link_name].get_matrix()
                if matrix.dim() == 2:
                    matrix = matrix.unsqueeze(0)
                if matrix.shape[0] == 1:
                    matrix = matrix.expand(batch_size, -1, -1)
                elif matrix.shape[0] != batch_size:
                    raise ValueError(
                        f"Unexpected FK matrix batch size for link '{link_name}': "
                        f"{matrix.shape[0]} (expected {batch_size})."
                    )
                link_matrices.append(matrix)

            link_transforms = torch.stack(link_matrices, dim=1)  # (B_flat, n_links, 4, 4)

            batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(-1)
            contact_link_transforms = link_transforms[
                batch_idx, link_indices
            ]  # (B_flat, n_contact, 4, 4)

            # Convert local points to homogeneous coordinates
            ones = torch.ones(
                batch_size, n_contact, 1, dtype=local_contact_points.dtype, device=self.device
            )
            contact_points_homog = torch.cat([local_contact_points, ones], dim=-1)

            # Transform into the hand base frame
            contact_points_in_hand_base_frame = (
                contact_link_transforms @ contact_points_homog.unsqueeze(-1)
            )[..., :3, 0]

            # Transform into world frame (flattened batch already matches)
            contact_points = self._transform_points_to_world(
                contact_points_in_hand_base_frame
            )
            self.state.set_contact_state(contact_point_indices, contact_points)
        else:
            self.state.set_contact_state(None, None)

    def decompose_hand_pose(
        self, hand_pose: torch.Tensor, rot_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decomposes the hand_pose tensor into global translation, global rotation matrix, and joint angles.

        ✅ MIGRATED: This method has been migrated to HandPhysics as a static method.
        This is now a proxy to the HandPhysics static method implementation.
        """
        current_rot_type = rot_type if rot_type is not None else self.rot_type
        return self.physics.decompose_hand_pose(
            hand_pose, self.n_dofs, current_rot_type
        )

    def cal_distance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the distance from each point in `x` to the hand model.
        This is now a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_distance(x)

    def _compute_object_penetration(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the signed distance from each point in object point cloud `x` to the hand model.
        This is now a proxy to the HandPhysics implementation.
        """
        return self.physics.compute_object_penetration(x)

    def cal_self_penetration_energy(self) -> torch.Tensor:
        """
        Calculate self penetration energy.
        This is a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_self_penetration_energy()

    def cal_joint_limit_energy(self) -> torch.Tensor:
        """
        Calculate joint limit energy.
        This is a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_joint_limit_energy()

    def cal_finger_finger_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-finger distance energy.
        This is a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_finger_finger_distance_energy()

    def cal_finger_palm_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-palm distance energy.
        This is a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_finger_palm_distance_energy()

    def cal_table_penetration(
        self, table_pos: torch.Tensor, table_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate table penetration energy.
        This is a proxy to the HandPhysics implementation.
        """
        return self.physics.cal_table_penetration(table_pos, table_normal)

    def _collect_and_transform_points(self, key: str) -> torch.Tensor:
        """
        Collect and transform points from all links for a given key.
        Optimized version using vectorized operations where possible.

        Args:
            key: The key to access points in mesh data (e.g., 'surface_points', 'contact_candidates')

        Returns:
            Transformed points in hand base frame: (B*num_grasps, N_total, 3)
        """
        batch_size = self.state.batch_size_flat  # B * num_grasps
        device = self.device

        # Pre-allocate list with known size for better performance
        link_names = list(self.mesh.keys())
        points_list = []

        # Collect all non-empty point sets first
        for link_name in link_names:
            link_local_points = self.mesh[link_name][key]  # Shape (N_k, 3)
            n_points_for_this_link = link_local_points.shape[0]

            if n_points_for_this_link == 0:
                # Store empty tensor info for later
                points_list.append(
                    torch.empty(
                        (batch_size, 0, 3), dtype=link_local_points.dtype, device=device
                    )
                )
                continue

            # Transform points for this link
            transformed_link_points = self.state.current_status[
                link_name
            ].transform_points(link_local_points)

            # Normalize transformed points to correct batch size
            normalized_points = self._normalize_transformed_points(
                transformed_link_points, batch_size, n_points_for_this_link, link_name
            )
            points_list.append(normalized_points)

        # Vectorized concatenation with error handling
        if not points_list:
            return torch.empty((batch_size, 0, 3), dtype=torch.float32, device=device)

        try:
            # Use torch.cat with pre-allocated list for better performance
            final_points = torch.cat(points_list, dim=1).to(
                device
            )  # Changed dim=-2 to dim=1 for clarity
        except RuntimeError as e:
            self._log_concatenation_error(e, points_list, key)
            return torch.empty((batch_size, 0, 3), dtype=torch.float32, device=device)

        return final_points

    def _normalize_transformed_points(
        self,
        transformed_points: torch.Tensor,
        expected_batch_size: int,
        n_points: int,
        link_name: str,
    ) -> torch.Tensor:
        """
        Normalize transformed points to have the correct batch size.

        Args:
            transformed_points: Points from transform_points
            expected_batch_size: Expected batch size
            n_points: Expected number of points per batch
            link_name: Link name for error reporting

        Returns:
            Normalized points: (expected_batch_size, n_points, 3)
        """
        original_shape = transformed_points.shape

        # Handle different output shapes from transform_points
        if transformed_points.ndim == 3:
            # Case 1: Output is (1, N_k, 3) but batch_size > 1. Needs expansion.
            if (
                transformed_points.shape[0] == 1
                and expected_batch_size > 1
                and transformed_points.shape[1] == n_points
                and transformed_points.shape[2] == 3
            ):
                transformed_points = transformed_points.expand(
                    expected_batch_size, n_points, 3
                )
        elif transformed_points.ndim == 2:
            # Case 2a: Output is (N_k, 3). This is the main problematic case.
            if (
                transformed_points.shape[0] == n_points
                and transformed_points.shape[1] == 3
            ):
                transformed_points = transformed_points.unsqueeze(0).expand(
                    expected_batch_size, n_points, 3
                )
            # Case 2b: Output is (B, 3). This implies N_k was 1 and that dimension was squeezed.
            elif (
                transformed_points.shape[0] == expected_batch_size
                and transformed_points.shape[1] == 3
                and n_points == 1
            ):
                transformed_points = transformed_points.unsqueeze(
                    1
                )  # Shape: (batch_size, 1, 3)

        # Validate final shape
        expected_shape = (expected_batch_size, n_points, 3)
        if transformed_points.shape != expected_shape:
            self._log_warning(
                f"Shape mismatch for link '{link_name}' AFTER adjustments. "
                f"Original output from transform_points was {original_shape}. "
                f"Got {transformed_points.shape}, expected {expected_shape}."
            )

        return transformed_points

    def _log_concatenation_error(
        self, error: RuntimeError, points: list, operation: str
    ) -> None:
        """Log detailed error information for concatenation failures."""
        error_msg = f"Error during torch.cat in {operation}: {error}"
        shape_info = "Tensor shapes collected:"
        for i, p in enumerate(points):
            link_name = (
                list(self.mesh.keys())[i] if i < len(self.mesh) else f"unknown_{i}"
            )
            shape_info += (
                f"\n  points[{i}] (link: {link_name}): shape {p.shape}, ndim {p.ndim}"
            )

        self._log_warning(
            f"{error_msg}\n{shape_info}\nReturning empty tensor as fallback."
        )

    def _log_warning(self, message: str) -> None:
        """Centralized warning logging."""
        self.logger.warning(message)

    def get_surface_points(self) -> torch.Tensor:
        """Get surface points transformed to world coordinates."""
        points_in_base_frame = self._collect_and_transform_points("surface_points")
        return self._transform_points_to_world(points_in_base_frame)

    def get_contact_candidates(self) -> torch.Tensor:
        """
        Get all contact candidates

        Returns
        -------
        points: (B*num_grasps, `n_contact_candidates`, 3) torch.Tensor
        """
        points_in_base_frame = self._collect_and_transform_points("contact_candidates")
        return self._transform_points_to_world(points_in_base_frame)

    def get_penetration_keypoints(self) -> torch.Tensor:
        """
        Get penetration keypoints

        Returns
        -------
        points: (B*num_grasps, `n_keypoints`, 3) torch.Tensor
        """
        points_in_base_frame = self._collect_and_transform_points(
            "penetration_keypoints"
        )
        return self._transform_points_to_world(points_in_base_frame)

    def get_plotly_data(
        self,
        i: int,
        opacity: float = 0.5,
        color: str = "lightblue",
        with_contact_points: bool = False,
        with_contact_candidates: bool = False,
        with_surface_points: bool = False,
        with_penetration_keypoints: bool = False,
        pose: Optional[np.ndarray] = None,
        visual: bool = False,
    ) -> list:
        """
        Get visualization data for plotly.graph_objects.
        This is a proxy to the HandVisualizer implementation.
        """
        return self.visualizer.get_plotly_data(
            i=i,
            opacity=opacity,
            color=color,
            with_contact_points=with_contact_points,
            with_contact_candidates=with_contact_candidates,
            with_surface_points=with_surface_points,
            with_penetration_keypoints=with_penetration_keypoints,
            pose=pose,
            visual=visual,
        )

    def get_trimesh_data(self, i: int) -> tm.Trimesh:
        """
        Get full mesh.
        This is a proxy to the HandVisualizer implementation.
        """
        return self.visualizer.get_trimesh_data(i)

    @property
    def n_fingers(self) -> int:
        return LEAP_HAND_NUM_FINGERS

    # Backward compatibility properties
    @property
    def hand_pose(self) -> Optional[torch.Tensor]:
        """Access to hand pose state."""
        return self.state.hand_pose

    @property
    def global_translation(self) -> Optional[torch.Tensor]:
        """Access to global translation state."""
        return self.state.global_translation

    @property
    def global_rotation(self) -> Optional[torch.Tensor]:
        """Access to global rotation state."""
        return self.state.global_rotation

    @property
    def current_status(self):
        """Access to current kinematics status."""
        return self.state.current_status

    @property
    def contact_point_indices(self) -> Optional[torch.Tensor]:
        """Access to contact point indices."""
        return self.state.contact_point_indices

    @property
    def contact_points(self) -> Optional[torch.Tensor]:
        """Access to contact points."""
        return self.state.contact_points

    @property
    def batch_size_original(self) -> Optional[int]:
        """Access to original batch size."""
        return self.state.batch_size_original

    @property
    def num_grasps(self) -> Optional[int]:
        """Access to number of grasps."""
        return self.state.num_grasps

    @property
    def batch_size(self) -> int:
        """返回展平后的批次大小 (B * num_grasps)，用于内部计算"""
        return self.state.batch_size_flat

    @property
    def num_fingers(self) -> int:
        return self.n_fingers

    @property
    def fingertip_keywords(self) -> list:
        return LEAP_HAND_FINGERTIP_KEYWORDS

    @property
    def fingertip_names(self) -> list:
        return LEAP_HAND_FINGERTIP_NAMES

    @property
    def joint_names(self) -> list:
        return LEAP_HAND_JOINT_NAMES

    @property
    def num_joints(self) -> int:
        num_joints = LEAP_HAND_NUM_JOINTS
        assert num_joints == self.n_dofs, f"{num_joints} != {self.n_dofs}"
        return num_joints

    @property
    def default_joint_angles(self) -> torch.Tensor:
        return LEAP_HAND_DEFAULT_JOINT_ANGLES

    @property
    def default_orientation(self) -> torch.Tensor:
        return LEAP_HAND_DEFAULT_ORIENTATION

    @property
    def urdf_path(self) -> pathlib.Path:
        return LEAP_HAND_URDF_PATH

    @property
    def contact_points_path(self) -> pathlib.Path:
        return LEAP_HAND_CONTACT_POINTS_PATH

    @property
    def penetration_points_path(self) -> pathlib.Path:
        return LEAP_HAND_PENETRATION_POINTS_PATH

    def _transform_points_to_world(
        self, points_in_base_frame: torch.Tensor
    ) -> torch.Tensor:
        """
        Transforms points from the hand's base frame to the world frame.
        Unified multi-grasp format processing.

        Args:
            points_in_base_frame: (B*num_grasps, N, 3) tensor of points.

        Returns:
            (B*num_grasps, N, 3) tensor of points in world frame.
        """
        # Use unified multi-grasp format processing
        global_rotation_flat, global_translation_flat = (
            self._get_flattened_global_transforms()
        )

        points_world = points_in_base_frame @ global_rotation_flat.transpose(
            1, 2
        ) + global_translation_flat.unsqueeze(1)
        return points_world

    def __call__(
        self,
        hand_pose: torch.Tensor,
        scene_pc: Optional[torch.Tensor] = None,
        # plane_parameters: Optional[torch.Tensor] = None,
        with_meshes: bool = False,
        with_surface_points: bool = False,
        with_contact_candidates: bool = False,
        with_penetration_keypoints: bool = False,
        with_penetration: bool = False,
        with_distance: bool = False,  # ADDED
        with_fingertip_keypoints: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Main interface to get various hand model information based on hand_pose.

        Unified multi-grasp format (single grasp is treated as num_grasps=1):
        - Expected format: [B, num_grasps, pose_dim]
        - Single grasp compatibility: [B, pose_dim] -> [B, 1, pose_dim]

        Args:
            hand_pose: (B, num_grasps, 3+rot_dim+n_dofs) torch.FloatTensor
                translation, rotation parameters, and joint angles.
                For backward compatibility, (B, pose_dim) is also accepted.
            scene_pc: (B, N_points, 4) or (N_points, 4) [Optional] torch.FloatTensor
                Scene point cloud, last dimension is (x, y, z, mask).
                Will be expanded to match the flattened batch dimension (B*num_grasps).
            with_meshes: If True, returns 'vertices' and 'faces'.
            with_surface_points: If True, returns 'surface_points'.
            with_contact_candidates: If True and scene_pc is provided, returns 'contact_candidates_dis'.
            with_penetration_keypoints: If True, returns 'penetration_keypoints'.
            with_penetration: If True and scene_pc is provided, returns 'penetration' (signed distances).
            with_distance: If True and scene_pc is provided, returns 'distance' (signed distances).
            with_fingertip_keypoints: If True, returns 'fingertip_keypoints'.

        Returns:
            A dictionary `hand_dict` containing the requested tensors.
            All tensors have batch dimension (B*num_grasps, ...).
        """
        # 设置参数（统一为多抓取格式）
        self.set_parameters(hand_pose)

        # 处理scene_pc的维度匹配
        scene_pc_processed = self._expand_input_for_multi_grasp(scene_pc)
        hand_dict: Dict[str, torch.Tensor] = {}

        # Process different output requests
        if with_surface_points:
            self._add_surface_points_to_dict(hand_dict)

        if with_penetration_keypoints:
            self._add_penetration_keypoints_to_dict(hand_dict)

        if with_contact_candidates:
            self._add_contact_candidates_to_dict(hand_dict, scene_pc_processed)

        if with_meshes:
            self._add_meshes_to_dict(hand_dict)

        if with_fingertip_keypoints:
            self._add_fingertip_keypoints_to_dict(hand_dict)

        return hand_dict

    def _add_surface_points_to_dict(self, hand_dict: Dict[str, torch.Tensor]) -> None:
        """Add surface points to the result dictionary."""
        hand_dict["surface_points"] = self.get_surface_points()

    def _add_penetration_keypoints_to_dict(
        self, hand_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Add penetration keypoints to the result dictionary."""
        hand_dict["penetration_keypoints"] = self.get_penetration_keypoints()

    def _add_contact_candidates_to_dict(
        self,
        hand_dict: Dict[str, torch.Tensor],
        scene_pc_processed: Optional[torch.Tensor],
    ) -> None:
        """Add contact candidates distance to the result dictionary."""
        if scene_pc_processed is None:
            self._log_warning(
                "with_contact_candidates is True but scene_pc is None. Skipping 'contact_candidates_dis'."
            )
            return

        if not _PYTORCH3D_AVAILABLE:
            raise ImportError(
                "PyTorch3D (specifically knn_points) is required for 'contact_candidates_dis'. Please install it."
            )

        # Validate and prepare scene_pc
        scene_pc_batched = self._prepare_scene_pc_for_contact_candidates(
            scene_pc_processed
        )
        scene_pc_xyz = scene_pc_batched[..., :3]

        # Compute distances
        contact_candidates_world = (
            self.get_contact_candidates()
        )  # (B_flat, N_candidates, 3)
        if contact_candidates_world.shape[1] == 0:
            hand_dict["contact_candidates_dis"] = torch.empty(
                self.hand_pose.shape[0], scene_pc_xyz.shape[1], device=self.device
            )
        else:
            scene_pc_xyz, contact_candidates_world = self._ensure_device_consistency(
                scene_pc_xyz, contact_candidates_world
            )
            dists_sq, _, _ = knn_points(scene_pc_xyz, contact_candidates_world, K=1)
            hand_dict["contact_candidates_dis"] = dists_sq.squeeze(-1)

    def _prepare_scene_pc_for_contact_candidates(
        self, scene_pc_processed: torch.Tensor
    ) -> torch.Tensor:
        """Prepare scene_pc for contact candidates distance calculation."""
        # Validate dimensions
        if scene_pc_processed.ndim == 2:
            if self.state.batch_size_flat == 1:  # 使用展平后的批次大小
                scene_pc_batched = scene_pc_processed.unsqueeze(0)
            else:
                raise ValueError(
                    f"scene_pc_processed (shape {scene_pc_processed.shape}) is 2D, "
                    f"but flattened batch_size is {self.state.batch_size_flat}. "
                    f"Ambiguous broadcast for contact_candidates_dis."
                )
        elif scene_pc_processed.ndim == 3:
            if (
                scene_pc_processed.shape[0] != self.state.batch_size_flat
            ):  # 使用展平后的批次大小
                raise ValueError(
                    f"scene_pc_processed batch_size {scene_pc_processed.shape[0]} "
                    f"does not match flattened batch_size {self.state.batch_size_flat} "
                    f"for contact_candidates_dis."
                )
            scene_pc_batched = scene_pc_processed
        else:
            raise ValueError(
                f"scene_pc_processed must have ndim 2 or 3 for contact_candidates_dis, "
                f"got {scene_pc_processed.ndim}"
            )

        if scene_pc_batched.shape[-1] < 3:
            raise ValueError(
                f"scene_pc must have at least 3 dimensions (xyz) for contact_candidates_dis, "
                f"got {scene_pc_batched.shape[-1]}"
            )

        return scene_pc_batched

    def _add_meshes_to_dict(self, hand_dict: Dict[str, torch.Tensor]) -> None:
        """Add mesh vertices and faces to the result dictionary. Optimized version."""
        link_names_ordered = list(self.mesh.keys())

        # Pre-allocate lists with known sizes
        vertices_list = []
        faces_list = []
        current_n_verts_offset = 0

        for link_name in link_names_ordered:
            link_mesh_data = self.mesh[link_name]

            # Process vertices
            link_verts_local = link_mesh_data["vertices"]
            if link_verts_local.numel() > 0:  # Only process non-empty vertices
                link_verts_transformed = self.state.current_status[
                    link_name
                ].transform_points(link_verts_local)

                if link_verts_transformed.ndim == 2:
                    link_verts_transformed = link_verts_transformed.unsqueeze(0).expand(
                        self.batch_size, -1, -1
                    )

                # Transform to world coordinates
                link_verts_world = self._transform_points_to_world(
                    link_verts_transformed
                )
                vertices_list.append(link_verts_world)

            # Process faces with offset
            if "faces" in link_mesh_data and link_mesh_data["faces"].numel() > 0:
                link_faces_local = link_mesh_data["faces"]
                # Apply vertex offset for proper indexing in concatenated mesh
                faces_with_offset = link_faces_local + current_n_verts_offset
                faces_list.append(faces_with_offset)

            # Update vertex offset for next link
            current_n_verts_offset += link_verts_local.shape[0]

        # Vectorized concatenation
        if vertices_list:
            hand_dict["vertices"] = torch.cat(vertices_list, dim=1)
        else:
            hand_dict["vertices"] = torch.empty(
                self.batch_size, 0, 3, device=self.device
            )

        if faces_list:
            hand_dict["faces"] = torch.cat(faces_list, dim=0)
        else:
            hand_dict["faces"] = torch.empty(0, 3, dtype=torch.long, device=self.device)

    def _add_fingertip_keypoints_to_dict(
        self, hand_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Add fingertip keypoints to the result dictionary. Optimized version."""
        if not self.fingertip_names:
            self._log_warning(
                "with_fingertip_keypoints is True, but self.fingertip_names is empty."
            )
            hand_dict["fingertip_keypoints"] = torch.empty(
                self.batch_size, 0, 3, device=self.device
            )
            return

        # Pre-filter valid fingertip links
        valid_fingertip_names = [
            name for name in self.fingertip_names if name in self.state.current_status
        ]

        if not valid_fingertip_names:
            self._log_warning("No valid fingertip links found in current_status.")
            hand_dict["fingertip_keypoints"] = torch.empty(
                self.batch_size, 0, 3, device=self.device
            )
            return

        # Vectorized processing of fingertip origins
        fingertip_origins_list = []

        for link_name in valid_fingertip_names:
            matrix = self.state.current_status[link_name].get_matrix()
            origin_in_hand_base = matrix[..., :3, 3]

            if origin_in_hand_base.ndim == 1:
                origin_in_hand_base = origin_in_hand_base.unsqueeze(0).expand(
                    self.batch_size, -1
                )

            # Transform to world coordinates and add fingertip dimension
            origin_world = self._transform_points_to_world(origin_in_hand_base)
            fingertip_origins_list.append(origin_world.unsqueeze(1))

        # Vectorized concatenation
        hand_dict["fingertip_keypoints"] = torch.cat(fingertip_origins_list, dim=1)

    def _flatten_to_multi_grasp(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Flatten a tensor from [B, num_grasps, ...] to [B*num_grasps, ...] format.

        Args:
            tensor: Input tensor with shape [B, num_grasps, ...]

        Returns:
            Flattened tensor with shape [B*num_grasps, ...]
        """
        if tensor.dim() < 2:
            raise ValueError(
                f"Tensor must have at least 2 dimensions, got {tensor.dim()}"
            )

        # Flatten first two dimensions
        new_shape = [-1] + list(tensor.shape[2:])
        return tensor.view(*new_shape)

    def _unflatten_from_multi_grasp(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Unflatten a tensor from [B*num_grasps, ...] to [B, num_grasps, ...] format.

        Args:
            tensor: Input tensor with shape [B*num_grasps, ...]

        Returns:
            Unflattened tensor with shape [B, num_grasps, ...]
        """
        if tensor.dim() < 1:
            raise ValueError(
                f"Tensor must have at least 1 dimension, got {tensor.dim()}"
            )

        # Unflatten first dimension
        new_shape = [self.batch_size_original, self.num_grasps] + list(tensor.shape[1:])
        return tensor.view(*new_shape)

    def _get_flattened_global_transforms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get flattened global rotation and translation for multi-grasp processing.

        Returns:
            Tuple of (global_rotation_flat, global_translation_flat)
            - global_rotation_flat: (B*num_grasps, 3, 3)
            - global_translation_flat: (B*num_grasps, 3)
        """
        return self.state.get_flattened_global_transforms()

    def _expand_input_for_multi_grasp(
        self, input_tensor: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Expands a tensor to match the flattened batch size for unified multi-grasp processing.
        E.g., from (B, N, D) to (B*num_grasps, N, D).

        Args:
            input_tensor: A tensor to potentially expand, e.g., scene_pc.

        Returns:
            The expanded tensor with shape (B*num_grasps, ...), or None if input was None.
        """
        if input_tensor is None:
            return input_tensor

        if input_tensor.ndim == 2:
            if self.state.batch_size_original == 1:
                input_batched = input_tensor.unsqueeze(0)  # (N, D) -> (1, N, D)
            else:
                raise ValueError(
                    f"Input tensor (shape {input_tensor.shape}) is 2D, but original batch_size is {self.state.batch_size_original}. Ambiguous broadcast."
                )
        elif input_tensor.ndim >= 3:
            if input_tensor.shape[0] != self.state.batch_size_original:
                raise ValueError(
                    f"Input tensor batch_size {input_tensor.shape[0]} does not match original batch_size {self.state.batch_size_original}."
                )
            input_batched = input_tensor
        else:
            raise ValueError(
                f"Input tensor must have ndim 2 or 3, got {input_tensor.ndim}"
            )

        # 扩展到 (B, num_grasps, N, D) 然后展平为 (B*num_grasps, N, D)
        reshape_shape = [-1] + list(input_batched.shape[1:])
        expanded = (
            input_batched.unsqueeze(1)
            .expand(
                self.state.batch_size_original,
                self.state.num_grasps,
                *input_batched.shape[1:],
            )
            .reshape(*reshape_shape)
        )

        return expanded


if __name__ == "__main__":
    # Initialize hand model
    hand_model = HandModel(
        hand_model_type=HandModelType.LEAP, device="cpu", n_surface_points=1000
    )
    print(hand_model.urdf_path)
