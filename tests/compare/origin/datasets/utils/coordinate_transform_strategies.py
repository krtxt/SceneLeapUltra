"""
Coordinate transformation strategies for SceneLeapPro datasets.

This module implements the Strategy pattern for coordinate transformations,
allowing different transformation modes to be handled in a clean, extensible way.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from .dataset_config import CONFIG


class TransformationData:
    """
    Data container for coordinate transformation inputs and outputs.
    
    This class encapsulates all the data needed for coordinate transformations
    and provides a clean interface for passing data between strategies.
    """
    
    def __init__(self, 
                 pc_cam_raw_xyz_rgb: np.ndarray,
                 grasps_omf: torch.Tensor,
                 obj_verts: torch.Tensor,
                 R_omf_to_cf_np: Optional[np.ndarray] = None,
                 t_omf_to_cf_np: Optional[np.ndarray] = None,
                 object_mask_np: Optional[np.ndarray] = None):
        """
        Initialize transformation data container.
        
        Args:
            pc_cam_raw_xyz_rgb: Point cloud in camera frame with RGB (N, 6)
            grasps_omf: Hand poses in object model frame
            obj_verts: Object vertices in object model frame
            R_omf_to_cf_np: Rotation matrix from object model frame to camera frame
            t_omf_to_cf_np: Translation vector from object model frame to camera frame
            object_mask_np: Boolean mask indicating object points in point cloud
        """
        self.pc_cam_raw_xyz_rgb = pc_cam_raw_xyz_rgb
        self.grasps_omf = grasps_omf
        self.obj_verts = obj_verts
        self.R_omf_to_cf_np = R_omf_to_cf_np
        self.t_omf_to_cf_np = t_omf_to_cf_np
        self.object_mask_np = object_mask_np if object_mask_np is not None else np.array([])
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data shapes and types."""
        if not isinstance(self.pc_cam_raw_xyz_rgb, np.ndarray):
            raise TypeError("pc_cam_raw_xyz_rgb must be numpy array")
        
        if self.pc_cam_raw_xyz_rgb.ndim != 2 or self.pc_cam_raw_xyz_rgb.shape[1] != CONFIG.PC_XYZRGB_DIM:
            raise ValueError(f"pc_cam_raw_xyz_rgb must have shape (N, {CONFIG.PC_XYZRGB_DIM})")
        
        if not isinstance(self.grasps_omf, torch.Tensor):
            raise TypeError("grasps_omf must be torch.Tensor")
        
        if not isinstance(self.obj_verts, torch.Tensor):
            raise TypeError("obj_verts must be torch.Tensor")
        
        # Validate transformation matrices if provided
        if self.R_omf_to_cf_np is not None:
            if not isinstance(self.R_omf_to_cf_np, np.ndarray) or self.R_omf_to_cf_np.shape != (3, 3):
                raise ValueError("R_omf_to_cf_np must be (3, 3) numpy array")
        
        if self.t_omf_to_cf_np is not None:
            if not isinstance(self.t_omf_to_cf_np, np.ndarray) or self.t_omf_to_cf_np.shape != (3,):
                raise ValueError("t_omf_to_cf_np must be (3,) numpy array")
    
    def has_valid_transform(self) -> bool:
        """Check if valid transformation matrices are available."""
        return (self.R_omf_to_cf_np is not None and 
                self.t_omf_to_cf_np is not None)


class CoordinateTransformStrategy(ABC):
    """
    Abstract base class for coordinate transformation strategies.
    
    This class defines the interface that all coordinate transformation
    strategies must implement, following the Strategy design pattern.
    """
    
    def __init__(self, mode_name: str):
        """
        Initialize the transformation strategy.
        
        Args:
            mode_name: Name of the transformation mode
        """
        self.mode_name = mode_name
        if mode_name not in CONFIG.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode_name}'. Valid modes: {CONFIG.VALID_MODES}")
    
    @abstractmethod
    def transform(self, data: TransformationData) -> Dict[str, Any]:
        """
        Apply coordinate transformation to the input data.
        
        Args:
            data: TransformationData container with input data
            
        Returns:
            Dict[str, Any]: Dictionary containing transformed data with keys:
                - 'pc': Transformed point cloud (np.ndarray)
                - 'grasps': Transformed grasps (torch.Tensor)
                - 'obj_verts': Transformed object vertices (torch.Tensor)
        """
        pass
    
    def _convert_numpy_to_tensor(self, array: np.ndarray, 
                                device: torch.device, 
                                dtype: torch.dtype = None) -> torch.Tensor:
        """
        Convert numpy array to torch tensor with specified device and dtype.
        
        Args:
            array: Numpy array to convert
            device: Target device
            dtype: Target dtype (defaults to float32)
            
        Returns:
            torch.Tensor: Converted tensor
        """
        if dtype is None:
            dtype = CONFIG.DEFAULT_DTYPE
        
        return torch.from_numpy(array.astype(np.float32)).to(device=device, dtype=dtype)
    
    def _ensure_tensor_device_dtype(self, tensor: torch.Tensor, 
                                   reference_tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor has same device and dtype as reference tensor.
        
        Args:
            tensor: Tensor to convert
            reference_tensor: Reference tensor for device/dtype
            
        Returns:
            torch.Tensor: Tensor with matching device/dtype
        """
        return tensor.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    
    def _handle_grasp_dimensions(self, grasps: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Handle different grasp tensor dimensions and return normalized format.
        
        Args:
            grasps: Input grasp tensor, can be (23,) or (N, 23)
            
        Returns:
            Tuple[torch.Tensor, bool]: (normalized_grasps, was_single_grasp)
                - normalized_grasps: Always (N, 23) format
                - was_single_grasp: True if input was (23,) format
        """
        if grasps.dim() == 1 and grasps.shape[0] == CONFIG.POSE_DIM:
            # Single grasp (23,) -> (1, 23)
            return grasps.unsqueeze(0), True
        elif grasps.dim() == 2 and grasps.shape[1] == CONFIG.POSE_DIM:
            # Multiple grasps (N, 23)
            return grasps, False
        else:
            raise ValueError(f"Invalid grasp tensor shape: {grasps.shape}. Expected (23,) or (N, 23)")
    
    def _restore_grasp_dimensions(self, grasps: torch.Tensor, was_single_grasp: bool) -> torch.Tensor:
        """
        Restore original grasp tensor dimensions.
        
        Args:
            grasps: Normalized grasp tensor (N, 23)
            was_single_grasp: Whether original input was single grasp
            
        Returns:
            torch.Tensor: Grasp tensor in original format
        """
        if was_single_grasp and grasps.shape[0] == 1:
            return grasps.squeeze(0)  # (1, 23) -> (23,)
        return grasps
    
    def _extract_pose_components(self, grasps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract position, quaternion, and joint components from grasp poses.
        
        Args:
            grasps: Grasp tensor with shape (N, 23)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (positions, quaternions, joints)
                - positions: (N, 3) position vectors
                - quaternions: (N, 4) quaternions in wxyz format
                - joints: (N, 16) joint angles
        """
        positions = grasps[:, :CONFIG.POSITION_DIM]
        quaternions = grasps[:, CONFIG.POSITION_DIM:CONFIG.POSITION_DIM + CONFIG.QUATERNION_DIM]
        joints = grasps[:, CONFIG.POSITION_DIM + CONFIG.QUATERNION_DIM:]
        
        return positions, quaternions, joints
    
    def _combine_pose_components(self, positions: torch.Tensor, 
                                quaternions: torch.Tensor, 
                                joints: torch.Tensor) -> torch.Tensor:
        """
        Combine position, quaternion, and joint components into grasp poses.
        
        Args:
            positions: (N, 3) position vectors
            quaternions: (N, 4) quaternions in wxyz format
            joints: (N, 16) joint angles
            
        Returns:
            torch.Tensor: Combined grasp tensor (N, 23)
        """
        return torch.cat([positions, quaternions, joints], dim=1)


# Import transformation functions
from .common_utils import transform_point_cloud
from .transform_utils import transform_hand_poses_omf_to_cf


class ObjectCentricStrategy(CoordinateTransformStrategy):
    """
    Object-centric coordinate transformation strategy.

    Transforms point cloud from camera frame to object model frame.
    Keeps grasps and object vertices in object model frame.
    """

    def __init__(self):
        super().__init__("object_centric")

    def transform(self, data: TransformationData) -> Dict[str, Any]:
        """
        Apply object-centric transformation.

        Args:
            data: TransformationData container with input data

        Returns:
            Dict containing transformed data
        """
        # Default outputs (no transformation)
        final_pc_xyz_rgb_np = data.pc_cam_raw_xyz_rgb  # Default to camera frame
        final_grasps = data.grasps_omf  # Default to object model frame
        final_obj_verts = data.obj_verts.clone()  # Default to object model frame

        # Transform point cloud to object model frame if transformation is available
        if data.has_valid_transform():
            pc_omf_xyz = transform_point_cloud(
                data.pc_cam_raw_xyz_rgb[:, :3],
                data.R_omf_to_cf_np,
                data.t_omf_to_cf_np
            )
            final_pc_xyz_rgb_np = np.hstack((pc_omf_xyz, data.pc_cam_raw_xyz_rgb[:, 3:6]))

        return {
            'pc': final_pc_xyz_rgb_np,
            'grasps': final_grasps,
            'obj_verts': final_obj_verts
        }


class CameraCentricStrategy(CoordinateTransformStrategy):
    """
    Camera-centric coordinate transformation strategy.

    Keeps point cloud in camera frame.
    Transforms grasps and object vertices from object model frame to camera frame.
    """

    def __init__(self):
        super().__init__("camera_centric")

    def transform(self, data: TransformationData) -> Dict[str, Any]:
        """
        Apply camera-centric transformation.

        Args:
            data: TransformationData container with input data

        Returns:
            Dict containing transformed data
        """
        # Default outputs
        final_pc_xyz_rgb_np = data.pc_cam_raw_xyz_rgb  # Keep in camera frame
        final_grasps = data.grasps_omf  # Default to object model frame
        final_obj_verts = data.obj_verts.clone()  # Default to object model frame

        # Transform grasps and object vertices to camera frame if transformation is available
        if data.has_valid_transform():
            # Transform grasps from OMF to CF
            final_grasps = transform_hand_poses_omf_to_cf(
                data.grasps_omf,
                data.R_omf_to_cf_np,
                data.t_omf_to_cf_np
            )

            # Transform object vertices from OMF to CF
            if data.obj_verts.numel() > 0:
                R_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                    data.R_omf_to_cf_np, data.obj_verts.device
                )
                t_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                    data.t_omf_to_cf_np, data.obj_verts.device
                )
                final_obj_verts = torch.matmul(data.obj_verts, R_omf_to_cf_tensor.T) + t_omf_to_cf_tensor.unsqueeze(0)

        return {
            'pc': final_pc_xyz_rgb_np,
            'grasps': final_grasps,
            'obj_verts': final_obj_verts
        }


class CameraCentricObjNormalizedStrategy(CoordinateTransformStrategy):
    """
    Camera-centric object-mean-normalized coordinate transformation strategy.

    Transforms data to camera frame, then normalizes by subtracting object center.
    """

    def __init__(self):
        super().__init__("camera_centric_obj_mean_normalized")

    def transform(self, data: TransformationData) -> Dict[str, Any]:
        """
        Apply camera-centric object-normalized transformation.

        Args:
            data: TransformationData container with input data

        Returns:
            Dict containing transformed data
        """
        pc_cf_xyz_rgb = data.pc_cam_raw_xyz_rgb  # Point cloud is in camera frame

        # Calculate object mean from point cloud in camera frame
        obj_mean_cf_for_norm = np.zeros(3, dtype=np.float32)
        if len(data.object_mask_np) == len(pc_cf_xyz_rgb):
            obj_pts_cf_xyz = pc_cf_xyz_rgb[data.object_mask_np, :3]
            if obj_pts_cf_xyz.shape[0] > 0:
                obj_mean_cf_for_norm = np.mean(obj_pts_cf_xyz, axis=0)

        # Transform PC by subtracting object mean
        final_pc_xyz = pc_cf_xyz_rgb[:, :3] - obj_mean_cf_for_norm.reshape(1, 3)
        final_pc_xyz_rgb_np = np.hstack((final_pc_xyz, pc_cf_xyz_rgb[:, 3:6]))

        # Transform grasps to camera frame, then normalize
        grasps_cf_raw = data.grasps_omf  # Default to OMF if transform fails
        if data.has_valid_transform():
            grasps_cf_raw = transform_hand_poses_omf_to_cf(
                data.grasps_omf, data.R_omf_to_cf_np, data.t_omf_to_cf_np
            )

        # Normalize grasps by object mean
        final_grasps = self._normalize_grasps_by_mean(
            grasps_cf_raw, obj_mean_cf_for_norm
        )

        # Transform obj_verts to camera frame, then normalize
        obj_verts_cf = data.obj_verts.clone()  # Start with OMF
        if data.has_valid_transform() and data.obj_verts.numel() > 0:
            R_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                data.R_omf_to_cf_np, data.obj_verts.device
            )
            t_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                data.t_omf_to_cf_np, data.obj_verts.device
            )
            obj_verts_cf = torch.matmul(data.obj_verts, R_omf_to_cf_tensor.T) + t_omf_to_cf_tensor.unsqueeze(0)

        obj_mean_cf_tensor_for_verts = self._convert_numpy_to_tensor(
            obj_mean_cf_for_norm, obj_verts_cf.device
        )
        final_obj_verts = obj_verts_cf - obj_mean_cf_tensor_for_verts.unsqueeze(0)

        return {
            'pc': final_pc_xyz_rgb_np,
            'grasps': final_grasps,
            'obj_verts': final_obj_verts
        }

    def _normalize_grasps_by_mean(self, grasps: torch.Tensor, mean_vector: np.ndarray) -> torch.Tensor:
        """
        Normalize grasp positions by subtracting mean vector.

        Args:
            grasps: Grasp tensor (N, 23) or (23,)
            mean_vector: Mean vector to subtract (3,)

        Returns:
            Normalized grasp tensor
        """
        # Handle both single grasp (23,) and multiple grasps (N, 23) cases
        grasps_normalized, was_single_grasp = self._handle_grasp_dimensions(grasps)

        if grasps_normalized.numel() > 0 and grasps_normalized.shape[0] > 0 and grasps_normalized.shape[1] == CONFIG.POSE_DIM:
            positions, quaternions, joints = self._extract_pose_components(grasps_normalized)

            mean_tensor = self._convert_numpy_to_tensor(mean_vector, grasps_normalized.device)
            positions_shifted = positions - mean_tensor.unsqueeze(0)

            grasps_transformed = self._combine_pose_components(positions_shifted, quaternions, joints)
            return self._restore_grasp_dimensions(grasps_transformed, was_single_grasp)
        else:
            return grasps  # Keep as is if invalid


class CameraCentricSceneNormalizedStrategy(CoordinateTransformStrategy):
    """
    Camera-centric scene-mean-normalized coordinate transformation strategy.

    Transforms data to camera frame, then normalizes by subtracting scene center.
    """

    def __init__(self):
        super().__init__("camera_centric_scene_mean_normalized")

    def transform(self, data: TransformationData) -> Dict[str, Any]:
        """
        Apply camera-centric scene-normalized transformation.

        Args:
            data: TransformationData container with input data

        Returns:
            Dict containing transformed data
        """
        pc_cf_xyz_rgb = data.pc_cam_raw_xyz_rgb  # Point cloud is in camera frame

        # Calculate scene mean from entire scene point cloud in camera frame
        scene_mean_cf_for_norm = np.zeros(3, dtype=np.float32)
        if pc_cf_xyz_rgb.shape[0] > 0:
            scene_mean_cf_for_norm = np.mean(pc_cf_xyz_rgb[:, :3], axis=0)

        # Transform PC by subtracting scene mean
        final_pc_xyz = pc_cf_xyz_rgb[:, :3] - scene_mean_cf_for_norm.reshape(1, 3)
        final_pc_xyz_rgb_np = np.hstack((final_pc_xyz, pc_cf_xyz_rgb[:, 3:6]))

        # Transform grasps to camera frame, then normalize
        grasps_cf_raw = data.grasps_omf  # Default to OMF if transform fails
        if data.has_valid_transform():
            grasps_cf_raw = transform_hand_poses_omf_to_cf(
                data.grasps_omf, data.R_omf_to_cf_np, data.t_omf_to_cf_np
            )

        # Normalize grasps by scene mean
        final_grasps = self._normalize_grasps_by_mean(
            grasps_cf_raw, scene_mean_cf_for_norm
        )

        # Transform obj_verts to camera frame, then normalize
        obj_verts_cf = data.obj_verts.clone()  # Start with OMF
        if data.has_valid_transform() and data.obj_verts.numel() > 0:
            R_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                data.R_omf_to_cf_np, data.obj_verts.device
            )
            t_omf_to_cf_tensor = self._convert_numpy_to_tensor(
                data.t_omf_to_cf_np, data.obj_verts.device
            )
            obj_verts_cf = torch.matmul(data.obj_verts, R_omf_to_cf_tensor.T) + t_omf_to_cf_tensor.unsqueeze(0)

        scene_mean_cf_tensor_for_verts = self._convert_numpy_to_tensor(
            scene_mean_cf_for_norm, obj_verts_cf.device
        )
        final_obj_verts = obj_verts_cf - scene_mean_cf_tensor_for_verts.unsqueeze(0)

        return {
            'pc': final_pc_xyz_rgb_np,
            'grasps': final_grasps,
            'obj_verts': final_obj_verts
        }

    def _normalize_grasps_by_mean(self, grasps: torch.Tensor, mean_vector: np.ndarray) -> torch.Tensor:
        """
        Normalize grasp positions by subtracting mean vector.

        Args:
            grasps: Grasp tensor (N, 23) or (23,)
            mean_vector: Mean vector to subtract (3,)

        Returns:
            Normalized grasp tensor
        """
        # Handle both single grasp (23,) and multiple grasps (N, 23) cases
        grasps_normalized, was_single_grasp = self._handle_grasp_dimensions(grasps)

        if grasps_normalized.numel() > 0 and grasps_normalized.shape[0] > 0 and grasps_normalized.shape[1] == CONFIG.POSE_DIM:
            positions, quaternions, joints = self._extract_pose_components(grasps_normalized)

            mean_tensor = self._convert_numpy_to_tensor(mean_vector, grasps_normalized.device)
            positions_shifted = positions - mean_tensor.unsqueeze(0)

            grasps_transformed = self._combine_pose_components(positions_shifted, quaternions, joints)
            return self._restore_grasp_dimensions(grasps_transformed, was_single_grasp)
        else:
            return grasps  # Keep as is if invalid


def create_transform_strategy(mode: str) -> CoordinateTransformStrategy:
    """
    Factory function to create appropriate transformation strategy.

    Args:
        mode: Transformation mode name

    Returns:
        CoordinateTransformStrategy: Strategy instance for the specified mode

    Raises:
        ValueError: If mode is not supported
    """
    if mode == "object_centric":
        return ObjectCentricStrategy()
    elif mode == "camera_centric":
        return CameraCentricStrategy()
    elif mode == "camera_centric_obj_mean_normalized":
        return CameraCentricObjNormalizedStrategy()
    elif mode == "camera_centric_scene_mean_normalized":
        return CameraCentricSceneNormalizedStrategy()
    else:
        raise ValueError(f"Unsupported transformation mode: {mode}. Valid modes: {CONFIG.VALID_MODES}")
