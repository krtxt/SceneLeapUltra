import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from torch.utils.data import Dataset

from .utils.common_utils import CameraInfo, create_point_cloud_from_depth_image
from .utils.coordinate_transform_strategies import (TransformationData,
                                                    create_transform_strategy)
from .utils.data_processing_utils import (build_data_index, collate_batch_data,
                                          collate_variable_grasps_batch,
                                          get_depth_view_indices_from_scene,
                                          load_and_process_hand_pose_data,
                                          load_scene_metadata,
                                          validate_dataset_configuration)
from .utils.error_utils import handle_loading_exception, log_dataset_warning
from .utils.io_utils import load_object_mesh, load_scene_images
from .utils.mask_utils import extract_object_mask
from .utils.pointcloud_utils import (add_rgb_to_pointcloud,
                                     crop_point_cloud_to_objects_with_mask,
                                     downsample_point_cloud_with_mask,
                                     map_2d_mask_to_3d_pointcloud)
from .utils.transform_utils import (create_se3_matrix_from_pose,
                                    extract_object_name_from_code,
                                    generate_negative_prompts,
                                    get_camera_transform,
                                    get_specific_hand_pose)


class _BaseLeapProDataset(Dataset):
    """
    Base class for SceneLeapPro datasets containing shared functionality.
    """

    def __init__(self, root_dir: str, succ_grasp_dir: str, obj_root_dir: str, mode: str = "camera_centric",
                 max_grasps_per_object: Optional[int] = 200, mesh_scale: float = 0.1,
                 num_neg_prompts: int = 4, enable_cropping: bool = True, max_points: int = 10000):
        # Validate configuration
        if not validate_dataset_configuration(root_dir, succ_grasp_dir, obj_root_dir, mode):
            raise ValueError("Invalid dataset configuration")

        # Store configuration
        self.root_dir = root_dir
        self.succ_grasp_dir = succ_grasp_dir
        self.obj_root_dir = obj_root_dir
        self.mesh_scale = mesh_scale
        self.num_neg_prompts = num_neg_prompts
        self.enable_cropping = enable_cropping
        self.max_points = max_points
        self.mode = mode
        self.max_grasps_per_object = max_grasps_per_object

        # Get scene directories (only folders starting with "scene")
        self.scene_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                           if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('scene')]

        # Load scene metadata
        self.instance_maps, self.scene_gt, self.collision_free_grasp_info = load_scene_metadata(self.scene_dirs)

        # Load and process hand pose data
        self.hand_pose_data = load_and_process_hand_pose_data(
            self.scene_dirs, self.collision_free_grasp_info, succ_grasp_dir
        )

        # Load camera configuration
        self.camera = self._load_camera_config(root_dir)

    def _load_camera_config(self, root_dir: str) -> CameraInfo:
        """Load camera configuration from JSON file."""
        camera_json_path = os.path.join(root_dir, 'camera.json')
        if os.path.exists(camera_json_path):
            return CameraInfo(**json.load(open(camera_json_path, 'r')))

        if self.scene_dirs:
            camera_json_path = os.path.join(self.scene_dirs[0], 'camera.json')
            if os.path.exists(camera_json_path):
                return CameraInfo(**json.load(open(camera_json_path, 'r')))
            else:
                raise FileNotFoundError(f"camera.json not found at {root_dir} or in the first scene {self.scene_dirs[0]}")
        else:
            raise FileNotFoundError(f"camera.json not found at {root_dir} and no scenes available to check.")

    def __len__(self):
        # This should be implemented in subclasses
        raise NotImplementedError("__len__ should be implemented in subclasses")

    def _load_scene_data(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load scene images and basic data."""
        scene_id = item_data['scene_id']
        scene_dir = os.path.join(self.root_dir, scene_id)
        depth_view_index = item_data['depth_view_index']

        # Load images
        depth_image, rgb_image, instance_mask_image = load_scene_images(scene_dir, depth_view_index)

        if depth_image is None or rgb_image is None or instance_mask_image is None:
            return None

        return {
            'scene_id': scene_id,
            'scene_dir': scene_dir,
            'depth_view_index': depth_view_index,
            'depth_image': depth_image,
            'rgb_image': rgb_image,
            'instance_mask_image': instance_mask_image
        }

    def _process_point_cloud_and_masks(self, scene_data: Dict[str, Any], item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process point cloud and extract object masks."""
        depth_image = scene_data['depth_image']
        rgb_image = scene_data['rgb_image']
        instance_mask_image = scene_data['instance_mask_image']
        scene_id = scene_data['scene_id']
        depth_view_index = scene_data['depth_view_index']
        category_id_for_masking = item_data['category_id_for_masking']

        # Create point cloud from depth
        scene_pc_raw_camera_frame = create_point_cloud_from_depth_image(depth_image, self.camera, organized=False)

        # Add RGB colors to point cloud
        scene_pc_with_rgb = self._add_rgb_to_pointcloud(scene_pc_raw_camera_frame, rgb_image, self.camera)

        # Extract object mask from instance segmentation and map to 3D point cloud
        view_specific_instance_attributes = self.instance_maps.get(scene_id, {}).get(str(depth_view_index), [])
        object_mask_2d = self._extract_object_mask(instance_mask_image, category_id_for_masking, view_specific_instance_attributes)

        # Reshape 2D mask back to image dimensions for proper mapping
        mask_2d_reshaped = object_mask_2d.reshape(instance_mask_image.shape)

        # Map 2D mask to 3D point cloud using projection
        object_mask_np = self._map_2d_mask_to_3d_pointcloud(
            scene_pc_with_rgb[:, :3], mask_2d_reshaped, self.camera
        )

        # Crop point cloud to focus on object regions (optional)
        if self.enable_cropping:
            scene_pc_with_rgb, object_mask_np = crop_point_cloud_to_objects_with_mask(
                scene_pc_with_rgb, object_mask_np, instance_mask_image, self.camera
            )

        return {
            'scene_pc_with_rgb': scene_pc_with_rgb,
            'object_mask_np': object_mask_np
        }

    def _get_camera_transform(self, scene_gt_for_view: List[Dict[str, Any]],
                             target_obj_id_from_index: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera transformation matrices for object."""
        return get_camera_transform(scene_gt_for_view, target_obj_id_from_index)

    def _extract_object_mask(self, instance_mask_image: np.ndarray, target_category_id: int,
                            view_specific_instance_attributes: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract object mask from instance segmentation image.

        Args:
            instance_mask_image: Instance segmentation mask
            target_category_id: Category ID of the target object
            view_specific_instance_attributes: View-specific instance attributes

        Returns:
            object_mask: Boolean mask indicating target object pixels
        """
        return extract_object_mask(instance_mask_image, target_category_id, view_specific_instance_attributes)

    def _map_2d_mask_to_3d_pointcloud(self, point_cloud_xyz: np.ndarray, mask_2d: np.ndarray,
                                    camera: CameraInfo) -> np.ndarray:
        """
        Map 2D instance mask to 3D point cloud mask by projecting points to image plane.

        Args:
            point_cloud_xyz: Point cloud coordinates (N, 3)
            mask_2d: 2D boolean mask (H, W)
            camera: Camera object with intrinsic parameters

        Returns:
            object_mask_3d: Boolean mask for point cloud (N,)
        """
        return map_2d_mask_to_3d_pointcloud(point_cloud_xyz, mask_2d, camera)

    def _add_rgb_to_pointcloud(self, pointcloud_xyz: np.ndarray, rgb_image: np.ndarray,
                              camera: CameraInfo) -> np.ndarray:
        """Add RGB colors to point cloud."""
        return add_rgb_to_pointcloud(pointcloud_xyz, rgb_image, camera)

    def _extract_object_name_from_code(self, object_code: str) -> str:
        """
        Extract object name from object code.

        Args:
            object_code: Object code in format "name_uid" where uid is a long hex string

        Returns:
            object_name: Extracted object name
        """
        return extract_object_name_from_code(object_code)

    def _generate_negative_prompts(self, scene_id: str, current_obj_name: str) -> List[str]:
        """
        Generate negative prompts from other objects in the scene.

        Args:
            scene_id: Scene identifier
            current_obj_name: Name of the current target object

        Returns:
            neg_prompts: List of negative prompt strings
        """
        return generate_negative_prompts(
            self.collision_free_grasp_info.get(scene_id, []),
            current_obj_name,
            self.num_neg_prompts
        )

    def _transform_data_by_mode(self, pc_cam_raw_xyz_rgb: np.ndarray, grasps_omf: torch.Tensor,
                               obj_verts: torch.Tensor, R_omf_to_cf_np: Optional[np.ndarray],
                               t_omf_to_cf_np: Optional[np.ndarray], object_mask_np: np.ndarray) -> Dict[str, Any]:
        """
        Transform point cloud, grasps, and object vertices based on coordinate frame mode.

        This method now uses the strategy pattern for coordinate transformations,
        providing better maintainability and extensibility.

        Args:
            pc_cam_raw_xyz_rgb: Point cloud with RGB in camera frame (P, 6)
            grasps_omf: Grasp poses in object model frame (N, 23) or (23,)
            obj_verts: Object vertices in object model frame (V, 3)
            R_omf_to_cf_np: Rotation matrix from object model frame to camera frame (3, 3)
            t_omf_to_cf_np: Translation vector from object model frame to camera frame (3,)
            object_mask_np: Object mask for point cloud (P,)

        Returns:
            Dict containing transformed 'pc', 'grasps', and 'obj_verts'
        """
        # Create transformation data container
        transform_data = TransformationData(
            pc_cam_raw_xyz_rgb=pc_cam_raw_xyz_rgb,
            grasps_omf=grasps_omf,
            obj_verts=obj_verts,
            R_omf_to_cf_np=R_omf_to_cf_np,
            t_omf_to_cf_np=t_omf_to_cf_np,
            object_mask_np=object_mask_np
        )

        # Get appropriate transformation strategy
        strategy = create_transform_strategy(self.mode)

        # Apply transformation
        return strategy.transform(transform_data)

    def _package_data_base(self, item_data: Dict[str, Any], transformed_data: Dict[str, Any],
                          object_mask_np: np.ndarray, obj_faces: torch.Tensor,
                          is_batch: bool = False) -> Dict[str, Any]:
        """
        Base method for packaging transformed data into final return dictionary.

        Args:
            item_data: Item metadata
            transformed_data: Transformed point cloud, grasps, and object vertices
            object_mask_np: Object mask for point cloud
            obj_faces: Object faces tensor
            is_batch: Whether to handle batch of grasps (True) or single grasp (False)

        Returns:
            Packaged data dictionary
        """
        final_pc_xyz_rgb_np = transformed_data['pc']
        final_grasps = transformed_data['grasps']
        final_obj_verts = transformed_data['obj_verts']

        # Downsample point cloud and mask together
        downsampled_pc_with_rgb_np, downsampled_object_mask_np = downsample_point_cloud_with_mask(
            final_pc_xyz_rgb_np, object_mask_np, max_points=self.max_points
        )
        downsampled_pc_with_rgb_tensor = torch.from_numpy(downsampled_pc_with_rgb_np).float()

        # Generate prompts
        object_name = self._extract_object_name_from_code(item_data['object_code'])
        positive_prompt = object_name
        negative_prompts = self._generate_negative_prompts(item_data['scene_id'], object_name)

        # Process grasps and create SE3 matrices
        if is_batch:
            hand_model_pose_reordered, se3_matrices = self._process_batch_grasps(
                final_grasps, downsampled_pc_with_rgb_tensor.device
            )
        else:
            hand_model_pose_reordered, se3_matrices = self._process_single_grasp(final_grasps)

        # Create base return dictionary
        result = {
            'obj_code': item_data['object_code'],
            'scene_pc': downsampled_pc_with_rgb_tensor,
            'object_mask': torch.from_numpy(downsampled_object_mask_np).bool(),
            'hand_model_pose': hand_model_pose_reordered,
            'se3': se3_matrices,
            'scene_id': item_data['scene_id'],
            'category_id_from_object_index': item_data['category_id_for_masking'],
            'depth_view_index': item_data['depth_view_index'],
            'obj_verts': final_obj_verts,
            'obj_faces': obj_faces,
            'positive_prompt': positive_prompt.replace('_', ' '),
            'negative_prompts': [prompt.replace('_', ' ') for prompt in negative_prompts]
        }

        # Add grasp_npy_idx for single grasp case
        if not is_batch:
            result['grasp_npy_idx'] = item_data['grasp_npy_idx']

        return result

    def _process_single_grasp(self, final_grasps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single grasp pose and create SE3 matrix.

        Args:
            final_grasps: Single grasp tensor (23,)

        Returns:
            Tuple of (reordered_pose, se3_matrix)
        """
        # Ensure final_grasps is (23,) tensor
        if not isinstance(final_grasps, torch.Tensor) or final_grasps.shape[0] != 23:
            final_grasps = torch.zeros(23, dtype=torch.float32)

        # Extract components
        P = final_grasps[:3]
        Q_wxyz = final_grasps[3:7]
        Joints = final_grasps[7:]

        # Reorder pose components: [P, Joints, Q_wxyz]
        hand_model_pose_reordered = torch.cat([P, Joints, Q_wxyz], dim=-1)

        # Create SE(3) matrix
        se3_matrix = create_se3_matrix_from_pose(P, Q_wxyz)

        return hand_model_pose_reordered, se3_matrix

    def _process_batch_grasps(self, final_grasps: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of grasp poses and create SE3 matrices.

        Args:
            final_grasps: Batch of grasp tensors (N, 23)
            device: Target device for tensors

        Returns:
            Tuple of (reordered_poses, se3_matrices)
        """
        # Ensure final_grasps is (N, 23) tensor
        if not (isinstance(final_grasps, torch.Tensor) and final_grasps.dim() == 2 and final_grasps.shape[1] == 23):
            final_grasps = torch.zeros((0, 23), dtype=torch.float32, device=device)

        N = final_grasps.shape[0]
        dtype = final_grasps.dtype

        # Initialize outputs
        hand_model_pose_reordered = torch.zeros((0, 23), device=device, dtype=dtype)
        se3_matrices = torch.zeros((0, 4, 4), device=device, dtype=dtype)

        if N > 0:
            # Extract components
            P_all = final_grasps[:, :3]
            Q_all_wxyz = final_grasps[:, 3:7]
            Joints_all = final_grasps[:, 7:]

            # Reorder pose components: [P, Joints, Q_wxyz]
            hand_model_pose_reordered = torch.cat([P_all, Joints_all, Q_all_wxyz], dim=1)

            # Correct zero quaternions
            Q_corrected_wxyz = Q_all_wxyz.clone()
            norms = torch.norm(Q_corrected_wxyz, dim=1, keepdim=True)
            zero_q_mask = norms.squeeze(1) < 1e-6

            identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
            Q_corrected_wxyz[zero_q_mask] = identity_q

            # Create SE(3) matrices
            R_all = quaternion_to_matrix(Q_corrected_wxyz)
            se3_matrices = torch.zeros((N, 4, 4), device=device, dtype=dtype)
            se3_matrices[:, :3, :3] = R_all
            se3_matrices[:, :3, 3] = P_all
            se3_matrices[:, 3, 3] = 1.0

        return hand_model_pose_reordered, se3_matrices

    def _create_error_return_dict_base(self, item_data: Dict[str, Any], error_msg: str,
                                      is_batch: bool = False) -> Dict[str, Any]:
        """
        Base method for creating standardized error return dictionaries.

        Args:
            item_data: Item data dictionary
            error_msg: Error message describing the failure
            is_batch: Whether to create batch-compatible format

        Returns:
            Standardized error dictionary
        """
        object_code = item_data.get('object_code', 'unknown')
        object_name = self._extract_object_name_from_code(object_code)
        scene_id = item_data.get('scene_id', 'unknown')
        negative_prompts = self._generate_negative_prompts(scene_id, object_name)

        # Base error dictionary structure
        error_dict = {
            'obj_code': object_code,
            'scene_pc': torch.zeros((0, 6)),  # 6D for xyz+rgb
            'object_mask': torch.zeros((0), dtype=torch.bool),
            'scene_id': scene_id,
            'category_id_from_object_index': item_data.get('category_id_for_masking', -1),
            'depth_view_index': item_data.get('depth_view_index', -1),
            'obj_verts': torch.zeros((0, 3), dtype=torch.float32),
            'obj_faces': torch.zeros((0, 3), dtype=torch.long),
            'positive_prompt': object_name.replace('_', ' '),
            'negative_prompts': [prompt.replace('_', ' ') for prompt in negative_prompts],
            'error': error_msg
        }

        # Format-specific fields
        if is_batch:
            # Batch format (ForMatchSceneLeapProDataset)
            error_dict.update({
                'hand_model_pose': torch.zeros((0, 23), dtype=torch.float32),
                'se3': torch.zeros((0, 4, 4), dtype=torch.float32)
            })
        else:
            # Single grasp format (SceneLeapProDataset)
            error_dict.update({
                'hand_model_pose': torch.zeros(23, dtype=torch.float32),
                'se3': torch.eye(4, dtype=torch.float32),
                'grasp_npy_idx': item_data.get('grasp_npy_idx', -1)
            })

        return error_dict


class SceneLeapProDataset(_BaseLeapProDataset):
    """
    SceneLeapPro dataset that returns single grasp per item.
    """

    def __init__(self, root_dir: str, succ_grasp_dir: str, obj_root_dir: str, mode: str = "camera_centric",
                 max_grasps_per_object: Optional[int] = 200, mesh_scale: float = 0.1,
                 num_neg_prompts: int = 4, enable_cropping: bool = True, max_points: int = 10000):
        super().__init__(root_dir, succ_grasp_dir, obj_root_dir, mode, max_grasps_per_object,
                         mesh_scale, num_neg_prompts, enable_cropping, max_points)

        # Build data index
        self.data = build_data_index(
            self.scene_dirs, self.collision_free_grasp_info, self.hand_pose_data, max_grasps_per_object
        )

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.data)}")

        item_data = self.data[idx]

        try:
            # Load scene data
            scene_data = self._load_scene_data(item_data)
            if scene_data is None:
                return self._create_error_return_dict(item_data, "Failed to load scene data")

            # Process point cloud and masks
            processed_data = self._process_point_cloud_and_masks(scene_data, item_data)
            if processed_data is None:
                return self._create_error_return_dict(item_data, "Failed to process point cloud and masks")

            # Load single grasp pose and object mesh
            grasp_object_data = self._load_grasp_and_object_data(item_data)
            if grasp_object_data is None:
                return self._create_error_return_dict(item_data, "Failed to load grasp and object data")

            # Transform data by coordinate frame mode
            transformed_data = self._transform_data_by_mode(
                pc_cam_raw_xyz_rgb=processed_data['scene_pc_with_rgb'],
                grasps_omf=grasp_object_data['hand_pose_tensor'],
                obj_verts=grasp_object_data['obj_verts'],
                R_omf_to_cf_np=grasp_object_data['cam_R_model_to_camera'],
                t_omf_to_cf_np=grasp_object_data['cam_t_model_to_camera'],
                object_mask_np=processed_data['object_mask_np']
            )

            # Package final data
            return self._package_data(item_data, transformed_data, processed_data['object_mask_np'], grasp_object_data['obj_faces'])

        except Exception as e:
            return self._create_error_return_dict(item_data, f"Unexpected error: {str(e)}")

    def _get_specific_hand_pose(self, scene_id: str, object_code: str,
                               grasp_npy_idx: int) -> Optional[torch.Tensor]:
        """Get specific hand pose from dataset."""
        return get_specific_hand_pose(self.hand_pose_data, scene_id, object_code, grasp_npy_idx)

    def _load_grasp_and_object_data(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load grasp poses and object mesh data."""
        scene_id = item_data['scene_id']
        object_code = item_data['object_code']
        category_id_for_masking = item_data['category_id_for_masking']
        depth_view_index = item_data['depth_view_index']
        grasp_npy_idx = item_data['grasp_npy_idx']

        # Get hand pose
        hand_pose_tensor = self._get_specific_hand_pose(scene_id, object_code, grasp_npy_idx)
        if hand_pose_tensor is None:
            return None

        # Get camera transformation
        scene_gt_for_view = self.scene_gt.get(scene_id, {}).get(str(depth_view_index), [])
        cam_R_model_to_camera, cam_t_model_to_camera = self._get_camera_transform(scene_gt_for_view, category_id_for_masking)

        # Load object mesh
        obj_verts, obj_faces = load_object_mesh(self.obj_root_dir, object_code, self.mesh_scale)
        if obj_verts is None or obj_faces is None:
            return None

        return {
            'hand_pose_tensor': hand_pose_tensor,
            'scene_gt_for_view': scene_gt_for_view,
            'cam_R_model_to_camera': cam_R_model_to_camera,
            'cam_t_model_to_camera': cam_t_model_to_camera,
            'obj_verts': obj_verts,
            'obj_faces': obj_faces
        }

    def _create_error_return_dict_old(self, item_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        pass





    def _create_error_return_dict(self, item_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """
        Create standardized error return dictionary for single grasp format.

        Args:
            item_data: Item data dictionary
            error_msg: Error message describing the failure

        Returns:
            Standardized error dictionary
        """
        return self._create_error_return_dict_base(item_data, error_msg, is_batch=False)

    def _package_data(self, item_data: Dict[str, Any], transformed_data: Dict[str, Any],
                     object_mask_np: np.ndarray, obj_faces: torch.Tensor) -> Dict[str, Any]:
        """Package transformed data into final return dictionary for single grasp."""
        return self._package_data_base(item_data, transformed_data, object_mask_np, obj_faces, is_batch=False)

    @staticmethod
    def collate_fn(batch):
        return collate_batch_data(batch)

class ForMatchSceneLeapProDataset(_BaseLeapProDataset):
    """
    A dataset that returns all collision-free grasps for a given object in a specific scene view.
    Inherits from _BaseLeapProDataset and modifies the data organization and return format.
    """

    def __init__(self, root_dir: str, succ_grasp_dir: str, obj_root_dir: str, mode: str = "camera_centric",
                 max_grasps_per_object: Optional[int] = 200, mesh_scale: float = 0.1,
                 num_neg_prompts: int = 4, enable_cropping: bool = True, max_points: int = 10000):
        """
        Initialize ForMatchSceneLeapProDataset.

        Args:
            root_dir: Root directory containing scene data
            succ_grasp_dir: Directory containing successful grasp data
            obj_root_dir: Directory containing object mesh data
            mode: Coordinate frame mode ("camera_centric", "object_centric", "camera_centric_obj_mean_normalized")
            max_grasps_per_object: Maximum number of grasps per object (None for unlimited)
            mesh_scale: Scale factor for object meshes
            num_neg_prompts: Number of negative prompts to generate
            enable_cropping: Whether to enable point cloud cropping
            max_points: Maximum number of points in downsampled point cloud
        """
        super().__init__(root_dir, succ_grasp_dir, obj_root_dir, mode, max_grasps_per_object,
                         mesh_scale, num_neg_prompts, enable_cropping, max_points)

        # Filter hand_pose_data to keep only collision-free poses
        self._filter_collision_free_poses()

        # Rebuild data index using ForMatch-specific logic (one item per object per view)
        self.data = self._build_data_index()

    def _filter_collision_free_poses(self):
        """
        Filter hand_pose_data to keep only collision-free poses based on collision_free_grasp_info.
        This modifies self.hand_pose_data in place.
        """
        # Iterate over a copy of scene_ids in self.hand_pose_data in case scenes are removed
        for scene_id_hp, scene_poses_hp in list(self.hand_pose_data.items()):
            if scene_id_hp not in self.collision_free_grasp_info:
                # This scene has no collision-free info, so all its objects have 0 collision-free grasps
                for obj_code_hp in list(scene_poses_hp.keys()):
                    all_poses_for_obj = scene_poses_hp[obj_code_hp]
                    pose_dim = 23  # Default pose dimension
                    dtype = torch.float32
                    device = 'cpu'  # Default device
                    if isinstance(all_poses_for_obj, torch.Tensor) and all_poses_for_obj.numel() > 0:
                        if all_poses_for_obj.ndim == 2:
                            pose_dim = all_poses_for_obj.shape[1]
                        dtype = all_poses_for_obj.dtype
                        device = all_poses_for_obj.device
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                continue

            grasp_entries_for_scene = self.collision_free_grasp_info[scene_id_hp]
            obj_code_to_cf_indices = {}
            for obj_grasp_entry in grasp_entries_for_scene:
                obj_name = obj_grasp_entry.get('object_name')
                obj_uid = obj_grasp_entry.get('uid')
                if not obj_name or not obj_uid:
                    continue
                current_obj_code = f"{obj_name}_{obj_uid}"
                obj_code_to_cf_indices[current_obj_code] = obj_grasp_entry.get('collision_free_indices', [])

            for obj_code_hp, all_poses_for_obj in list(scene_poses_hp.items()):
                pose_dim = 23  # Default
                dtype = torch.float32
                device = 'cpu'  # Default
                if isinstance(all_poses_for_obj, torch.Tensor) and all_poses_for_obj.numel() > 0:
                    if all_poses_for_obj.ndim == 2:
                        pose_dim = all_poses_for_obj.shape[1]
                    dtype = all_poses_for_obj.dtype
                    device = all_poses_for_obj.device
                else:  # all_poses_for_obj is None or not a Tensor or an empty Tensor
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                    continue

                collision_free_indices = obj_code_to_cf_indices.get(obj_code_hp)

                if collision_free_indices is not None and len(collision_free_indices) > 0:
                    try:
                        cf_indices_tensor = torch.tensor(collision_free_indices, dtype=torch.long)
                        valid_mask = (cf_indices_tensor >= 0) & (cf_indices_tensor < all_poses_for_obj.shape[0])
                        valid_cf_indices = cf_indices_tensor[valid_mask]

                        if self.max_grasps_per_object is not None and len(valid_cf_indices) > self.max_grasps_per_object:
                            valid_cf_indices = valid_cf_indices[:self.max_grasps_per_object]

                        if len(valid_cf_indices) > 0:
                            self.hand_pose_data[scene_id_hp][obj_code_hp] = all_poses_for_obj.index_select(0, valid_cf_indices.to(all_poses_for_obj.device))
                        else:
                            self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                    except Exception:  # Broad catch for safety during indexing
                        self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                else:  # No collision_free_indices for this obj_code or an empty list
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)

    def _build_data_index(self):
        """
        Builds a data index where each item corresponds to an object in a scene view,
        referencing all its available grasps.
        """
        data_index = []
        for scene_dir_path_local in self.scene_dirs:
            scene_id = os.path.basename(scene_dir_path_local)
            current_scene_collision_info = self.collision_free_grasp_info.get(scene_id, [])

            # Get depth view indices
            depth_view_indices = get_depth_view_indices_from_scene(scene_dir_path_local)
            if not depth_view_indices:
                continue

            for obj_grasp_entry in current_scene_collision_info:
                obj_name = obj_grasp_entry.get('object_name')
                obj_uid = obj_grasp_entry.get('uid')
                category_id_for_masking = obj_grasp_entry.get('object_index')

                if not obj_name or not obj_uid or category_id_for_masking is None:
                    continue

                object_code = f"{obj_name}_{obj_uid}"

                # Check if we have hand poses for this object
                obj_poses_tensor = self.hand_pose_data.get(scene_id, {}).get(object_code)
                if obj_poses_tensor is None or obj_poses_tensor.shape[0] == 0:
                    continue  # No hand poses available for this object

                for depth_view_idx in depth_view_indices:
                    data_index.append({
                        'scene_id': scene_id,
                        'object_code': object_code,
                        'category_id_for_masking': category_id_for_masking,
                        'depth_view_index': depth_view_idx
                        # Note: 'grasp_npy_idx' is not included in item_data
                    })

        if not data_index:
            print("Warning: ForMatchSceneLeapProDataset data_index is empty after build.")
        return data_index

    def _get_all_hand_poses_for_object(self, scene_id: str, object_code: str) -> torch.Tensor:
        """
        Retrieves all reverted hand poses for a given scene and object code.
        Returns a tensor of shape (N, 23) or (0, 23) if no poses are found or data is invalid.

        Args:
            scene_id: Scene identifier
            object_code: Object code

        Returns:
            torch.Tensor: All hand poses for the object, shape (N, 23)
        """
        all_reverted_poses_tensor = self.hand_pose_data.get(scene_id, {}).get(object_code)
        default_pose_dim = 23  # Must match the pose dimension used in SceneLeapProDataset

        if all_reverted_poses_tensor is None:
            return torch.zeros((0, default_pose_dim), dtype=torch.float32)

        if not isinstance(all_reverted_poses_tensor, torch.Tensor):
            return torch.zeros((0, default_pose_dim), dtype=torch.float32)

        if all_reverted_poses_tensor.ndim != 2 or all_reverted_poses_tensor.shape[1] != default_pose_dim:
            # This indicates malformed data for this specific object's poses
            return torch.zeros((0, default_pose_dim), dtype=torch.float32)

        return all_reverted_poses_tensor

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get item for ForMatchSceneLeapProDataset.
        Returns all grasps for a given object in a scene view.
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for ForMatchSceneLeapProDataset with length {len(self.data)}")

        item_data = self.data[idx]

        try:
            # Load scene data
            scene_data = self._load_scene_data(item_data)
            if scene_data is None:
                return self._create_error_return_dict(item_data, "Failed to load scene data")

            # Process point cloud and masks
            processed_data = self._process_point_cloud_and_masks(scene_data, item_data)
            if processed_data is None:
                return self._create_error_return_dict(item_data, "Failed to process point cloud and masks")

            # Load all grasp poses and object mesh
            grasp_object_data = self._load_grasp_and_object_data_for_all_grasps(item_data)
            if grasp_object_data is None:
                return self._create_error_return_dict(item_data, "Failed to load grasp and object data")

            # Transform data by coordinate frame mode
            transformed_data = self._transform_data_by_mode(
                pc_cam_raw_xyz_rgb=processed_data['scene_pc_with_rgb'],
                grasps_omf=grasp_object_data['hand_pose_tensor'],
                obj_verts=grasp_object_data['obj_verts'],
                R_omf_to_cf_np=grasp_object_data['cam_R_model_to_camera'],
                t_omf_to_cf_np=grasp_object_data['cam_t_model_to_camera'],
                object_mask_np=processed_data['object_mask_np']
            )

            # Package final data for all grasps
            return self._package_data_for_all_grasps(item_data, transformed_data, processed_data['object_mask_np'], grasp_object_data['obj_faces'])

        except Exception as e:
            return self._create_error_return_dict(item_data, f"Unexpected error: {str(e)}")

    def _load_grasp_and_object_data_for_all_grasps(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load grasp poses and object mesh data for all grasps.
        Modified version that gets all hand poses instead of a specific one.
        """
        scene_id = item_data['scene_id']
        object_code = item_data['object_code']
        category_id_for_masking = item_data['category_id_for_masking']
        depth_view_index = item_data['depth_view_index']

        # Get all hand poses for this object (N, 23)
        hand_pose_tensor = self._get_all_hand_poses_for_object(scene_id, object_code)

        # Get camera transformation
        scene_gt_for_view = self.scene_gt.get(scene_id, {}).get(str(depth_view_index), [])
        cam_R_model_to_camera, cam_t_model_to_camera = self._get_camera_transform(scene_gt_for_view, category_id_for_masking)

        # Load object mesh
        obj_verts, obj_faces = load_object_mesh(self.obj_root_dir, object_code, self.mesh_scale)
        if obj_verts is None or obj_faces is None:
            return None

        return {
            'hand_pose_tensor': hand_pose_tensor,
            'scene_gt_for_view': scene_gt_for_view,
            'cam_R_model_to_camera': cam_R_model_to_camera,
            'cam_t_model_to_camera': cam_t_model_to_camera,
            'obj_verts': obj_verts,
            'obj_faces': obj_faces
        }

    def _package_data_for_all_grasps(self, item_data: Dict[str, Any], transformed_data: Dict[str, Any],
                                   object_mask_np: np.ndarray, obj_faces: torch.Tensor) -> Dict[str, Any]:
        """Package transformed data into final return dictionary for all grasps."""
        return self._package_data_base(item_data, transformed_data, object_mask_np, obj_faces, is_batch=True)

    def _create_error_return_dict(self, item_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """
        Create standardized error return dictionary for batch format.

        Args:
            item_data: Item data dictionary
            error_msg: Error message describing the failure

        Returns:
            Standardized error dictionary compatible with batch format
        """
        return self._create_error_return_dict_base(item_data, error_msg, is_batch=True)

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of items from ForMatchSceneLeapProDataset.
        'hand_model_pose' and 'se3' are padded to become dense tensors.
        Other tensor fields are handled with standard collation or kept as lists.
        """
        return collate_variable_grasps_batch(batch)

