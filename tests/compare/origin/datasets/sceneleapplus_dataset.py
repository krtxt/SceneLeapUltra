import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from torch.utils.data import Dataset

from .sceneleappro_dataset import _BaseLeapProDataset
from .utils.common_utils import CameraInfo, create_point_cloud_from_depth_image
from .utils.coordinate_transform_strategies import (TransformationData,
                                                    create_transform_strategy)
from .utils.data_processing_utils import (collate_batch_data,
                                          collate_variable_grasps_batch,
                                          get_depth_view_indices_from_scene,
                                          load_and_process_hand_pose_data,
                                          load_scene_metadata,
                                          validate_dataset_configuration)
from .utils.error_utils import handle_loading_exception, log_dataset_warning
from .utils.grasp_sampling_utils import (generate_exhaustive_chunks,
                                         sample_grasps_from_available,
                                         sample_indices_from_available)
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


class SceneLeapPlusDataset(_BaseLeapProDataset):
    """
    SceneLeapPlus dataset returning a fixed number of grasps per item for parallel multi-grasp learning.

    This dataset is designed for parallel multi-grasp distribution learning. It differs from other
    datasets by returning a fixed number of grasps (`num_grasps`) per sample, with hand poses
    formatted as [num_grasps, 23] and SE3 matrices as [num_grasps, 4, 4]. Each item corresponds
    to one object in a scene view and handles grasp sampling or padding as needed.
    """

    def __init__(self, root_dir: str, succ_grasp_dir: str, obj_root_dir: str,
                 num_grasps: int = 8, mode: str = "camera_centric",
                 max_grasps_per_object: Optional[int] = 200, mesh_scale: float = 0.1,
                 num_neg_prompts: int = 4, enable_cropping: bool = True, max_points: int = 10000,
                 grasp_sampling_strategy: str = "random", use_exhaustive_sampling: bool = False,
                 exhaustive_sampling_strategy: str = "sequential"):
        """
        Initializes the SceneLeapPlusDataset.

        Args:
            root_dir: Root directory of the scene data.
            succ_grasp_dir: Directory with successful grasp data.
            obj_root_dir: Directory with object mesh data.
            num_grasps: Fixed number of grasps to return per sample.
            mode: Coordinate frame mode.
            max_grasps_per_object: Maximum number of grasps per object.
            mesh_scale: Scale factor for object meshes.
            num_neg_prompts: Number of negative prompts to generate.
            enable_cropping: Whether to enable point cloud cropping.
            max_points: Maximum points in the downsampled point cloud.
            grasp_sampling_strategy: Strategy for sampling grasps.
            use_exhaustive_sampling: Whether to use exhaustive sampling for 100% data utilization.
            exhaustive_sampling_strategy: Strategy for exhaustive sampling.
        """
        super().__init__(root_dir, succ_grasp_dir, obj_root_dir, mode, max_grasps_per_object,
                         mesh_scale, num_neg_prompts, enable_cropping, max_points)

        self.num_grasps = num_grasps
        self.grasp_sampling_strategy = grasp_sampling_strategy
        self.use_exhaustive_sampling = use_exhaustive_sampling
        self.exhaustive_sampling_strategy = exhaustive_sampling_strategy

        if num_grasps <= 0:
            raise ValueError(f"num_grasps must be positive, got {num_grasps}")

        valid_strategies = ["random", "first_n", "repeat", "farthest_point", "nearest_point",
                           "chunk_farthest_point", "chunk_nearest_point"]
        if grasp_sampling_strategy not in valid_strategies:
            raise ValueError(f"Invalid grasp_sampling_strategy: {grasp_sampling_strategy}")

        valid_exhaustive_strategies = ["sequential", "random", "interleaved",
                                     "chunk_farthest_point", "chunk_nearest_point"]
        if exhaustive_sampling_strategy not in valid_exhaustive_strategies:
            raise ValueError(f"Invalid exhaustive_sampling_strategy: {exhaustive_sampling_strategy}")
        
        self._filter_collision_free_poses()

        if self.use_exhaustive_sampling:
            self.data = self._build_exhaustive_data_index()
            logging.info(f"SceneLeapPlusDataset: Using exhaustive sampling with {len(self.data)} items.")
        else:
            self.data = self._build_data_index()
            logging.info(f"SceneLeapPlusDataset: Using traditional sampling with {len(self.data)} items.")

    def _filter_collision_free_poses(self):
        """Filters hand_pose_data in-place to keep only collision-free poses."""
        for scene_id_hp, scene_poses_hp in list(self.hand_pose_data.items()):
            if scene_id_hp not in self.collision_free_grasp_info:
                for obj_code_hp in list(scene_poses_hp.keys()):
                    all_poses_for_obj = scene_poses_hp[obj_code_hp]
                    pose_dim = 23
                    dtype = torch.float32
                    device = 'cpu'
                    if isinstance(all_poses_for_obj, torch.Tensor) and all_poses_for_obj.numel() > 0:
                        if all_poses_for_obj.ndim == 2: pose_dim = all_poses_for_obj.shape[1]
                        dtype, device = all_poses_for_obj.dtype, all_poses_for_obj.device
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                continue

            grasp_entries_for_scene = self.collision_free_grasp_info[scene_id_hp]
            obj_code_to_cf_indices = {f"{e.get('object_name')}_{e.get('uid')}": e.get('collision_free_indices', [])
                                      for e in grasp_entries_for_scene if e.get('object_name') and e.get('uid')}

            for obj_code_hp, all_poses_for_obj in list(scene_poses_hp.items()):
                pose_dim, dtype, device = 23, torch.float32, 'cpu'
                if isinstance(all_poses_for_obj, torch.Tensor) and all_poses_for_obj.numel() > 0:
                    if all_poses_for_obj.ndim == 2: pose_dim = all_poses_for_obj.shape[1]
                    dtype, device = all_poses_for_obj.dtype, all_poses_for_obj.device
                else:
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                    continue

                collision_free_indices = obj_code_to_cf_indices.get(obj_code_hp)
                if collision_free_indices:
                    try:
                        cf_indices_tensor = torch.tensor(collision_free_indices, dtype=torch.long)
                        valid_mask = (cf_indices_tensor >= 0) & (cf_indices_tensor < all_poses_for_obj.shape[0])
                        valid_cf_indices = cf_indices_tensor[valid_mask]

                        if self.max_grasps_per_object is not None and len(valid_cf_indices) > self.max_grasps_per_object:
                            sampled_indices = sample_indices_from_available(
                                valid_cf_indices.tolist(),
                                all_poses_for_obj,
                                self.max_grasps_per_object,
                                self.grasp_sampling_strategy
                            )
                            valid_cf_indices = torch.tensor(sampled_indices, dtype=torch.long)

                        if len(valid_cf_indices) > 0:
                            self.hand_pose_data[scene_id_hp][obj_code_hp] = all_poses_for_obj.index_select(0, valid_cf_indices.to(all_poses_for_obj.device))
                        else:
                            self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                    except Exception:
                        self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)
                else:
                    self.hand_pose_data[scene_id_hp][obj_code_hp] = torch.zeros((0, pose_dim), dtype=dtype, device=device)

    def __len__(self):
        return len(self.data)

    def _build_data_index(self):
        """Builds a data index where each item corresponds to an object in a scene view."""
        data_index = []
        for scene_dir_path_local in self.scene_dirs:
            scene_id = os.path.basename(scene_dir_path_local)
            current_scene_collision_info = self.collision_free_grasp_info.get(scene_id, [])
            depth_view_indices = get_depth_view_indices_from_scene(scene_dir_path_local)
            if not depth_view_indices: continue

            for obj_grasp_entry in current_scene_collision_info:
                obj_name, obj_uid = obj_grasp_entry.get('object_name'), obj_grasp_entry.get('uid')
                category_id_for_masking = obj_grasp_entry.get('object_index')
                if not obj_name or not obj_uid or category_id_for_masking is None: continue

                object_code = f"{obj_name}_{obj_uid}"
                obj_poses_tensor = self.hand_pose_data.get(scene_id, {}).get(object_code)
                if obj_poses_tensor is None or obj_poses_tensor.shape[0] == 0: continue

                for depth_view_idx in depth_view_indices:
                    data_index.append({
                        'scene_id': scene_id, 'object_code': object_code,
                        'category_id_for_masking': category_id_for_masking,
                        'depth_view_index': depth_view_idx
                    })
        if not data_index:
            logging.warning("SceneLeapPlusDataset data_index is empty after build.")
        return data_index

    def _build_exhaustive_data_index(self):
        """Builds an exhaustive data index by creating chunks of grasps for 100% data utilization."""
        exhaustive_data, total_available_grasps, total_used_grasps = [], 0, 0
        base_data_index = self._build_data_index()

        scene_object_groups = {}
        for item_data in base_data_index:
            key = f"{item_data['scene_id']}_{item_data['object_code']}"
            if key not in scene_object_groups:
                scene_object_groups[key] = {
                    'scene_id': item_data['scene_id'], 'object_code': item_data['object_code'],
                    'category_id_for_masking': item_data['category_id_for_masking'], 'views': []
                }
            scene_object_groups[key]['views'].append(item_data['depth_view_index'])

        for key, group_data in scene_object_groups.items():
            scene_id, object_code = group_data['scene_id'], group_data['object_code']
            obj_poses_tensor = self.hand_pose_data.get(scene_id, {}).get(object_code)
            if obj_poses_tensor is None or obj_poses_tensor.shape[0] == 0: continue

            total_grasps = obj_poses_tensor.shape[0]
            total_available_grasps += total_grasps

            if total_grasps < self.num_grasps:
                for view_idx in group_data['views']:
                    exhaustive_data.append({
                        'scene_id': scene_id, 'object_code': object_code,
                        'category_id_for_masking': group_data['category_id_for_masking'],
                        'depth_view_index': view_idx, 'is_exhaustive': False,
                        'chunk_idx': 0, 'total_chunks': 1, 'grasp_indices': None
                    })
                continue

            chunk_indices = generate_exhaustive_chunks(
                obj_poses_tensor,
                self.num_grasps,
                self.exhaustive_sampling_strategy
            )
            for view_idx in group_data['views']:
                for chunk_idx, indices in enumerate(chunk_indices):
                    exhaustive_data.append({
                        'scene_id': scene_id, 'object_code': object_code,
                        'category_id_for_masking': group_data['category_id_for_masking'],
                        'depth_view_index': view_idx, 'is_exhaustive': True,
                        'chunk_idx': chunk_idx, 'total_chunks': len(chunk_indices),
                        'grasp_indices': indices
                    })
                    total_used_grasps += len(indices)

        if total_available_grasps > 0:
            utilization_rate = total_used_grasps / total_available_grasps
            logging.info(f"Exhaustive sampling stats: Available={total_available_grasps}, "
                         f"Used={total_used_grasps}, Utilization={utilization_rate:.1%}, "
                         f"Expansion={len(exhaustive_data) / len(base_data_index):.1f}x")
        return exhaustive_data



    def _sample_indices_from_available(self, indices, all_poses_for_obj, num_samples, strategy):
        """Wrapper to reuse shared sampling utilities."""
        return sample_indices_from_available(indices, all_poses_for_obj, num_samples, strategy)

    def _sample_grasps_from_available(self, available_grasps: torch.Tensor) -> torch.Tensor:
        """Wrapper around shared grasp sampling helper."""
        return sample_grasps_from_available(
            available_grasps,
            self.num_grasps,
            self.grasp_sampling_strategy
        )

    def _get_fixed_number_hand_poses(self, scene_id: str, object_code: str,
                                   grasp_indices: Optional[List[int]] = None) -> torch.Tensor:
        """Gets a fixed number of hand poses for a given scene and object."""
        all_poses_tensor = self.hand_pose_data.get(scene_id, {}).get(object_code)
        default_pose_dim = 23

        if not isinstance(all_poses_tensor, torch.Tensor) or all_poses_tensor.ndim != 2 or all_poses_tensor.shape[1] != default_pose_dim:
            return torch.zeros((self.num_grasps, default_pose_dim), dtype=torch.float32)

        if grasp_indices:
            valid_indices = [i for i in grasp_indices if 0 <= i < all_poses_tensor.shape[0]]
            if len(valid_indices) >= self.num_grasps:
                return all_poses_tensor[torch.tensor(valid_indices[:self.num_grasps], dtype=torch.long)]

        return sample_grasps_from_available(
            all_poses_tensor,
            self.num_grasps,
            self.grasp_sampling_strategy
        )

    def __getitem__(self, idx):
        """Retrieves an item, returning a fixed number of grasps for an object in a scene view."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.data)}")
        item_data = self.data[idx]
        try:
            scene_data = self._load_scene_data(item_data)
            if not scene_data: return self._create_error_return_dict(item_data, "Failed to load scene data")

            processed_data = self._process_point_cloud_and_masks(scene_data, item_data)
            if not processed_data: return self._create_error_return_dict(item_data, "Failed to process point cloud/masks")

            grasp_object_data = self._load_grasp_and_object_data_for_fixed_grasps(item_data)
            if not grasp_object_data: return self._create_error_return_dict(item_data, "Failed to load grasp/object data")

            transformed_data = self._transform_data_by_mode(
                pc_cam_raw_xyz_rgb=processed_data['scene_pc_with_rgb'],
                grasps_omf=grasp_object_data['hand_pose_tensor'],
                obj_verts=grasp_object_data['obj_verts'],
                R_omf_to_cf_np=grasp_object_data['cam_R_model_to_camera'],
                t_omf_to_cf_np=grasp_object_data['cam_t_model_to_camera'],
                object_mask_np=processed_data['object_mask_np']
            )
            return self._package_data_for_fixed_grasps(item_data, transformed_data, processed_data['object_mask_np'], grasp_object_data['obj_faces'])
        except Exception as e:
            return self._create_error_return_dict(item_data, f"Unexpected error: {e}")

    def _load_grasp_and_object_data_for_fixed_grasps(self, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Loads grasp poses and object mesh data for a fixed number of grasps."""
        scene_id, object_code = item_data['scene_id'], item_data['object_code']
        category_id, depth_view_idx = item_data['category_id_for_masking'], item_data['depth_view_index']
        grasp_indices = item_data.get('grasp_indices')

        hand_pose_tensor = self._get_fixed_number_hand_poses(scene_id, object_code, grasp_indices)
        scene_gt_for_view = self.scene_gt.get(scene_id, {}).get(str(depth_view_idx), [])
        cam_R, cam_t = self._get_camera_transform(scene_gt_for_view, category_id)
        obj_verts, obj_faces = load_object_mesh(self.obj_root_dir, object_code, self.mesh_scale)
        if obj_verts is None or obj_faces is None: return None

        return {
            'hand_pose_tensor': hand_pose_tensor, 'scene_gt_for_view': scene_gt_for_view,
            'cam_R_model_to_camera': cam_R, 'cam_t_model_to_camera': cam_t,
            'obj_verts': obj_verts, 'obj_faces': obj_faces
        }

    def _package_data_for_fixed_grasps(self, item_data: Dict[str, Any], transformed_data: Dict[str, Any],
                                     object_mask_np: np.ndarray, obj_faces: torch.Tensor) -> Dict[str, Any]:
        """Packages transformed data into the final dictionary for a fixed number of grasps."""
        return self._package_data_base(item_data, transformed_data, object_mask_np, obj_faces, is_batch=True)

    def _create_error_return_dict(self, item_data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Creates a standardized error dictionary for the fixed grasp format."""
        return self._create_error_return_dict_base(item_data, error_msg, is_batch=True)

    @staticmethod
    def collate_fn(batch):
        """Collates a batch of items, stacking tensors with a fixed number of grasps."""
        if not batch: return {}
        batch = [item for item in batch if isinstance(item, dict)]
        if not batch: return {}

        collated_output, all_keys = {}, set().union(*[d.keys() for d in batch])

        expected_shapes = {}
        for item in batch:
            for key in ['hand_model_pose', 'se3']:
                if key not in expected_shapes and key in item and isinstance(item[key], torch.Tensor):
                    expected_shapes[key] = item[key].shape
            if 'hand_model_pose' in expected_shapes and 'se3' in expected_shapes: break

        for key in all_keys:
            items = [d.get(key) for d in batch]
            if key in ['hand_model_pose', 'se3']:
                try:
                    valid_tensors = []
                    expected_shape = expected_shapes.get(key)
                    for item in items:
                        if isinstance(item, torch.Tensor) and item.shape == expected_shape:
                            valid_tensors.append(item)
                        elif expected_shape:
                            dtype = item.dtype if isinstance(item, torch.Tensor) else torch.float32
                            device = item.device if isinstance(item, torch.Tensor) else 'cpu'
                            valid_tensors.append(torch.zeros(expected_shape, dtype=dtype, device=device))
                        else: # Fallback if no valid tensor was found
                            shape = (4, 23) if key == 'hand_model_pose' else (4, 4, 4)
                            valid_tensors.append(torch.zeros(shape, dtype=torch.float32))
                    collated_output[key] = torch.stack(valid_tensors) if valid_tensors else torch.empty((0, *expected_shapes.get(key, ())), dtype=torch.float32)
                except (RuntimeError, TypeError, AttributeError):
                    collated_output[key] = items
            elif key in ['obj_verts', 'obj_faces', 'positive_prompt', 'negative_prompts', 'error']:
                collated_output[key] = items
            else:
                try:
                    collated_output[key] = torch.utils.data.dataloader.default_collate(items)
                except (RuntimeError, TypeError, AttributeError):
                    collated_output[key] = items
        return collated_output
