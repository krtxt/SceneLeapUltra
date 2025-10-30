import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .utils.collate_utils import collate_batch_data
from .utils.data_processing_utils import load_objectcentric_hand_pose_data
from .utils.grasp_sampling_utils import (generate_exhaustive_chunks,
                                         sample_grasps_from_available,
                                         sample_indices_from_available)
from .utils.io_utils import load_object_mesh_with_textures
from .utils.pointcloud_utils import (create_table_plane,
                                     sample_points_from_mesh_separated)
from .utils.transform_utils import (apply_object_pose_to_vertices,
                                    center_hand_poses_xy, center_points_xy,
                                    create_se3_matrices_from_pose_batch,
                                    reorder_hand_pose_batch)


class ObjectCentricGraspDataset(Dataset):
    """
    使用居中归一化策略的手-物对数据集（无场景信息）。

    数据流程：
    1. 从成功抓取目录加载 hand poses 与 obj pose（GW坐标系）
    2. 使用 obj pose 将 mesh 从 OMF 转到 GW
    3. 以 obj pose 的平移部分为中心，对物体与抓取在 XY 平面居中
    4. 构造桌面平面并从物体 + 桌面 mesh 采样点云
    5. 返回固定数量抓取的 hand_model_pose 与 SE(3) 矩阵
    """

    def __init__(
        self,
        succ_grasp_dir: str,
        obj_root_dir: str,
        num_grasps: int = 8,
        max_points: int = 4096,
        max_grasps_per_object: Optional[int] = None,
        mesh_scale: float = 0.1,
        grasp_sampling_strategy: str = "random",
        use_exhaustive_sampling: bool = False,
        exhaustive_sampling_strategy: str = "sequential",
        object_sampling_ratio: float = 0.8,
        table_size: float = 0.4,
    ):
        self.succ_grasp_dir = succ_grasp_dir
        self.obj_root_dir = obj_root_dir
        self.num_grasps = num_grasps
        self.max_points = max_points
        self.max_grasps_per_object = max_grasps_per_object
        self.mesh_scale = mesh_scale
        self.grasp_sampling_strategy = grasp_sampling_strategy
        self.use_exhaustive_sampling = use_exhaustive_sampling
        self.exhaustive_sampling_strategy = exhaustive_sampling_strategy
        self.object_sampling_ratio = object_sampling_ratio
        self.table_size = table_size

        self._validate_init_params()

        self.hand_pose_data, self.obj_pose_data = load_objectcentric_hand_pose_data(
            self.succ_grasp_dir
        )
        logging.info(
            "ObjectCentricGraspDataset: Loaded %d objects with hand poses.",
            len(self.hand_pose_data),
        )

        if self.max_grasps_per_object is not None:
            self._apply_max_grasps_limit()

        if self.use_exhaustive_sampling:
            self.data = self._build_exhaustive_data_index()
            logging.info(
                "ObjectCentricGraspDataset: Using exhaustive sampling with %d items.",
                len(self.data),
            )
        else:
            self.data = self._build_data_index()
            logging.info(
                "ObjectCentricGraspDataset: Using traditional sampling with %d items.",
                len(self.data),
            )

    def _validate_init_params(self) -> None:
        if self.num_grasps <= 0:
            raise ValueError(f"num_grasps must be positive, got {self.num_grasps}")

        valid_strategies = {
            "random",
            "first_n",
            "repeat",
            "farthest_point",
            "nearest_point",
            "chunk_farthest_point",
            "chunk_nearest_point",
        }
        if self.grasp_sampling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid grasp_sampling_strategy: {self.grasp_sampling_strategy}"
            )

        valid_exhaustive = {
            "sequential",
            "random",
            "interleaved",
            "chunk_farthest_point",
            "chunk_nearest_point",
        }
        if self.exhaustive_sampling_strategy not in valid_exhaustive:
            raise ValueError(
                f"Invalid exhaustive_sampling_strategy: {self.exhaustive_sampling_strategy}"
            )

        if not (0.0 < self.object_sampling_ratio <= 1.0):
            raise ValueError(
                f"object_sampling_ratio must be in (0, 1], got {self.object_sampling_ratio}"
            )

        if self.table_size <= 0:
            raise ValueError(f"table_size must be positive, got {self.table_size}")

    def _apply_max_grasps_limit(self) -> None:
        """按照策略裁剪每个物体的抓取数量上限。"""
        for object_code, poses in list(self.hand_pose_data.items()):
            if not isinstance(poses, torch.Tensor) or poses.numel() == 0:
                continue
            if poses.shape[0] <= self.max_grasps_per_object:
                continue

            sampled_indices = sample_indices_from_available(
                range(poses.shape[0]),
                poses,
                self.max_grasps_per_object,
                self.grasp_sampling_strategy,
            )
            index_tensor = torch.tensor(sampled_indices, dtype=torch.long)
            self.hand_pose_data[object_code] = poses.index_select(0, index_tensor)

    def _build_data_index(self) -> List[Dict[str, Any]]:
        """构建标准数据索引，每个 item 对应一个物体。"""
        data_index: List[Dict[str, Any]] = []
        for object_code, poses in self.hand_pose_data.items():
            if not isinstance(poses, torch.Tensor) or poses.shape[0] == 0:
                continue
            data_index.append(
                {
                    "object_code": object_code,
                    "grasp_indices": None,
                    "is_exhaustive": False,
                    "chunk_idx": 0,
                    "total_chunks": 1,
                }
            )

        if not data_index:
            logging.warning("ObjectCentricGraspDataset data_index is empty after build.")
        return data_index

    def _build_exhaustive_data_index(self) -> List[Dict[str, Any]]:
        """构建穷尽数据索引以提升抓取利用率。"""
        exhaustive_data: List[Dict[str, Any]] = []
        total_available = 0
        total_used = 0

        for object_code, poses in self.hand_pose_data.items():
            if not isinstance(poses, torch.Tensor):
                continue

            num_available = poses.shape[0]
            total_available += num_available
            if num_available == 0:
                continue

            if num_available < self.num_grasps:
                exhaustive_data.append(
                    {
                        "object_code": object_code,
                        "grasp_indices": None,
                        "is_exhaustive": False,
                        "chunk_idx": 0,
                        "total_chunks": 1,
                    }
                )
                continue

            chunk_indices = generate_exhaustive_chunks(
                poses, self.num_grasps, self.exhaustive_sampling_strategy
            )
            for chunk_idx, indices in enumerate(chunk_indices):
                exhaustive_data.append(
                    {
                        "object_code": object_code,
                        "grasp_indices": indices,
                        "is_exhaustive": True,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunk_indices),
                    }
                )
                total_used += len(indices)

        if total_available > 0:
            utilization = total_used / total_available
            logging.info(
                "Exhaustive sampling stats: Available=%d, Used=%d, Utilization=%.1f%%",
                total_available,
                total_used,
                utilization * 100,
            )
        else:
            logging.warning(
                "ObjectCentricGraspDataset: no available grasps found during exhaustive build."
            )
        return exhaustive_data

    def _get_fixed_number_hand_poses(
        self, object_code: str, grasp_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """返回指定物体的固定数量抓取姿态。"""
        default_dim = 23
        all_poses_tensor = self.hand_pose_data.get(object_code)

        if (
            not isinstance(all_poses_tensor, torch.Tensor)
            or all_poses_tensor.ndim != 2
            or all_poses_tensor.shape[1] != default_dim
        ):
            return torch.zeros((self.num_grasps, default_dim), dtype=torch.float32)

        if grasp_indices:
            valid_indices = [
                idx for idx in grasp_indices if 0 <= idx < all_poses_tensor.shape[0]
            ]
            if len(valid_indices) >= self.num_grasps:
                index_tensor = torch.tensor(valid_indices[: self.num_grasps], dtype=torch.long)
                return all_poses_tensor.index_select(0, index_tensor)

        return sample_grasps_from_available(
            all_poses_tensor, self.num_grasps, self.grasp_sampling_strategy
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of bounds for dataset with length {len(self.data)}"
            )

        item = self.data[idx]
        object_code = item["object_code"]
        grasp_indices = item.get("grasp_indices")

        try:
            obj_verts_omf, obj_faces, obj_textures = load_object_mesh_with_textures(
                self.obj_root_dir, object_code, self.mesh_scale
            )
            if obj_verts_omf is None or obj_faces is None:
                return self._create_error_return_dict(object_code, "Failed to load mesh")

            obj_pose = self.obj_pose_data.get(object_code)
            if obj_pose is None:
                return self._create_error_return_dict(object_code, "obj_pose not found")

            obj_pose = obj_pose.to(obj_verts_omf.device, dtype=obj_verts_omf.dtype)
            obj_verts_gw = apply_object_pose_to_vertices(obj_verts_omf, obj_pose)

            hand_pose_tensor_gw = self._get_fixed_number_hand_poses(
                object_code, grasp_indices
            )

            center = obj_pose[:3]
            obj_verts_centered = center_points_xy(obj_verts_gw, center)
            hand_pose_centered = center_hand_poses_xy(hand_pose_tensor_gw, center)

            table_device = (
                obj_verts_centered.device
                if obj_verts_centered.numel() > 0
                else hand_pose_centered.device
            )
            table_dtype = (
                obj_verts_centered.dtype
                if obj_verts_centered.numel() > 0
                else hand_pose_centered.dtype
            )
            table_verts, table_faces = create_table_plane(
                self.table_size, table_device, table_dtype, table_z=0.0
            )

            scene_pc_centered, object_mask = sample_points_from_mesh_separated(
                obj_verts_centered,
                obj_faces,
                table_verts,
                table_faces,
                self.max_points,
                self.object_sampling_ratio,
                obj_textures=obj_textures,
                return_rgb=True,
                table_rgb=(0.6, 0.6, 0.6),
            )

            positions = hand_pose_centered[:, :3]
            quaternions = hand_pose_centered[:, 3:7]
            se3_matrices = create_se3_matrices_from_pose_batch(positions, quaternions)
            hand_model_pose = reorder_hand_pose_batch(hand_pose_centered)

            return {
                "scene_pc": scene_pc_centered,
                "hand_model_pose": hand_model_pose,
                "se3": se3_matrices,
                "obj_verts": obj_verts_centered,
                "obj_faces": obj_faces,
                "obj_code": object_code,
                "scene_id": object_code,
                "depth_view_index": 0,
                "category_id_from_object_index": 0,
                "positive_prompt": object_code,
                "negative_prompts": [],
                "object_mask": object_mask,
                "is_exhaustive": item.get("is_exhaustive", False),
                "chunk_idx": item.get("chunk_idx", 0),
                "total_chunks": item.get("total_chunks", 1),
            }

        except Exception as exc:
            logging.warning(
                "Error loading item %d (object: %s): %s", idx, object_code, exc
            )
            return self._create_error_return_dict(object_code, f"Unexpected error: {exc}")

    def _create_error_return_dict(
        self, object_code: str, error_msg: str
    ) -> Dict[str, Any]:
        """统一的错误返回结构。"""
        return {
            "scene_pc": torch.zeros((self.max_points, 6), dtype=torch.float32),
            "hand_model_pose": torch.zeros((self.num_grasps, 23), dtype=torch.float32),
            "se3": torch.zeros((self.num_grasps, 4, 4), dtype=torch.float32),
            "obj_verts": torch.zeros((0, 3), dtype=torch.float32),
            "obj_faces": torch.zeros((0, 3), dtype=torch.long),
            "obj_code": object_code,
            "scene_id": object_code,
            "depth_view_index": 0,
            "category_id_from_object_index": 0,
            "positive_prompt": object_code,
            "negative_prompts": [],
            "object_mask": torch.zeros((self.max_points,), dtype=torch.bool),
            "error": error_msg,
            "is_exhaustive": False,
            "chunk_idx": 0,
            "total_chunks": 1,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = collate_batch_data(batch)
        # 对齐 SceneLeapPlus：将 object_mask 从 list 堆叠为张量，便于下游使用 .dim()
        try:
            masks = collated.get("object_mask")
            if isinstance(masks, list) and masks:
                collated["object_mask"] = torch.utils.data.dataloader.default_collate(masks)
        except Exception:
            # 兜底：保持 collated 不变
            pass
        return collated
