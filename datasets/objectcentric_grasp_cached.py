"""Compact cached dataset for object-centric grasp data."""

import atexit
import copy
import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .objectcentric_grasp_dataset import ObjectCentricGraspDataset
from .utils.cache_utils import CacheManager
from .utils.constants import STANDARD_CACHE_KEYS
from .utils.dataset_config import CachedDatasetConfig
from .utils.distributed_utils import (
    distributed_barrier,
    get_rank_info,
    is_distributed_training,
    is_main_process,
)


class ObjectCentricGraspDatasetCached(ObjectCentricGraspDataset):
    """HDF5-backed cache for ObjectCentricGraspDataset."""

    def __init__(
        self,
        succ_grasp_dir: str,
        obj_root_dir: str,
        *,
        cache_root_dir: Optional[str] = None,
        num_grasps: int = 8,
        max_points: int = 4096,
        max_grasps_per_object: Optional[int] = None,
        mesh_scale: float = 0.1,
        grasp_sampling_strategy: str = "random",
        use_exhaustive_sampling: bool = False,
        exhaustive_sampling_strategy: str = "sequential",
        object_sampling_ratio: float = 0.8,
        table_size: float = 0.4,
        cache_version: str = "v1.0_objectcentric",
        cache_mode: str = "train",
    ):
        logging.info("ObjectCentricGraspDatasetCached: initializing...")
        self.cache_root_dir = cache_root_dir or succ_grasp_dir
        self.cache_version = cache_version
        self.cache_mode = cache_mode

        super().__init__(
            succ_grasp_dir=succ_grasp_dir,
            obj_root_dir=obj_root_dir,
            num_grasps=num_grasps,
            max_points=max_points,
            max_grasps_per_object=max_grasps_per_object,
            mesh_scale=mesh_scale,
            grasp_sampling_strategy=grasp_sampling_strategy,
            use_exhaustive_sampling=use_exhaustive_sampling,
            exhaustive_sampling_strategy=exhaustive_sampling_strategy,
            object_sampling_ratio=object_sampling_ratio,
            table_size=table_size,
        )

        self.num_items = len(self.data)

        os.makedirs(self.cache_root_dir, exist_ok=True)

        self.config = CachedDatasetConfig(
            root_dir=self.cache_root_dir,
            succ_grasp_dir=succ_grasp_dir,
            obj_root_dir=obj_root_dir,
            mode="object_centric",
            max_grasps_per_object=max_grasps_per_object,
            mesh_scale=mesh_scale,
            num_neg_prompts=0,
            enable_cropping=False,
            max_points=max_points,
            cache_version=cache_version,
            cache_mode=cache_mode,
        )

        self.cache_manager = None
        self.hf: Optional[h5py.File] = None
        self.cache_loaded = False
        self.cache_keys = self._build_cache_keys()
        self.error_defaults = self._build_error_defaults()

        if self.num_items == 0:
            logging.warning("ObjectCentricGraspDatasetCached: dataset is empty, skipping cache.")
            return

        self.cache_manager = CacheManager(self._create_modified_config(), self.num_items)
        self.hf, self.cache_loaded = self.cache_manager.setup_cache()
        atexit.register(self._cleanup)

        self._ensure_cache_populated()
        logging.info(
            "ObjectCentricGraspDatasetCached: ready. cache_loaded=%s", self.cache_loaded
        )

    def _create_modified_config(self) -> CachedDatasetConfig:
        modified = copy.deepcopy(self.config)
        suffix = (
            f"_ng{self.num_grasps}"
            f"_gs{self.grasp_sampling_strategy}"
            f"_objratio{self.object_sampling_ratio}"
            f"_table{self.table_size}"
        )
        if self.use_exhaustive_sampling:
            suffix += f"_ex_{self.exhaustive_sampling_strategy}"
        else:
            suffix += "_noex"

        succ_hash = hashlib.md5(os.path.abspath(self.config.succ_grasp_dir).encode("utf-8")).hexdigest()[:8]
        modified.cache_version = f"{self.cache_version}{suffix}_succ{succ_hash}"
        return modified

    def _cleanup(self) -> None:
        if self.cache_manager:
            self.cache_manager.cleanup()
        self.hf = None
        self.cache_loaded = False
        logging.info("ObjectCentricGraspDatasetCached: cleanup complete.")

    def _ensure_cache_populated(self) -> None:
        if not self.cache_manager:
            return

        needs_population = not self.cache_loaded or self._is_cache_empty()
        if not needs_population:
            return

        if is_distributed_training():
            if is_main_process():
                logging.info("ObjectCentricGraspDatasetCached: main process building cache (DDP).")
                self.cache_manager.create_cache(self._create_cache_data)
            distributed_barrier()
            if self.cache_manager._is_cache_available():
                try:
                    if self.hf:
                        self.hf.close()
                    self.hf = h5py.File(self.cache_manager.cache_path, "r")
                    self.cache_loaded = True
                except Exception as exc:
                    logging.error("ObjectCentricGraspDatasetCached: failed to load cache after DDP sync: %s", exc)
                    self.hf = None
                    self.cache_loaded = False
        else:
            logging.info("ObjectCentricGraspDatasetCached: building cache in single-process mode.")
            self.cache_manager.create_cache(self._create_cache_data)
            self.hf = self.cache_manager.get_file_handle()
            self.cache_loaded = self.cache_manager.is_loaded()

    def _is_cache_empty(self) -> bool:
        if not self.hf:
            return True
        try:
            actual = len(self.hf)
            expected = self.num_items
            if actual < expected:
                logging.warning("ObjectCentricGraspDatasetCached: cache incomplete (%d/%d).", actual, expected)
                return True
            return False
        except Exception as exc:
            logging.error("ObjectCentricGraspDatasetCached: failed to inspect cache: %s", exc)
            return True

    def _create_cache_data(self, cache_path: str, num_items: int) -> None:
        with h5py.File(cache_path, "w") as hf:
            for idx in tqdm(range(num_items), desc="Caching ObjectCentric data"):
                try:
                    item = ObjectCentricGraspDataset.__getitem__(self, idx)
                    if "error" in item:
                        self._create_error_group(
                            hf,
                            idx,
                            item.get("error", "Unknown error"),
                            {k: item.get(k, self.error_defaults.get(k)) for k in self.cache_keys},
                        )
                    else:
                        self._create_data_group(hf, idx, item)
                except Exception as exc:
                    logging.error("ObjectCentricGraspDatasetCached: failed to process item %d: %s", idx, exc)
                    self._create_error_group(hf, idx, f"Cache creation error: {exc}", self.error_defaults)

    def _build_cache_keys(self) -> List[str]:
        base_keys = STANDARD_CACHE_KEYS.copy()
        extra_keys = [
            "obj_verts",
            "obj_faces",
            "object_mask",
            "obj_code",
            "scene_id",
            "category_id_from_object_index",
            "depth_view_index",
            "is_exhaustive",
            "chunk_idx",
            "total_chunks",
        ]
        return base_keys + extra_keys

    def _build_error_defaults(self) -> Dict[str, Any]:
        return {
            "scene_pc": torch.zeros((self.max_points, 3), dtype=torch.float32),
            "hand_model_pose": torch.zeros((self.num_grasps, 23), dtype=torch.float32),
            "se3": torch.zeros((self.num_grasps, 4, 4), dtype=torch.float32),
            "positive_prompt": "error_object",
            "negative_prompts": [],
            "obj_verts": torch.zeros((0, 3), dtype=torch.float32),
            "obj_faces": torch.zeros((0, 3), dtype=torch.long),
            "object_mask": torch.zeros((self.max_points,), dtype=torch.bool),
            "obj_code": "unknown",
            "scene_id": "unknown",
            "category_id_from_object_index": 0,
            "depth_view_index": 0,
            "is_exhaustive": False,
            "chunk_idx": 0,
            "total_chunks": 1,
        }

    def _create_data_group(self, hf: h5py.File, idx: int, data: Dict[str, Any]) -> None:
        group = hf.create_group(str(idx))
        group.create_dataset("is_error", data=False)
        for key in self.cache_keys:
            if key in data:
                self._save_value(group, key, data[key])
            else:
                logging.debug("ObjectCentricGraspDatasetCached: missing key '%s'; fallback to default.", key)

    def _create_error_group(
        self,
        hf: h5py.File,
        idx: int,
        error_msg: str,
        error_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        group = hf.create_group(str(idx))
        group.create_dataset("is_error", data=True)
        group.create_dataset("error_msg", data=error_msg.encode("utf-8"))
        values = error_values or self.error_defaults
        for key, value in values.items():
            self._save_value(group, key, value)

    def _save_value(self, group: h5py.Group, key: str, value: Any) -> None:
        try:
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy()
                group.create_dataset(key, data=array, compression="gzip")
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression="gzip")
            elif isinstance(value, list):
                if key == "negative_prompts":
                    encoded = [str(item).encode("utf-8") for item in value]
                    group.create_dataset(key, data=encoded)
                else:
                    group.create_dataset(key, data=np.array(value), compression="gzip")
            elif isinstance(value, str):
                group.create_dataset(key, data=value.encode("utf-8"))
            else:
                group.create_dataset(key, data=value)
        except Exception as exc:
            logging.error("ObjectCentricGraspDatasetCached: failed to persist key '%s': %s", key, exc)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not 0 <= idx < self.num_items:
            raise IndexError(f"Index {idx} out of bounds for dataset length {self.num_items}")

        if not self._is_cache_ready():
            logging.warning(
                "ObjectCentricGraspDatasetCached: cache unavailable, falling back to on-the-fly sampling (rank=%s).",
                get_rank_info(),
            )
            return ObjectCentricGraspDataset.__getitem__(self, idx)

        try:
            self.cache_manager.periodic_health_check(idx)
            data = self._load_item_from_cache(idx)
            if "error" in data:
                return self._format_error_item(data)
            return data
        except Exception as exc:
            logging.error("ObjectCentricGraspDatasetCached: failed to load cache item %d: %s", idx, exc)
            return ObjectCentricGraspDataset.__getitem__(self, idx)

    def _is_cache_ready(self) -> bool:
        return self.cache_manager is not None and self.cache_manager.is_loaded()

    def _load_item_from_cache(self, idx: int) -> Dict[str, Any]:
        assert self.hf is not None
        group = self.hf[str(idx)]
        if group["is_error"][()]:
            error_msg = (
                group["error_msg"][()].decode("utf-8")
                if "error_msg" in group
                else "Cached error"
            )
            data = {key: self._load_dataset(group, key) for key in group.keys()}
            data["error"] = error_msg
            return data

        return {
            key: self._load_dataset(group, key)
            for key in self.cache_keys
            if key in group
        }

    def _load_dataset(self, group: h5py.Group, key: str) -> Any:
        dataset = group[key]
        if key in ["scene_pc", "hand_model_pose", "se3", "obj_verts"]:
            return torch.from_numpy(dataset[:]).float()
        if key == "obj_faces":
            return torch.from_numpy(dataset[:]).long()
        if key == "object_mask":
            return torch.from_numpy(dataset[:]).bool()
        if key in ["positive_prompt", "obj_code", "scene_id"]:
            return dataset[()].decode("utf-8")
        if key == "negative_prompts":
            return [p.decode("utf-8") for p in dataset[:]]
        if key in ["is_exhaustive"]:
            return bool(dataset[()])
        if key in ["chunk_idx", "total_chunks", "category_id_from_object_index", "depth_view_index"]:
            return int(dataset[()])
        return dataset[()]

    def _format_error_item(self, cached_item: Dict[str, Any]) -> Dict[str, Any]:
        formatted = {key: cached_item.get(key, self.error_defaults.get(key)) for key in self.cache_keys}
        formatted["error"] = cached_item.get("error", "Unknown error")
        return formatted

    def get_cache_info(self) -> Dict[str, Any]:
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        return {
            "cache_loaded": False,
            "num_items": self.num_items,
            "error": "Cache manager not initialized",
        }

    def __len__(self) -> int:
        return self.num_items

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return ObjectCentricGraspDataset.collate_fn(batch)
