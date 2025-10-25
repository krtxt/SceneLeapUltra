"""
SceneLeapPlus Cached Dataset Implementation
Efficient HDF5-based cached version supporting a fixed number of grasps per sample.
"""

import atexit
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
    from datasets.sceneleappro_cached import _BaseCachedDataset
    from datasets.utils.cache_utils import (CacheManager, check_cache_health,
                                            cleanup_cache_file,
                                            generate_cache_filename,
                                            get_cache_directory,
                                            get_cache_info, log_cache_status,
                                            validate_cache_file, wait_for_file)
    from datasets.utils.collate_utils import BatchCollator
    from datasets.utils.common_utils import CameraInfo
    from datasets.utils.constants import (DEFAULT_CACHE_VERSION,
                                          DEFAULT_ENABLE_CROPPING,
                                          DEFAULT_FORMATCH_CACHE_VERSION,
                                          DEFAULT_MAX_GRASPS_PER_OBJECT,
                                          DEFAULT_MAX_POINTS,
                                          DEFAULT_MESH_SCALE, DEFAULT_MODE,
                                          DEFAULT_NUM_NEG_PROMPTS)
    from datasets.utils.data_formatters import (DataFormatter,
                                                ForMatchDataFormatter)
    from datasets.utils.dataset_config import CachedDatasetConfig
    from datasets.utils.distributed_utils import (distributed_barrier,
                                                  ensure_directory_exists,
                                                  get_distributed_info,
                                                  get_rank_info,
                                                  is_distributed_training,
                                                  is_main_process,
                                                  should_create_cache)
    from datasets.utils.error_utils import (handle_loading_exception,
                                            log_dataset_warning)
    from datasets.utils.hdf5_utils import (create_data_group,
                                           create_error_group,
                                           get_default_cache_keys,
                                           get_default_error_values,
                                           load_dataset_from_group,
                                           load_item_from_cache,
                                           save_value_to_group)
except ImportError:
    logging.warning("Could not import SceneLeapPlusDataset. Using dummy implementation.")


class SceneLeapPlusDatasetCached(_BaseCachedDataset, SceneLeapPlusDataset):
    """
    Cached version of SceneLeapPlusDataset using HDF5.

    This class provides an efficient, cached data loading mechanism for the SceneLeapPlus dataset.
    It supports 6D point cloud data, text conditions, a fixed number of grasps per sample,
    and object mesh data, with optimizations for distributed training and memory usage.
    """

    def __init__(
        self,
        root_dir: str,
        succ_grasp_dir: str,
        obj_root_dir: str,
        num_grasps: int = 8,
        mode: str = DEFAULT_MODE,
        max_grasps_per_object: Optional[int] = DEFAULT_MAX_GRASPS_PER_OBJECT,
        mesh_scale: float = DEFAULT_MESH_SCALE,
        num_neg_prompts: int = DEFAULT_NUM_NEG_PROMPTS,
        enable_cropping: bool = DEFAULT_ENABLE_CROPPING,
        max_points: int = DEFAULT_MAX_POINTS,
        grasp_sampling_strategy: str = "random",
        cache_version: str = "v1.0_plus",
        cache_mode: str = "train",
        use_exhaustive_sampling: bool = False,
        exhaustive_sampling_strategy: str = "sequential",
        use_object_mask: bool = True,
        use_negative_prompts: bool = True
    ):
        """
        Initializes the cached SceneLeapPlus dataset.

        Args:
            root_dir: Dataset root directory.
            succ_grasp_dir: Directory with successful grasp data.
            obj_root_dir: Directory with object mesh data.
            num_grasps: Fixed number of grasps per sample.
            mode: Coordinate system mode.
            max_grasps_per_object: Maximum grasps per object.
            mesh_scale: Scale factor for object meshes.
            num_neg_prompts: Number of negative prompts.
            enable_cropping: Whether to enable point cloud cropping.
            max_points: Maximum number of point cloud points.
            grasp_sampling_strategy: Grasp sampling strategy.
            cache_version: Cache version identifier.
            cache_mode: Cache mode ("train" or "val").
            use_exhaustive_sampling: Whether to use exhaustive sampling for 100% data utilization.
            exhaustive_sampling_strategy: Strategy for exhaustive sampling.
            use_object_mask: Whether to include object_mask in returned data. When True,
                           concatenates object_mask to scene_pc (xyz+rgb+mask).
            use_negative_prompts: Whether to include negative_prompts in returned data.
        """
        logging.info("SceneLeapPlusDatasetCached: Initializing...")

        self.num_grasps = num_grasps
        self.grasp_sampling_strategy = grasp_sampling_strategy
        self.cache_mode = cache_mode
        self.use_exhaustive_sampling = use_exhaustive_sampling
        self.exhaustive_sampling_strategy = exhaustive_sampling_strategy
        self.use_object_mask = use_object_mask
        self.use_negative_prompts = use_negative_prompts

        SceneLeapPlusDataset.__init__(
            self,
            root_dir=root_dir,
            succ_grasp_dir=succ_grasp_dir,
            obj_root_dir=obj_root_dir,
            num_grasps=num_grasps,
            mode=mode,
            max_grasps_per_object=max_grasps_per_object,
            mesh_scale=mesh_scale,
            num_neg_prompts=num_neg_prompts,
            enable_cropping=enable_cropping,
            max_points=max_points,
            grasp_sampling_strategy=grasp_sampling_strategy,
            use_exhaustive_sampling=use_exhaustive_sampling,
            exhaustive_sampling_strategy=exhaustive_sampling_strategy
        )

        config = CachedDatasetConfig(
            root_dir=root_dir,
            succ_grasp_dir=succ_grasp_dir,
            obj_root_dir=obj_root_dir,
            mode=mode,
            max_grasps_per_object=max_grasps_per_object,
            mesh_scale=mesh_scale,
            num_neg_prompts=num_neg_prompts,
            enable_cropping=enable_cropping,
            max_points=max_points,
            cache_version=cache_version,
            cache_mode=cache_mode
        )

        self.config = config
        self.num_items = len(self.data)
        self._setup_cache_system()

        self.succ_grasp_dir = succ_grasp_dir
        self.obj_root_dir = obj_root_dir
        self.max_grasps_per_object = max_grasps_per_object
        self.mesh_scale = mesh_scale
        self.num_neg_prompts = num_neg_prompts
        self.enable_cropping = enable_cropping
        self.max_points = max_points
        self.cache_version = cache_version
        self.coordinate_system_mode = mode

        self._ensure_cache_populated()

    def _setup_cache_system(self) -> None:
        """Sets up the cache manager and file handling."""
        if self.num_items > 0:
            self.cache_manager = self._create_sceneleapplus_cache_manager()
            self.hf, self.cache_loaded = self.cache_manager.setup_cache()
            atexit.register(self._cleanup)
            logging.info(f"SceneLeapPlusDatasetCached: Cache system setup. Loaded: {self.cache_loaded}")
        else:
            logging.warning("SceneLeapPlusDatasetCached: No items in dataset; cache will be empty.")
            self.cache_manager = None
            self.hf = None
            self.cache_loaded = False

    def _create_sceneleapplus_cache_manager(self):
        """Creates a cache manager with a SceneLeapPlus-specific cache filename."""
        modified_config = self._create_modified_config()
        cache_manager = CacheManager(modified_config, self.num_items)
        logging.info(f"SceneLeapPlusDatasetCached: Using modified config for cache generation.")
        return cache_manager

    def _create_modified_config(self):
        """Creates a modified config to include SceneLeapPlus parameters in the cache hash."""
        import copy
        modified_config = copy.deepcopy(self.config)

        original_version = modified_config.cache_version
        sceneleapplus_suffix = f"_ng{self.num_grasps}_gs{self.grasp_sampling_strategy}"
        sceneleapplus_suffix += f"_ex{self.exhaustive_sampling_strategy}" if self.use_exhaustive_sampling else "_noex"
        # Note: use_object_mask and use_negative_prompts are NOT included in hash
        # because they only affect data return format, not cache content
        modified_config.cache_version = original_version + sceneleapplus_suffix

        logging.info(f"SceneLeapPlusDatasetCached: Modified cache version: {modified_config.cache_version}")
        return modified_config

    def _is_cache_available(self) -> bool:
        """Checks if the cache is available and loaded."""
        return self.cache_manager.is_loaded() if self.cache_manager else False

    def _periodic_health_check(self, idx: int) -> None:
        """Performs periodic cache health checks."""
        if self.cache_manager:
            self.cache_manager.periodic_health_check(idx)

    def _cleanup(self) -> None:
        """Cleans up resources, closing the cache file."""
        if self.cache_manager:
            self.cache_manager.cleanup()
            self.cache_manager = None
        self.hf = None
        self.cache_loaded = False
        logging.info("SceneLeapPlusDatasetCached: Cleanup complete.")

    def get_cache_info(self) -> Dict[str, Any]:
        """Returns information about the cache."""
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        return {
            'cache_loaded': False,
            'num_items': self.num_items if hasattr(self, 'num_items') else 0,
            'error': 'Cache manager not initialized'
        }

    def _ensure_cache_populated(self) -> None:
        """Ensures the cache is populated with data if it's missing or incomplete."""
        if not self.cache_manager:
            return

        needs_population = not self.cache_loaded or self._is_cache_empty()
        if not needs_population:
            logging.info("SceneLeapPlusDatasetCached: Cache already populated.")
            return

        if is_distributed_training():
            if is_main_process():
                logging.info("SceneLeapPlusDatasetCached: Main process creating cache in DDP.")
                self.cache_manager.create_cache(self._create_cache_data)
            distributed_barrier()
            if self.cache_manager._is_cache_available():
                try:
                    if self.hf: self.hf.close()
                    self.hf = h5py.File(self.cache_manager.cache_path, 'r')
                    self.cache_loaded = True
                    logging.info(f"SceneLeapPlusDatasetCached: Cache loaded after DDP creation. Items: {len(self.hf)}")
                except Exception as e:
                    logging.error(f"SceneLeapPlusDatasetCached: Error loading cache after DDP creation: {e}")
                    self.hf, self.cache_loaded = None, False
        else:
            logging.info("SceneLeapPlusDatasetCached: Cache requires population, creating data...")
            self.cache_manager.create_cache(self._create_cache_data)
            self.hf = self.cache_manager.get_file_handle()
            self.cache_loaded = self.cache_manager.is_loaded()
            logging.info(f"SceneLeapPlusDatasetCached: Cache populated and loaded. Status: {self.cache_loaded}")

    def _is_cache_empty(self) -> bool:
        """Checks if the cache HDF5 file is empty or incomplete."""
        if not self.hf:
            return True
        try:
            actual_items, expected_items = len(self.hf), self.num_items
            if actual_items < expected_items:
                logging.warning(f"SceneLeapPlusDatasetCached: Cache incomplete ({actual_items}/{expected_items} items).")
                return True
            logging.info(f"SceneLeapPlusDatasetCached: Cache contains {actual_items} items as expected.")
            return False
        except Exception as e:
            logging.error(f"SceneLeapPlusDatasetCached: Error checking cache status: {e}")
            return True

    def _create_cache_data(self, cache_path: str, num_items: int) -> None:
        """
        Creates and populates the HDF5 cache file with SceneLeapPlus data.

        Args:
            cache_path: Path to the HDF5 cache file.
            num_items: Number of items to cache.
        """
        logging.info(f"SceneLeapPlusDatasetCached: Populating cache with {num_items} items...")
        keys_to_cache = self._get_sceneleapplus_cache_keys()
        default_error_values = self._get_sceneleapplus_default_error_values()

        with h5py.File(cache_path, 'w') as hf:
            for idx in tqdm(range(num_items), desc="Caching SceneLeapPlus data"):
                try:
                    full_return_dict = SceneLeapPlusDataset.__getitem__(self, idx)
                    if 'error' in full_return_dict:
                        error_msg = str(full_return_dict.get('error', 'Unknown error'))
                        error_values = {key: full_return_dict.get(key, default_error_values[key]) for key in keys_to_cache}
                        self._create_sceneleapplus_error_group(hf, idx, error_msg, error_values)
                    else:
                        self._create_sceneleapplus_data_group(hf, idx, full_return_dict, keys_to_cache)
                except Exception as e:
                    logging.error(f"SceneLeapPlusDatasetCached: Error processing item {idx}: {e}")
                    self._create_sceneleapplus_error_group(hf, idx, f"Cache creation error: {e}")
        logging.info("SceneLeapPlusDatasetCached: Cache creation complete.")

    def _get_sceneleapplus_cache_keys(self) -> List[str]:
        """Returns the list of keys to cache for the SceneLeapPlus dataset."""
        base_keys = [
            'scene_pc', 'object_mask', 'hand_model_pose', 'se3',
            'positive_prompt', 'negative_prompts', 'obj_verts', 'obj_faces'
        ]
        if self.cache_mode in ["val", "test"]:
            base_keys.extend(['obj_code', 'scene_id', 'category_id_from_object_index', 'depth_view_index'])
        return base_keys

    def _get_sceneleapplus_default_error_values(self) -> Dict[str, Any]:
        """Returns default values for error cases in the SceneLeapPlus dataset."""
        default_values = {
            'scene_pc': torch.zeros((0, 6), dtype=torch.float32),
            'object_mask': torch.zeros((0), dtype=torch.bool),
            'hand_model_pose': torch.zeros((self.num_grasps, 23), dtype=torch.float32),
            'se3': torch.zeros((self.num_grasps, 4, 4), dtype=torch.float32),
            'obj_verts': torch.zeros((0, 3), dtype=torch.float32),
            'obj_faces': torch.zeros((0, 3), dtype=torch.long),
            'positive_prompt': 'error_object',
            'negative_prompts': [''] * self.num_neg_prompts
        }
        if self.cache_mode in ["val", "test"]:
            default_values.update({
                'obj_code': 'unknown', 'scene_id': 'unknown',
                'category_id_from_object_index': -1, 'depth_view_index': -1
            })
        return default_values

    def _create_sceneleapplus_data_group(self, hf: h5py.File, idx: int, data_dict: Dict[str, Any], keys_to_cache: List[str]):
        """Creates an HDF5 group for a normal data item."""
        try:
            group = hf.create_group(str(idx))
            group.create_dataset('is_error', data=False)
            for key in keys_to_cache:
                if key in data_dict:
                    self._save_sceneleapplus_value_to_group(group, key, data_dict[key])
                else:
                    logging.warning(f"SceneLeapPlusDatasetCached: Key '{key}' not found for item {idx}.")
        except Exception as e:
            logging.error(f"Error creating SceneLeapPlus data group for item {idx}: {e}")

    def _create_sceneleapplus_error_group(self, hf: h5py.File, idx: int, error_msg: str, error_values: Optional[Dict[str, Any]] = None):
        """Creates an HDF5 group for an error case."""
        try:
            group = hf.create_group(str(idx))
            group.create_dataset('is_error', data=True)
            group.create_dataset('error_msg', data=error_msg.encode('utf-8'))
            if error_values is None:
                error_values = self._get_sceneleapplus_default_error_values()
            for key, value in error_values.items():
                self._save_sceneleapplus_value_to_group(group, key, value)
        except Exception as e:
            logging.error(f"Error creating SceneLeapPlus error group for item {idx}: {e}")

    def _save_sceneleapplus_value_to_group(self, group: h5py.Group, key: str, value: Any):
        """Saves a value to an HDF5 group with SceneLeapPlus-specific handling."""
        try:
            if isinstance(value, torch.Tensor):
                group.create_dataset(key, data=value.detach().cpu().numpy(), compression='gzip')
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            elif isinstance(value, str):
                group.create_dataset(key, data=value.encode('utf-8'))
            elif isinstance(value, list) and key == 'negative_prompts':
                group.create_dataset(key, data=[p.encode('utf-8') for p in value])
            else:
                group.create_dataset(key, data=value)
        except Exception as e:
            logging.error(f"Error saving SceneLeapPlus value for key '{key}': {e}")
            # Fallback for failed saves
            if key in ['hand_model_pose', 'se3']:
                shape = (self.num_grasps, 23) if key == 'hand_model_pose' else (self.num_grasps, 4, 4)
                group.create_dataset(key, data=np.zeros(shape), compression='gzip')
            elif key in ['scene_pc', 'obj_verts', 'obj_faces']:
                group.create_dataset(key, data=np.array([]), compression='gzip')
            elif key == 'positive_prompt': group.create_dataset(key, data=b'error_object')
            elif key == 'negative_prompts': group.create_dataset(key, data=[b''] * self.num_neg_prompts)
            elif key in ['obj_code', 'scene_id']: group.create_dataset(key, data=b'unknown')
            elif key in ['category_id_from_object_index', 'depth_view_index']: group.create_dataset(key, data=-1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a data item, formatted for SceneLeapPlus training.

        Returns a dictionary containing scene point cloud, hand poses, SE3 transformations,
        prompts, and object mesh data.
        """
        if not (0 <= idx < self.num_items):
            raise IndexError(f"Index {idx} out of bounds for dataset size {self.num_items}")
        if not self._is_cache_available():
            return self._get_cache_unavailable_error()
        try:
            self._periodic_health_check(idx)
            cached_item_data = self._load_sceneleapplus_item_from_cache(idx)
            return self._process_sceneleapplus_cached_data(cached_item_data, idx)
        except Exception as e:
            return self._handle_sceneleapplus_retrieval_error(idx, e)

    def _get_cache_unavailable_error(self) -> Dict[str, Any]:
        """Returns a standardized error response when the cache is unavailable."""
        logging.error(f"SceneLeapPlusDatasetCached: [{get_rank_info()}] Cache not available.")
        error_data = {**self._get_sceneleapplus_default_error_values(), 'error': "Cache is not available."}
        return self._format_sceneleapplus_error_data(error_data)

    def _load_sceneleapplus_item_from_cache(self, idx: int) -> Dict[str, Any]:
        """Loads a single SceneLeapPlus item from the HDF5 cache."""
        try:
            if self.hf is None: return {'error': "Cache file not available"}
            group = self.hf[str(idx)]
            cached_data = {}
            if group['is_error'][()]:
                cached_data['error'] = group['error_msg'][()].decode('utf-8') if 'error_msg' in group else "Unknown cached error"

            for key in self._get_sceneleapplus_cache_keys():
                if key in group:
                    dataset = group[key]
                    if key in ['scene_pc', 'hand_model_pose', 'se3', 'obj_verts', 'obj_faces', 'object_mask']:
                        numpy_data = dataset[:]
                        if key == 'obj_faces': cached_data[key] = torch.from_numpy(numpy_data).long()
                        elif key == 'object_mask': cached_data[key] = torch.from_numpy(numpy_data).bool()
                        else: cached_data[key] = torch.from_numpy(numpy_data).float()
                    elif key in ['positive_prompt', 'obj_code', 'scene_id']:
                        cached_data[key] = dataset[()].decode('utf-8')
                    elif key == 'negative_prompts':
                        cached_data[key] = [p.decode('utf-8') for p in dataset[:]]
                    else:
                        cached_data[key] = dataset[()]
            return cached_data
        except Exception as e:
            logging.error(f"SceneLeapPlusDatasetCached: Error loading item {idx} from cache: {e}")
            return {'error': f"Cache loading error: {e}"}

    def _process_sceneleapplus_cached_data(self, cached_item_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Processes cached data, formatting it for output."""
        if 'error' in cached_item_data:
            logging.warning(f"SceneLeapPlusDatasetCached: Returning cached error for index {idx}.")
            return self._format_sceneleapplus_error_data(cached_item_data)
        return self._format_sceneleapplus_normal_data(cached_item_data)

    def _format_sceneleapplus_error_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Formats error data with fallback values and conditional field filtering."""
        import torch

        default_values = self._get_sceneleapplus_default_error_values()
        result = {key: cached_item_data.get(key, default_values.get(key)) for key in self._get_sceneleapplus_cache_keys()}
        result['error'] = cached_item_data.get('error', 'Unknown error')

        # Handle object_mask inclusion: keep as separate field or remove if disabled
        if not self.use_object_mask:
            result.pop('object_mask', None)

        # Handle negative_prompts filtering
        if not self.use_negative_prompts:
            result.pop('negative_prompts', None)

        return result

    def _format_sceneleapplus_normal_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Formats normal cached data with conditional object_mask concatenation and field filtering."""
        # Create a copy to avoid modifying the original cached data
        result = cached_item_data.copy()

        # Handle object_mask inclusion: keep separate field or remove when disabled
        if not self.use_object_mask:
            result.pop('object_mask', None)

        # Handle negative_prompts filtering
        if not self.use_negative_prompts:
            result.pop('negative_prompts', None)

        return result

    def _handle_sceneleapplus_retrieval_error(self, idx: int, error: Exception) -> Dict[str, Any]:
        """Handles retrieval errors and returns a standardized error response."""
        logging.error(f"SceneLeapPlusDatasetCached: Error retrieving item {idx}: {error}")
        error_data = {**self._get_sceneleapplus_default_error_values(), 'error': f"Retrieval error: {error}"}
        return self._format_sceneleapplus_error_data(error_data)

    def __len__(self) -> int:
        """Returns the dataset size."""
        return self.num_items

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of items from the dataset.
        This is identical to the original SceneLeapPlusDataset.collate_fn.
        """
        return SceneLeapPlusDataset.collate_fn(batch)
