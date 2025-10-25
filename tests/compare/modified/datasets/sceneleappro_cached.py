"""
SceneLeapPro Cached Dataset Implementation
Efficient HDF5-based cached version supporting text conditions and 6D point cloud data
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
    from datasets.sceneleappro_dataset import (ForMatchSceneLeapProDataset,
                                               SceneLeapProDataset)
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
    logging.warning("Could not import SceneLeapProDataset. Using dummy implementation for demonstration.")


# Utility functions are now imported from utils modules


class _BaseCachedDataset:
    """
    Base class for cached SceneLeapPro datasets.

    This class extracts common functionality from cached dataset implementations
    to reduce code duplication and improve maintainability while preserving
    100% backward compatibility.
    """

    def __init__(self, config: CachedDatasetConfig, parent_dataset_class: type):
        """
        Initialize base cached dataset.

        Args:
            config: Cached dataset configuration
            parent_dataset_class: Parent dataset class to inherit from
        """
        self.config = config
        self._initialize_parent_dataset(parent_dataset_class)
        self._setup_cache_system()

    def _initialize_parent_dataset(self, parent_class: type) -> None:
        """
        Initialize parent dataset with exact same parameters.

        Args:
            parent_class: Parent dataset class (SceneLeapProDataset or ForMatchSceneLeapProDataset)
        """
        logging.info(f"_BaseCachedDataset: Initializing parent {parent_class.__name__}...")

        # Call parent constructor with identical arguments
        parent_class.__init__(
            self,
            root_dir=self.config.root_dir,
            succ_grasp_dir=self.config.succ_grasp_dir,
            obj_root_dir=self.config.obj_root_dir,
            mode=self.config.mode,
            max_grasps_per_object=self.config.max_grasps_per_object,
            mesh_scale=self.config.mesh_scale,
            num_neg_prompts=self.config.num_neg_prompts,
            enable_cropping=self.config.enable_cropping,
            max_points=self.config.max_points
        )

        # Set num_items from parent dataset length
        self.num_items = parent_class.__len__(self)
        logging.info(f"_BaseCachedDataset: Parent dataset initialized with {self.num_items} items")

    def _setup_cache_system(self) -> None:
        """
        Setup cache manager and file handling.
        """
        if self.num_items > 0:
            # Create cache manager
            self.cache_manager = CacheManager(self.config, self.num_items)

            # Setup cache
            self.hf, self.cache_loaded = self.cache_manager.setup_cache()

            # Register cleanup
            atexit.register(self._cleanup)

            logging.info(f"_BaseCachedDataset: Cache system setup completed. Loaded: {self.cache_loaded}")
        else:
            logging.warning("_BaseCachedDataset: Parent dataset found 0 items. Cache will be empty.")
            self.cache_manager = None
            self.hf = None
            self.cache_loaded = False

    def _is_cache_available(self) -> bool:
        """
        Check cache availability.

        Returns:
            bool: True if cache is available
        """
        if self.cache_manager is None:
            return False
        return self.cache_manager.is_loaded()

    def _periodic_health_check(self, idx: int) -> None:
        """
        Perform periodic cache health checks.

        Args:
            idx: Current item index
        """
        if self.cache_manager is not None:
            self.cache_manager.periodic_health_check(idx)

    def _cleanup(self) -> None:
        """
        Resource cleanup.
        """
        if self.cache_manager is not None:
            self.cache_manager.cleanup()
            self.cache_manager = None
        self.hf = None
        self.cache_loaded = False
        logging.info("_BaseCachedDataset: Cleanup completed")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.

        Returns:
            Dict[str, Any]: Cache information dictionary
        """
        if self.cache_manager is not None:
            return self.cache_manager.get_cache_info()
        else:
            return {
                'cache_loaded': False,
                'num_items': self.num_items if hasattr(self, 'num_items') else 0,
                'error': 'Cache manager not initialized'
            }


class SceneLeapProDatasetCached(_BaseCachedDataset, SceneLeapProDataset):
    """
    Cached version of SceneLeapProDataset using new infrastructure.

    Features:
    - Support for 6D point cloud data (xyz+rgb)
    - Support for text conditions (positive and negative prompts)
    - Efficient HDF5 cache storage
    - Distributed training support
    - Memory-optimized data loading
    """

    def __init__(
        self,
        root_dir: str,
        succ_grasp_dir: str,
        obj_root_dir: str,
        mode: str = DEFAULT_MODE,
        max_grasps_per_object: Optional[int] = DEFAULT_MAX_GRASPS_PER_OBJECT,
        mesh_scale: float = DEFAULT_MESH_SCALE,
        num_neg_prompts: int = DEFAULT_NUM_NEG_PROMPTS,
        enable_cropping: bool = DEFAULT_ENABLE_CROPPING,
        max_points: int = DEFAULT_MAX_POINTS,
        cache_version: str = DEFAULT_CACHE_VERSION
    ):
        """
        Initialize cached dataset with identical signature to preserve API.

        Args:
            root_dir: Dataset root directory
            succ_grasp_dir: Directory containing successful grasp data
            obj_root_dir: Directory containing object mesh data
            mode: Coordinate system mode
            max_grasps_per_object: Maximum number of grasps per object
            mesh_scale: Scale factor for object meshes
            num_neg_prompts: Number of negative prompts
            enable_cropping: Whether to enable point cloud cropping
            max_points: Maximum number of point cloud points
            cache_version: Cache version number
        """
        logging.info("SceneLeapProDatasetCached: Starting initialization...")

        # Create configuration using new infrastructure
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
            cache_mode="train"
        )

        # Initialize using base class infrastructure
        super().__init__(config, SceneLeapProDataset)

        # Store additional attributes for backward compatibility
        self.succ_grasp_dir = succ_grasp_dir
        self.obj_root_dir = obj_root_dir
        self.max_grasps_per_object = max_grasps_per_object
        self.mesh_scale = mesh_scale
        self.num_neg_prompts = num_neg_prompts
        self.enable_cropping = enable_cropping
        self.max_points = max_points
        self.cache_version = cache_version
        self.coordinate_system_mode = mode

        # Create cache if needed
        self._ensure_cache_populated()
    def _ensure_cache_populated(self) -> None:
        """Ensure cache is populated with data if it exists but is empty."""
        if self.cache_manager is not None and not self.cache_loaded:
            # Cache file exists but is not loaded (probably empty)
            # Need to populate it with data
            self.cache_manager.create_cache(self._create_cache_data)

    def _create_cache_data(self, cache_path: str, num_items: int) -> None:
        """
        Create cache file and populate it with data.

        Args:
            cache_path: Path to cache file
            num_items: Number of items to cache
        """
        logging.info(f"SceneLeapProDatasetCached: Populating cache with {num_items} items...")

        # Get default cache keys and error values
        keys_to_cache = get_default_cache_keys()
        default_error_values = get_default_error_values(self.num_neg_prompts)

        with h5py.File(cache_path, 'w') as hf:
            for idx in tqdm(range(num_items), desc=f"Caching data"):
                try:
                    # Call parent class __getitem__ to get fully processed data
                    full_return_dict = SceneLeapProDataset.__getitem__(self, idx)

                    if 'error' in full_return_dict:
                        # Handle error case
                        error_msg = str(full_return_dict.get('error', 'Unknown error'))
                        # Use default values for error case
                        error_values = {key: full_return_dict.get(key, default_error_values[key]) for key in keys_to_cache}
                        create_error_group(hf, idx, error_msg, error_values)
                    else:
                        # Handle normal data
                        create_data_group(hf, idx, full_return_dict, keys_to_cache)

                except Exception as e:
                    logging.error(f"SceneLeapProDatasetCached: Error processing item {idx}: {e}")
                    # Create error group for processing failure
                    create_error_group(hf, idx, f"Cache creation error: {str(e)}")

        logging.info(f"SceneLeapProDatasetCached: Cache creation completed.")
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve data item - must return identical format.

        Returns key fields required for training:
        - scene_pc: 6D point cloud data (xyz+rgb) - PointNet2 feature extraction
        - hand_model_pose: Hand pose - for generating norm_pose training target
        - se3: SE3 transformation matrix - some loss calculations need
        - positive_prompt: Positive prompt - text condition
        - negative_prompts: Negative prompts list - negative prompt loss
        """
        if idx < 0 or idx >= self.num_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.num_items}")

        # Check cache availability using base class method
        if not self._is_cache_available():
            return self._get_cache_unavailable_error()

        try:
            # Perform periodic health check using base class method
            self._periodic_health_check(idx)

            # Load cached data using utility function
            cached_item_data = load_item_from_cache(idx, self.hf)
            return self._process_cached_data(cached_item_data, idx)

        except Exception as e:
            return self._handle_retrieval_error(idx, e)

    def _get_cache_unavailable_error(self) -> Dict[str, Any]:
        """Get error response when cache is unavailable."""
        rank_info = get_rank_info()
        logging.error(f"SceneLeapProDatasetCached: [{rank_info}] Cache not available.")

        default_values = get_default_error_values(self.num_neg_prompts)
        default_values['error'] = "Cache is not available."
        return default_values

    def _process_cached_data(self, cached_item_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process cached data and return formatted result."""
        if 'error' in cached_item_data:
            logging.warning(f"SceneLeapProDatasetCached: Returning cached error for index {idx}.")
            return self._format_error_data(cached_item_data)
        else:
            return self._format_normal_data(cached_item_data)

    def _format_error_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error data with fallback values."""
        return DataFormatter.format_error_data(cached_item_data, self.num_neg_prompts)

    def _format_normal_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format normal cached data."""
        return DataFormatter.format_normal_data(cached_item_data, self.num_neg_prompts)

    def _handle_retrieval_error(self, idx: int, error: Exception) -> Dict[str, Any]:
        """Handle retrieval errors and return error response."""
        logging.error(f"SceneLeapProDatasetCached: Error retrieving item {idx}: {error}")
        default_values = get_default_error_values(self.num_neg_prompts)
        default_values['error'] = f"Retrieval error: {str(error)}"
        return default_values

    def __len__(self) -> int:
        """Returns dataset size - unchanged."""
        return self.num_items

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of items from SceneLeapProDatasetCached.
        Handles proper batching of negative_prompts and other fields for training.
        """
        return BatchCollator.collate_scene_batch(batch)


class ForMatchSceneLeapProDatasetCached(_BaseCachedDataset, ForMatchSceneLeapProDataset):
    """
    Cached version of ForMatchSceneLeapProDataset using new infrastructure.

    Features:
    - Support for 6D point cloud data (xyz+rgb)
    - Support for text conditions (positive and negative prompts)
    - Support for variable-length grasp arrays (N, 23) and SE3 matrices (N, 4, 4)
    - Support for object mesh data (vertices and faces)
    - Efficient HDF5 cache storage with variable-length data handling
    - Distributed training support
    - Memory-optimized data loading for match-based training
    """

    def __init__(
        self,
        root_dir: str,
        succ_grasp_dir: str,
        obj_root_dir: str,
        mode: str = DEFAULT_MODE,
        max_grasps_per_object: Optional[int] = DEFAULT_MAX_GRASPS_PER_OBJECT,
        mesh_scale: float = DEFAULT_MESH_SCALE,
        num_neg_prompts: int = DEFAULT_NUM_NEG_PROMPTS,
        enable_cropping: bool = DEFAULT_ENABLE_CROPPING,
        max_points: int = DEFAULT_MAX_POINTS,
        cache_version: str = DEFAULT_FORMATCH_CACHE_VERSION,
        cache_mode: str = "val"
    ):
        """
        Initialize cached ForMatch dataset with identical signature to preserve API.

        Args:
            root_dir: Dataset root directory
            succ_grasp_dir: Directory containing successful grasp data
            obj_root_dir: Directory containing object mesh data
            mode: Coordinate system mode
            max_grasps_per_object: Maximum number of grasps per object
            mesh_scale: Scale factor for object meshes
            num_neg_prompts: Number of negative prompts
            enable_cropping: Whether to enable point cloud cropping
            max_points: Maximum number of point cloud points
            cache_version: Cache version number
            cache_mode: Cache mode - "val" for validation (7 fields) or "test" for testing (11 fields with IDs)
        """
        logging.info("ForMatchSceneLeapProDatasetCached: Starting initialization...")

        # Create configuration using new infrastructure
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

        # Initialize using base class infrastructure
        super().__init__(config, ForMatchSceneLeapProDataset)

        # Store additional attributes for backward compatibility
        self.max_grasps_per_object = max_grasps_per_object
        self.mesh_scale = mesh_scale
        self.num_neg_prompts = num_neg_prompts
        self.enable_cropping = enable_cropping
        self.max_points = max_points
        self.cache_version = cache_version
        self.cache_mode = cache_mode

        # Create cache if needed
        self._ensure_cache_populated()

    def _ensure_cache_populated(self) -> None:
        """Ensure cache is populated with data if it exists but is empty."""
        if self.cache_manager is not None and not self.cache_loaded:
            # Cache file exists but is not loaded (probably empty)
            # Need to populate it with data
            self.cache_manager.create_cache(self._create_cache_data)

    def _create_cache_data(self, cache_path: str, num_items: int) -> None:
        """
        Create cache file and populate it with ForMatch data.

        Args:
            cache_path: Path to cache file
            num_items: Number of items to cache
        """
        logging.info(f"ForMatchSceneLeapProDatasetCached: Populating cache with {num_items} items...")

        # Get cache keys for ForMatch dataset (includes additional fields)
        keys_to_cache = self._get_formatch_cache_keys()
        default_error_values = self._get_formatch_default_error_values()

        with h5py.File(cache_path, 'w') as hf:
            for idx in tqdm(range(num_items), desc=f"Caching ForMatch data"):
                try:
                    # Call parent class __getitem__ to get fully processed data
                    full_return_dict = ForMatchSceneLeapProDataset.__getitem__(self, idx)

                    if 'error' in full_return_dict:
                        # Handle error case
                        error_msg = str(full_return_dict.get('error', 'Unknown error'))
                        # Use default values for error case
                        error_values = {key: full_return_dict.get(key, default_error_values[key]) for key in keys_to_cache}
                        self._create_formatch_error_group(hf, idx, error_msg, error_values)
                    else:
                        # Handle normal data
                        self._create_formatch_data_group(hf, idx, full_return_dict, keys_to_cache)

                except Exception as e:
                    logging.error(f"ForMatchSceneLeapProDatasetCached: Error processing item {idx}: {e}")
                    # Create error group for processing failure
                    self._create_formatch_error_group(hf, idx, f"Cache creation error: {str(e)}")

        logging.info(f"ForMatchSceneLeapProDatasetCached: Cache creation completed.")

    def _get_formatch_cache_keys(self) -> List[str]:
        """
        Get list of keys to cache for ForMatch training.

        Returns:
            list: Cache keys for ForMatch dataset based on cache_mode
        """
        return ForMatchDataFormatter.get_formatch_cache_keys(self.cache_mode)

    def _get_formatch_default_error_values(self) -> Dict[str, Any]:
        """
        Get default values for error cases in ForMatch dataset.

        Returns:
            dict: Default error values based on cache_mode
        """
        return ForMatchDataFormatter.get_formatch_default_error_values(self.num_neg_prompts, self.cache_mode)

    def _create_formatch_data_group(self, hf: h5py.File, idx: int, data_dict: Dict[str, Any], keys_to_cache: List[str]):
        """Create HDF5 group for normal ForMatch data item."""
        try:
            group = hf.create_group(str(idx))
            group.create_dataset('is_error', data=False)

            for key in keys_to_cache:
                if key in data_dict:
                    value = data_dict[key]
                    self._save_formatch_value_to_group(group, key, value)
                else:
                    logging.warning(f"ForMatchSceneLeapProDatasetCached: Key '{key}' not found in data_dict for item {idx}")

        except Exception as e:
            logging.error(f"Error creating ForMatch data group for item {idx}: {e}")

    def _create_formatch_error_group(self, hf: h5py.File, idx: int, error_msg: str, error_values: Optional[Dict[str, Any]] = None):
        """Create HDF5 group for ForMatch error case."""
        try:
            group = hf.create_group(str(idx))
            group.create_dataset('is_error', data=True)
            group.create_dataset('error_msg', data=error_msg.encode('utf-8'))

            if error_values is None:
                error_values = self._get_formatch_default_error_values()

            for key, value in error_values.items():
                self._save_formatch_value_to_group(group, key, value)

        except Exception as e:
            logging.error(f"Error creating ForMatch error group for item {idx}: {e}")

    def _save_formatch_value_to_group(self, group: h5py.Group, key: str, value: Any):
        """Save value to HDF5 group with ForMatch-specific handling."""
        try:
            if isinstance(value, torch.Tensor):
                # Convert tensor to numpy for HDF5 storage
                numpy_value = value.detach().cpu().numpy()
                group.create_dataset(key, data=numpy_value, compression='gzip')
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression='gzip')
            elif isinstance(value, str):
                # Store string as UTF-8 encoded bytes
                group.create_dataset(key, data=value.encode('utf-8'))
            elif isinstance(value, list):
                if key == 'negative_prompts':
                    # Handle list of strings for negative prompts
                    encoded_prompts = [prompt.encode('utf-8') for prompt in value]
                    group.create_dataset(key, data=encoded_prompts)
                else:
                    # Handle other list types
                    group.create_dataset(key, data=value)
            else:
                # Try to store as-is
                group.create_dataset(key, data=value)

        except Exception as e:
            logging.error(f"Error saving ForMatch value for key '{key}': {e}")
            # Store a placeholder for failed values
            if key in ['scene_pc', 'hand_model_pose', 'se3', 'obj_verts', 'obj_faces']:
                group.create_dataset(key, data=np.array([]), compression='gzip')
            elif key == 'positive_prompt':
                group.create_dataset(key, data=b'error_object')
            elif key == 'negative_prompts':
                group.create_dataset(key, data=[b''] * self.num_neg_prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve data item - must return identical format.

        Returns key fields required for ForMatch training:
        - scene_pc: 6D point cloud data (xyz+rgb) - PointNet2 feature extraction
        - hand_model_pose: All hand poses (N, 23) - for match-based training
        - se3: All SE3 transformation matrices (N, 4, 4) - loss calculations
        - positive_prompt: Positive prompt - text condition
        - negative_prompts: Negative prompts list - negative prompt loss
        - obj_verts: Object vertices (V, 3) - mesh data
        - obj_faces: Object faces (F, 3) - mesh data
        """
        if idx < 0 or idx >= self.num_items:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.num_items}")

        # Check cache availability using base class method
        if not self._is_cache_available():
            return self._get_cache_unavailable_error()

        try:
            # Perform periodic health check using base class method
            self._periodic_health_check(idx)

            # Load cached data using ForMatch-specific loader
            cached_item_data = self._load_formatch_item_from_cache(idx)
            return self._process_formatch_cached_data(cached_item_data, idx)

        except Exception as e:
            return self._handle_formatch_retrieval_error(idx, e)

    def _get_cache_unavailable_error(self) -> Dict[str, Any]:
        """Get error response when cache is unavailable."""
        rank_info = get_rank_info()
        logging.error(f"ForMatchSceneLeapProDatasetCached: [{rank_info}] Cache not available.")

        default_values = self._get_formatch_default_error_values()
        default_values['error'] = "Cache is not available."
        return default_values

    def _load_formatch_item_from_cache(self, idx: int) -> Dict[str, Any]:
        """Load ForMatch item from HDF5 cache."""
        try:
            group = self.hf[str(idx)]
            cached_data = {}

            # Check if this is an error item
            is_error = group['is_error'][()]

            if is_error:
                # Load error information
                error_msg = group['error_msg'][()].decode('utf-8')
                cached_data['error'] = error_msg

            # Load all cached fields
            cache_keys = self._get_formatch_cache_keys()
            for key in cache_keys:
                if key in group:
                    dataset = group[key]

                    if key in ['scene_pc', 'hand_model_pose', 'se3', 'obj_verts', 'obj_faces']:
                        # Convert numpy arrays back to tensors
                        numpy_data = dataset[:]
                        if key == 'obj_faces':
                            cached_data[key] = torch.from_numpy(numpy_data).long()
                        else:
                            cached_data[key] = torch.from_numpy(numpy_data).float()
                    elif key == 'positive_prompt':
                        # Decode UTF-8 string
                        cached_data[key] = dataset[()].decode('utf-8')
                    elif key == 'negative_prompts':
                        # Decode list of UTF-8 strings
                        encoded_prompts = dataset[:]
                        cached_data[key] = [prompt.decode('utf-8') for prompt in encoded_prompts]
                    else:
                        cached_data[key] = dataset[()]

            return cached_data

        except Exception as e:
            logging.error(f"ForMatchSceneLeapProDatasetCached: Error loading item {idx} from cache: {e}")
            return {'error': f"Cache loading error: {str(e)}"}

    def _process_formatch_cached_data(self, cached_item_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process cached data and return formatted result."""
        if 'error' in cached_item_data:
            logging.warning(f"ForMatchSceneLeapProDatasetCached: Returning cached error for index {idx}.")
            return self._format_formatch_error_data(cached_item_data)
        else:
            return self._format_formatch_normal_data(cached_item_data)

    def _format_formatch_error_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error data with fallback values."""
        return ForMatchDataFormatter.format_formatch_error_data(cached_item_data, self.num_neg_prompts, self.cache_mode)

    def _format_formatch_normal_data(self, cached_item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format normal cached data."""
        return ForMatchDataFormatter.format_formatch_normal_data(cached_item_data, self.num_neg_prompts, self.cache_mode)

    def _handle_formatch_retrieval_error(self, idx: int, error: Exception) -> Dict[str, Any]:
        """Handle retrieval errors and return error response."""
        logging.error(f"ForMatchSceneLeapProDatasetCached: Error retrieving item {idx}: {error}")
        default_values = self._get_formatch_default_error_values()
        default_values['error'] = f"Retrieval error: {str(error)}"
        return default_values

    def __len__(self) -> int:
        """Returns dataset size - unchanged."""
        return self.num_items

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of items from ForMatchSceneLeapProDatasetCached.
        'hand_model_pose' and 'se3' are padded to become dense tensors.
        Other tensor fields are handled with standard collation or kept as lists.
        """
        return BatchCollator.collate_formatch_batch(batch)
