"""
Cache Management Utilities for SceneLeapPro Dataset

This module provides utility functions for cache management,
including file validation, cache path generation, and health checks.
"""

import os
import time
import hashlib
import logging
import h5py
from typing import Optional, Dict, Any, Callable, Tuple
from .dataset_config import CachedDatasetConfig
from .constants import (
    CACHE_CREATION_TIMEOUT, CACHE_FILE_CHECK_INTERVAL, CACHE_VALIDATION_TEST_INDEX,
    CACHE_HEALTH_CHECK_INTERVAL
)
from .distributed_utils import (
    is_distributed_training,
    is_main_process,
    distributed_barrier,
    ensure_directory_exists,
    should_create_cache
)


def wait_for_file(file_path: str, timeout: int = CACHE_CREATION_TIMEOUT, check_interval: float = CACHE_FILE_CHECK_INTERVAL) -> bool:
    """
    Wait for file creation to complete.
    
    Args:
        file_path: File path
        timeout: Timeout in seconds
        check_interval: Check interval in seconds
    
    Returns:
        bool: Whether file was successfully created
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path):
            # File exists, wait a bit more to ensure writing is complete
            time.sleep(0.5)
            try:
                # Try to open file to verify its integrity
                with h5py.File(file_path, 'r') as f:
                    # If successfully opened, file creation is complete
                    return True
            except (OSError, IOError):
                # File might still be writing, continue waiting
                pass
        time.sleep(check_interval)
    return False


def generate_cache_filename(
    root_dir: str,
    mode: str,
    max_points: int,
    enable_cropping: bool,
    coordinate_system_mode: str,
    num_neg_prompts: int,
    max_grasps_per_scene: Optional[int],
    cache_version: str,
    prefix: str = "sceneleappro",
    num_grasps: Optional[int] = None,
    grasp_sampling_strategy: Optional[str] = None,
    use_exhaustive_sampling: bool = False,
    exhaustive_sampling_strategy: Optional[str] = None
) -> str:
    """
    Generate unique cache filename based on parameters.

    Args:
        root_dir: Dataset root directory
        mode: Coordinate system mode
        max_points: Maximum number of points
        enable_cropping: Whether cropping is enabled
        coordinate_system_mode: Coordinate system mode
        num_neg_prompts: Number of negative prompts
        max_grasps_per_scene: Maximum grasps per object (legacy parameter name for compatibility)
        cache_version: Cache version
        prefix: Filename prefix
        num_grasps: Number of grasps per sample (for SceneLeapPlus)
        grasp_sampling_strategy: Grasp sampling strategy (for SceneLeapPlus)
        use_exhaustive_sampling: Whether to use exhaustive sampling
        exhaustive_sampling_strategy: Exhaustive sampling strategy

    Returns:
        str: Generated cache filename
    """
    params_string = (
        f"root_dir={os.path.abspath(root_dir)},"
        f"mode={mode},"
        f"max_points={max_points},"
        f"enable_cropping={enable_cropping},"
        f"coordinate_system_mode={coordinate_system_mode},"
        f"num_neg_prompts={num_neg_prompts},"
        f"max_grasps_per_scene={max_grasps_per_scene},"
        f"cache_version={cache_version}"
    )

    # Add SceneLeapPlus specific parameters if provided
    if num_grasps is not None:
        params_string += f",num_grasps={num_grasps}"
    if grasp_sampling_strategy is not None:
        params_string += f",grasp_sampling_strategy={grasp_sampling_strategy}"
    if use_exhaustive_sampling:
        params_string += f",use_exhaustive_sampling={use_exhaustive_sampling}"
    if exhaustive_sampling_strategy is not None:
        params_string += f",exhaustive_sampling_strategy={exhaustive_sampling_strategy}"

    cache_hash = hashlib.md5(params_string.encode('utf-8')).hexdigest()
    return f'{prefix}_{cache_hash}.h5'


def validate_cache_file(cache_path: str, expected_items: int) -> bool:
    """
    Validate cache file integrity and item count.
    
    Args:
        cache_path: Path to cache file
        expected_items: Expected number of items
        
    Returns:
        bool: True if cache is valid
    """
    if not os.path.exists(cache_path):
        return False
    
    try:
        with h5py.File(cache_path, 'r') as hf_check:
            if len(hf_check) == expected_items:
                logging.info(f"Cache file has {len(hf_check)} items, matching expected {expected_items}. Cache considered valid.")
                return True
            else:
                logging.warning(f"Cache file has {len(hf_check)} items, but expected {expected_items}. Cache considered invalid.")
                return False
    except Exception as e:
        logging.error(f"Error checking cache validity {cache_path}: {e}. Cache considered invalid.")
        return False


def check_cache_health(hf: Optional[h5py.File], num_items: int) -> bool:
    """
    Check cache file health status.

    Args:
        hf: HDF5 file handle
        num_items: Total number of items

    Returns:
        bool: True if cache is healthy
    """
    try:
        if hf is None:
            return False
        # Try to access a random item to check file status
        # Use the correct cache key format: str(idx)
        test_idx = min(CACHE_VALIDATION_TEST_INDEX, num_items - 1)
        test_key = str(test_idx)  # 修复：使用 str(idx) 而不是 f'item_{idx}'
        if test_key in hf:
            # Try to actually read some data to ensure the cache is accessible
            group = hf[test_key]
            # Check if the group has expected structure
            return len(group.keys()) > 0
        return False
    except Exception:
        return False


def get_cache_directory(root_dir: str, cache_subdir: str = 'cache_sceneleappro') -> str:
    """
    Get cache directory path.
    
    Args:
        root_dir: Dataset root directory
        cache_subdir: Cache subdirectory name
        
    Returns:
        str: Cache directory path
    """
    return os.path.join(root_dir, cache_subdir)


def cleanup_cache_file(hf: Optional[h5py.File], cache_path: str = None):
    """
    Clean up cache file resources.
    
    Args:
        hf: HDF5 file handle to close
        cache_path: Cache file path for logging
    """
    if hf is not None:
        try:
            hf.close()
            if cache_path:
                logging.info(f"Cache file {cache_path} closed successfully.")
            else:
                logging.info("Cache file closed successfully.")
        except Exception as e:
            if cache_path:
                logging.warning(f"Error closing cache file {cache_path}: {e}")
            else:
                logging.warning(f"Error closing cache file: {e}")


def get_cache_info(
    cache_path: str,
    cache_loaded: bool,
    num_items: int,
    cache_version: str,
    max_points: int,
    num_neg_prompts: int,
    enable_cropping: bool,
    coordinate_system_mode: str
) -> Dict[str, Any]:
    """
    Get comprehensive cache information.
    
    Args:
        cache_path: Path to cache file
        cache_loaded: Whether cache is loaded
        num_items: Number of items
        cache_version: Cache version
        max_points: Maximum points
        num_neg_prompts: Number of negative prompts
        enable_cropping: Whether cropping is enabled
        coordinate_system_mode: Coordinate system mode
        
    Returns:
        dict: Cache information dictionary
    """
    info = {
        'cache_path': cache_path,
        'cache_loaded': cache_loaded,
        'num_items': num_items,
        'cache_version': cache_version,
        'max_points': max_points,
        'num_neg_prompts': num_neg_prompts,
        'enable_cropping': enable_cropping,
        'coordinate_system_mode': coordinate_system_mode
    }
    
    # Add file size if cache exists
    if os.path.exists(cache_path):
        try:
            file_size = os.path.getsize(cache_path)
            info['cache_file_size_bytes'] = file_size
            info['cache_file_size_mb'] = file_size / (1024 * 1024)
        except OSError:
            info['cache_file_size_bytes'] = None
            info['cache_file_size_mb'] = None
    
    return info


def log_cache_status(cache_path: str, cache_loaded: bool, num_items: int):
    """
    Log cache status information.
    
    Args:
        cache_path: Path to cache file
        cache_loaded: Whether cache is loaded
        num_items: Number of items
    """
    if cache_loaded:
        logging.info(f"Cache loaded successfully: {cache_path} with {num_items} items")
    else:
        logging.warning(f"Cache not loaded: {cache_path}")


def estimate_cache_size(num_items: int, avg_points_per_item: int = 4096) -> Dict[str, float]:
    """
    Estimate cache file size based on dataset parameters.
    
    Args:
        num_items: Number of items in dataset
        avg_points_per_item: Average points per item
        
    Returns:
        dict: Size estimates in different units
    """
    # Rough estimates based on typical data sizes
    # 6D point cloud: 6 * 4 bytes per point
    # Hand pose: 23 * 4 bytes
    # SE3 matrix: 16 * 4 bytes
    # Prompts: ~100 bytes average
    
    point_cloud_size = num_items * avg_points_per_item * 6 * 4  # 6D float32
    hand_pose_size = num_items * 23 * 4  # 23 float32
    se3_size = num_items * 16 * 4  # 4x4 float32
    prompt_size = num_items * 100  # Rough estimate for text
    
    total_bytes = point_cloud_size + hand_pose_size + se3_size + prompt_size
    
    # Add compression factor (HDF5 gzip typically achieves 2-4x compression)
    compressed_bytes = total_bytes / 3
    
    return {
        'uncompressed_bytes': total_bytes,
        'compressed_bytes': compressed_bytes,
        'uncompressed_mb': total_bytes / (1024 * 1024),
        'compressed_mb': compressed_bytes / (1024 * 1024),
        'uncompressed_gb': total_bytes / (1024 * 1024 * 1024),
        'compressed_gb': compressed_bytes / (1024 * 1024 * 1024)
    }


class CacheManager:
    """
    Manages HDF5 cache operations with distributed training support.

    This class centralizes all cache management logic including setup,
    creation, validation, and cleanup operations while maintaining
    compatibility with existing distributed training workflows.
    """

    def __init__(self, config: CachedDatasetConfig, num_items: int):
        """
        Initialize cache manager.

        Args:
            config: Cached dataset configuration
            num_items: Number of items in the dataset
        """
        self.config = config
        self.num_items = num_items
        self.cache_path = self._setup_cache_path()
        self.hf: Optional[h5py.File] = None
        self.cache_loaded = False

        # Health check configuration
        self.health_check_interval = CACHE_HEALTH_CHECK_INTERVAL
        self.last_health_check = 0

    def _setup_cache_path(self) -> str:
        """
        Setup cache directory and file path.

        Returns:
            str: Full path to cache file
        """
        cache_dir = get_cache_directory(self.config.root_dir)
        if not ensure_directory_exists(cache_dir):
            raise RuntimeError(f"Failed to create cache directory {cache_dir}")

        cache_filename = self.config.generate_cache_filename()
        cache_path = os.path.join(cache_dir, cache_filename)

        logging.info(f"CacheManager: Cache file path: {cache_path}")
        return cache_path

    def setup_cache(self) -> Tuple[Optional[h5py.File], bool]:
        """
        Setup cache file, returns (hf, cache_loaded).

        This method preserves the exact distributed training logic
        from the original implementation.

        Returns:
            Tuple[Optional[h5py.File], bool]: (file_handle, cache_loaded_flag)
        """
        logging.info("CacheManager: Setting up cache...")

        # Check if cache already exists and is valid
        if self._is_cache_available():
            logging.info("CacheManager: Valid cache found, loading...")
            try:
                self.hf = h5py.File(self.cache_path, 'r')
                self.cache_loaded = True
                logging.info(f"CacheManager: Cache loaded successfully with {len(self.hf)} items")
                return self.hf, self.cache_loaded
            except Exception as e:
                logging.error(f"CacheManager: Error loading cache: {e}")
                self.hf = None
                self.cache_loaded = False

        # Cache doesn't exist or is invalid, need to create it
        if is_distributed_training():
            # Distributed training logic
            if should_create_cache():
                logging.info("CacheManager: Main process will create cache in distributed training")
                # Main process creates cache, others wait
                if is_main_process():
                    self._create_cache_internal()

                # All processes wait for cache creation
                distributed_barrier()

                # All processes wait for cache file creation, but don't load empty cache
                if wait_for_file(self.cache_path, timeout=CACHE_CREATION_TIMEOUT):
                    # Don't load the cache here - it's empty and needs to be populated
                    # The cache will be loaded after data population in _ensure_cache_populated
                    self.hf = None
                    self.cache_loaded = False
                    logging.info("CacheManager: Empty cache file detected in distributed training, will populate later")
                else:
                    logging.error("CacheManager: Timeout waiting for cache creation in distributed training")
                    self.hf = None
                    self.cache_loaded = False
            else:
                logging.info("CacheManager: Cache creation disabled in distributed training")
                self.hf = None
                self.cache_loaded = False
        else:
            # Single process training
            logging.info("CacheManager: Creating cache in single process mode")
            self._create_cache_internal()

        return self.hf, self.cache_loaded

    def _is_cache_available(self) -> bool:
        """
        Check if cache is available and valid.

        Returns:
            bool: True if cache is available and valid
        """
        return validate_cache_file(self.cache_path, self.num_items)

    def _create_cache_internal(self) -> None:
        """
        Internal method to create cache file.

        This method handles the actual cache creation process.
        Note: This method only creates an empty cache file structure.
        Actual data population should be done via create_cache() method.
        """
        try:
            logging.info(f"CacheManager: Creating empty cache file: {self.cache_path}")

            # Create empty cache file
            with h5py.File(self.cache_path, 'w') as hf_write:
                # Create empty file - actual data will be populated later
                pass

            # Note: We don't load the cache here because it's empty
            # The cache will be loaded after data is populated
            logging.info(f"CacheManager: Empty cache file created: {self.cache_path}")

        except Exception as e:
            logging.error(f"CacheManager: Error creating cache file: {e}")
            # Clean up partial cache file
            if os.path.exists(self.cache_path):
                try:
                    os.remove(self.cache_path)
                    logging.info("CacheManager: Cleaned up partial cache file")
                except Exception as cleanup_e:
                    logging.warning(f"CacheManager: Error cleaning up partial cache: {cleanup_e}")
            raise

    def create_cache(self, data_loader_func: Callable) -> None:
        """
        Create cache using provided data loader function.
        In distributed training, only main process creates cache.

        Args:
            data_loader_func: Function that loads and processes data for caching
        """
        if self.cache_loaded:
            logging.info("CacheManager: Cache already loaded, skipping creation")
            return

        # 分布式训练保护：只有主进程或单进程模式才能创建缓存
        if is_distributed_training() and not is_main_process():
            logging.info("CacheManager: Non-main process in distributed training, skipping cache creation")
            return

        try:
            logging.info(f"CacheManager: Creating cache with {self.num_items} items...")

            # Create cache file and populate it
            data_loader_func(self.cache_path, self.num_items)

            # Validate the created cache
            if self.validate_cache():
                # Load the created cache
                self.hf = h5py.File(self.cache_path, 'r')
                self.cache_loaded = True
                logging.info("CacheManager: Cache creation completed successfully")
            else:
                logging.error("CacheManager: Cache validation failed after creation")
                self.hf = None
                self.cache_loaded = False

        except Exception as e:
            logging.error(f"CacheManager: Error in cache creation: {e}")
            self.hf = None
            self.cache_loaded = False
            raise

    def validate_cache(self) -> bool:
        """
        Validate cache file integrity.

        Returns:
            bool: True if cache is valid
        """
        return validate_cache_file(self.cache_path, self.num_items)

    def periodic_health_check(self, idx: int) -> None:
        """
        Perform periodic cache health checks.

        Args:
            idx: Current item index (used to determine when to check)
        """
        if idx - self.last_health_check >= self.health_check_interval:
            if not check_cache_health(self.hf, self.num_items):
                logging.warning(f"CacheManager: Cache health check failed at index {idx}")
            self.last_health_check = idx

    def cleanup(self) -> None:
        """
        Clean up cache resources.
        """
        cleanup_cache_file(self.hf, self.cache_path)
        self.hf = None
        self.cache_loaded = False

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.

        Returns:
            Dict[str, Any]: Cache information dictionary
        """
        return get_cache_info(
            cache_path=self.cache_path,
            cache_loaded=self.cache_loaded,
            num_items=self.num_items,
            cache_version=self.config.cache_version,
            max_points=self.config.max_points,
            num_neg_prompts=self.config.num_neg_prompts,
            enable_cropping=self.config.enable_cropping,
            coordinate_system_mode=self.config.mode
        )

    def log_status(self) -> None:
        """
        Log cache status information.
        """
        log_cache_status(self.cache_path, self.cache_loaded, self.num_items)

    def is_loaded(self) -> bool:
        """
        Check if cache is loaded.

        Returns:
            bool: True if cache is loaded
        """
        return self.cache_loaded and self.hf is not None

    def get_file_handle(self) -> Optional[h5py.File]:
        """
        Get the HDF5 file handle.

        Returns:
            Optional[h5py.File]: File handle if loaded, None otherwise
        """
        return self.hf if self.cache_loaded else None
