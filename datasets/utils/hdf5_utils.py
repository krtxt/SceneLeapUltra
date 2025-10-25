"""
HDF5 Utilities for SceneLeapPro Dataset

This module provides utility functions for HDF5 data serialization and deserialization,
including handling different data types and compression.
"""

import logging
from typing import Any, Dict, List, Union

import h5py
import numpy as np
import torch

from .constants import (DEFAULT_DTYPE, DEFAULT_EMPTY_PROMPT,
                        DEFAULT_ERROR_PROMPT, DEFAULT_LONG_DTYPE,
                        ERROR_POSE_SHAPE, ERROR_SCENE_PC_SHAPE,
                        ERROR_SE3_SHAPE, HDF5_COMPRESSION, STANDARD_CACHE_KEYS,
                        get_default_error_values)


def save_value_to_group(group: h5py.Group, key: str, value: Any):
    """
    Save value to HDF5 group, handling different data types.

    Args:
        group: HDF5 group to save to
        key: Key name for the dataset
        value: Value to save
    """
    try:
        if isinstance(value, torch.Tensor):
            # PyTorch tensor to numpy array
            group.create_dataset(
                key,
                data=value.numpy(),
                compression=HDF5_COMPRESSION,
                compression_opts=9,
            )
        elif isinstance(value, np.ndarray):
            # Numpy array directly
            group.create_dataset(
                key, data=value, compression=HDF5_COMPRESSION, compression_opts=9
            )
        elif isinstance(value, str):
            # String type
            group.create_dataset(key, data=value.encode("utf-8"))
        elif isinstance(value, list):
            if key == "negative_prompts":
                # Negative prompts list, convert to string array
                str_array = [str(item).encode("utf-8") for item in value]
                group.create_dataset(key, data=str_array)
            else:
                # Other list types, try to convert to numpy array
                group.create_dataset(
                    key, data=np.array(value), compression="gzip", compression_opts=9
                )
        elif isinstance(value, (int, float, bool, np.int_, np.float_, np.bool_)):
            # Scalar types
            group.create_dataset(key, data=value)
        else:
            logging.warning(f"Unsupported data type for key '{key}': {type(value)}")
            # Try to convert to string
            group.create_dataset(key, data=str(value).encode("utf-8"))
    except Exception as e:
        logging.error(f"Error saving key '{key}': {e}")


def load_dataset_from_group(group: h5py.Group, key: str) -> Any:
    """
    Load dataset from HDF5 group with appropriate type conversion.

    Args:
        group: HDF5 group to load from
        key: Key name of the dataset

    Returns:
        Loaded data with appropriate type
    """
    try:
        dataset = group[key]

        if key == "scene_pc":
            # 6D point cloud data
            return torch.from_numpy(dataset[:]).float()
        elif key == "hand_model_pose":
            # Hand pose
            return torch.from_numpy(dataset[:]).float()
        elif key == "se3":
            # SE3 transformation matrix
            return torch.from_numpy(dataset[:]).float()
        elif key == "object_mask":
            # Object mask
            return torch.from_numpy(dataset[:]).bool()
        elif key == "positive_prompt":
            # Positive prompt
            return dataset[()].decode("utf-8")
        elif key == "negative_prompts":
            # Negative prompts list
            return [item.decode("utf-8") for item in dataset[:]]
        elif key in ["obj_verts", "obj_faces"]:
            # Object vertices and faces
            if key == "obj_verts":
                return torch.from_numpy(dataset[:]).float()
            else:
                return torch.from_numpy(dataset[:]).long()
        else:
            # Other data types
            try:
                data = dataset[()]
                if isinstance(data, bytes):
                    return data.decode("utf-8")
                else:
                    return data
            except:
                return torch.from_numpy(dataset[:])

    except Exception as e:
        logging.error(f"Error loading key '{key}': {e}")
        return None


def load_item_from_cache(idx: int, hf: h5py.File) -> Dict[str, Any]:
    """
    Load single data item from cache.

    Args:
        idx: Item index
        hf: HDF5 file handle

    Returns:
        dict: Loaded data dictionary
    """
    try:
        group = hf[str(idx)]
        loaded_data = {}

        # Check if error item
        is_error = group["is_error"][()]

        if is_error:
            # Load error information
            error_msg = group["error_msg"][()].decode("utf-8")
            loaded_data["error"] = error_msg

        # Load all data fields
        for key in group.keys():
            if key in ["is_error", "error_msg"]:
                continue

            loaded_value = load_dataset_from_group(group, key)
            if loaded_value is not None:
                loaded_data[key] = loaded_value
            else:
                # 添加详细的调试信息
                logging.error(
                    f"Failed to load key '{key}' for item {idx}, got None value"
                )
                # 检查数据集的详细信息
                try:
                    dataset = group[key]
                    logging.error(
                        f"Dataset '{key}' info: shape={dataset.shape}, dtype={dataset.dtype}"
                    )
                    logging.error(
                        f"Dataset '{key}' first few values: {dataset[:min(5, dataset.size)]}"
                    )
                except Exception as dataset_e:
                    logging.error(f"Cannot inspect dataset '{key}': {dataset_e}")

        return loaded_data

    except Exception as e:
        logging.error(f"Error loading item {idx} from cache: {e}")
        return {"error": f"Cache loading error: {str(e)}"}


def create_error_group(
    hf: h5py.File, idx: int, error_msg: str, default_values: Dict[str, Any] = None
):
    """
    Create error group in HDF5 file.

    Args:
        hf: HDF5 file handle
        idx: Item index
        error_msg: Error message
        default_values: Default values to store for error case
    """
    try:
        group = hf.create_group(str(idx))
        group.create_dataset("is_error", data=True)
        group.create_dataset("error_msg", data=error_msg.encode("utf-8"))

        if default_values:
            for key, value in default_values.items():
                save_value_to_group(group, key, value)

    except Exception as e:
        logging.error(f"Error creating error group for item {idx}: {e}")


def create_data_group(
    hf: h5py.File, idx: int, data_dict: Dict[str, Any], keys_to_cache: List[str]
):
    """
    Create data group in HDF5 file.

    Args:
        hf: HDF5 file handle
        idx: Item index
        data_dict: Data dictionary to save
        keys_to_cache: List of keys to cache
    """
    try:
        group = hf.create_group(str(idx))
        group.create_dataset("is_error", data=False)

        for key in keys_to_cache:
            if key in data_dict:
                value = data_dict[key]
                save_value_to_group(group, key, value)
            else:
                logging.warning(f"Key '{key}' not found in item {idx}")

    except Exception as e:
        logging.error(f"Error creating data group for item {idx}: {e}")


def get_default_cache_keys() -> List[str]:
    """
    Get default list of keys to cache for training.

    Returns:
        list: Default cache keys
    """
    return STANDARD_CACHE_KEYS.copy()


def get_default_error_values(num_neg_prompts: int = 4) -> Dict[str, Any]:
    """
    Get default values for error cases.

    Args:
        num_neg_prompts: Number of negative prompts

    Returns:
        dict: Default error values from constants
    """
    # Use the centralized function from constants
    from .constants import \
        get_default_error_values as get_constants_error_values

    return get_constants_error_values(num_neg_prompts)


def validate_hdf5_file(file_path: str) -> bool:
    """
    Validate HDF5 file can be opened and read.

    Args:
        file_path: Path to HDF5 file

    Returns:
        bool: True if file is valid
    """
    try:
        with h5py.File(file_path, "r") as f:
            # Try to read basic info
            return len(f) >= 0
    except Exception:
        return False


def get_hdf5_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        dict: File information
    """
    info = {
        "exists": False,
        "valid": False,
        "num_groups": 0,
        "file_size_bytes": 0,
        "keys": [],
    }

    try:
        if not os.path.exists(file_path):
            return info

        info["exists"] = True
        info["file_size_bytes"] = os.path.getsize(file_path)

        with h5py.File(file_path, "r") as f:
            info["valid"] = True
            info["num_groups"] = len(f)
            info["keys"] = list(f.keys())[:10]  # First 10 keys as sample

    except Exception as e:
        logging.error(f"Error getting HDF5 file info for {file_path}: {e}")

    return info
