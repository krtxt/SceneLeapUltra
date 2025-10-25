"""
Error Handling Utilities for SceneLeapPro Dataset

This module provides utility functions for creating error return values
and handling dataset loading failures gracefully.
"""

from typing import Any, Dict, List

import torch


def create_error_return_dict(
    item_data: Dict[str, Any],
    error_msg: str,
    num_neg_prompts: int = 4,
    extract_object_name_func=None,
) -> Dict[str, Any]:
    """
    Create standardized error return dictionary for dataset failures.

    Args:
        item_data: Original item data dictionary
        error_msg: Error message describing the failure
        num_neg_prompts: Number of negative prompts to generate
        extract_object_name_func: Function to extract object name from object code

    Returns:
        Error dictionary with fallback values
    """
    grasp_idx = item_data.get("grasp_npy_idx", -1)

    hand_pose_fallback = torch.zeros(23, dtype=torch.float32)
    se3_fallback = torch.eye(4, dtype=torch.float32)

    # Generate fallback prompts
    object_code = item_data.get("object_code", "unknown")
    if extract_object_name_func:
        object_name = extract_object_name_func(object_code)
    else:
        object_name = object_code.split("_")[0] if "_" in object_code else object_code
    scene_id = item_data.get("scene_id", "unknown")

    error_dict = {
        "obj_code": object_code,
        "scene_pc": torch.zeros((0, 6)),  # 6D for xyz+rgb
        "object_mask": torch.zeros((0), dtype=torch.bool),
        "hand_model_pose": hand_pose_fallback,
        "se3": se3_fallback,
        "scene_id": scene_id,
        "category_id_from_object_index": item_data.get("category_id_for_masking", -1),
        "depth_view_index": item_data.get("depth_view_index", -1),
        "grasp_npy_idx": grasp_idx,
        "obj_verts": torch.zeros((0, 3), dtype=torch.float32),
        "obj_faces": torch.zeros((0, 3), dtype=torch.long),
        "positive_prompt": object_name,
        "negative_prompts": [""] * num_neg_prompts,
        "error": error_msg,
    }
    return error_dict


def create_error_return_tuple_fixed_batch(
    batch_item: Dict[str, Any],
    error_msg: str,
    fixed_batch_size: int,
    num_neg_prompts: int = 4,
) -> tuple:
    """
    Create error return tuple for fixed batch mode failures.

    Args:
        batch_item: Batch item data dictionary
        error_msg: Error message describing the failure
        fixed_batch_size: Fixed batch size for the dataset
        num_neg_prompts: Number of negative prompts to generate

    Returns:
        Tuple of (scene_id, error_batch_data_dict)
    """
    scene_id = batch_item.get("scene_id", "unknown")
    actual_batch_size = batch_item.get("batch_size", fixed_batch_size)

    # Create empty batch data dictionary
    error_batch_data_dict = {
        "obj_codes": ["unknown"] * actual_batch_size,
        "object_masks": [torch.zeros((0,), dtype=torch.bool)] * actual_batch_size,
        "hand_model_poses": torch.zeros((actual_batch_size, 23), dtype=torch.float32),
        "se3_matrices": torch.eye(4, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(actual_batch_size, 1, 1),
        "positive_prompts": ["unknown"] * actual_batch_size,
        "negative_prompts": [[""] * num_neg_prompts] * actual_batch_size,
        "obj_verts": [torch.zeros((0, 3), dtype=torch.float32)] * actual_batch_size,
        "obj_faces": [torch.zeros((0, 3), dtype=torch.long)] * actual_batch_size,
        "category_ids": [-1] * actual_batch_size,
        "grasp_indices": [-1] * actual_batch_size,
        "scene_ids": [scene_id] * actual_batch_size,
        "depth_view_indices": [-1] * actual_batch_size,
        "scene_pc": torch.zeros((0, 6), dtype=torch.float32),
    }

    print(f"Error in fixed-batch dataset loading: {error_msg}")
    return scene_id, error_batch_data_dict


def validate_tensor_shape(
    tensor: torch.Tensor, expected_shape: tuple, tensor_name: str = "tensor"
) -> bool:
    """
    Validate that a tensor has the expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape tuple
        tensor_name: Name of tensor for error reporting

    Returns:
        True if shape matches, False otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        print(f"Warning: {tensor_name} is not a tensor")
        return False

    if tensor.shape != expected_shape:
        print(
            f"Warning: {tensor_name} shape {tensor.shape} doesn't match expected {expected_shape}"
        )
        return False

    return True


def safe_tensor_access(
    tensor: torch.Tensor, index: int, default_value: torch.Tensor
) -> torch.Tensor:
    """
    Safely access tensor at given index with fallback to default value.

    Args:
        tensor: Tensor to access
        index: Index to access
        default_value: Default value to return if access fails

    Returns:
        Tensor value at index or default value
    """
    try:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return default_value

        if tensor.ndim == 0 or index >= tensor.shape[0] or index < 0:
            return default_value

        return tensor[index]
    except (IndexError, RuntimeError):
        return default_value


def log_dataset_warning(message: str, item_data: Dict[str, Any] = None):
    """
    Log dataset warning with optional item context.

    Args:
        message: Warning message
        item_data: Optional item data for context
    """
    if item_data:
        scene_id = item_data.get("scene_id", "unknown")
        obj_code = item_data.get("object_code", "unknown")
        print(f"Dataset Warning [{scene_id}/{obj_code}]: {message}")
    else:
        print(f"Dataset Warning: {message}")


def handle_loading_exception(
    e: Exception, context: str, item_data: Dict[str, Any] = None
) -> str:
    """
    Handle and format loading exceptions with context.

    Args:
        e: Exception that occurred
        context: Context description (e.g., "loading depth image")
        item_data: Optional item data for context

    Returns:
        Formatted error message
    """
    error_msg = f"Failed {context}: {str(e)}"
    log_dataset_warning(error_msg, item_data)
    return error_msg
