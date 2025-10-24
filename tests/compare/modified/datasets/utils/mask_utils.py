"""
Mask Processing Utilities for SceneLeapPro Dataset

This module provides utility functions for mask processing, including
instance segmentation mask extraction, object mask mapping, and mask validation.
"""

import numpy as np
from typing import List, Dict, Optional


def extract_object_mask(
    instance_mask_image: np.ndarray,
    target_category_id: int,
    view_specific_instance_attributes: List[Dict]
) -> np.ndarray:
    """
    Extract object mask from instance segmentation image.

    Args:
        instance_mask_image: Instance segmentation mask
        target_category_id: Category ID of the target object
        view_specific_instance_attributes: View-specific instance attributes

    Returns:
        Boolean mask indicating target object pixels (flattened)
    """
    mask_value = None

    # Find the mask value for the target category
    for obj_attr in view_specific_instance_attributes:
        if obj_attr.get('category_id') == target_category_id:
            mask_value = obj_attr.get('idx')
            break

    # Create object mask
    if mask_value is not None:
        object_mask = (instance_mask_image == mask_value)
    else:
        object_mask = np.zeros_like(instance_mask_image, dtype=bool)

    return object_mask.flatten()


def validate_mask_correspondence(
    point_cloud_size: int,
    mask_size: int,
    operation_name: str = "mask operation"
) -> bool:
    """
    Validate that mask size corresponds to point cloud size.

    Args:
        point_cloud_size: Number of points in point cloud
        mask_size: Number of elements in mask
        operation_name: Name of operation for error reporting

    Returns:
        True if sizes match, False otherwise
    """
    if point_cloud_size != mask_size:
        print(f"Warning: {operation_name} - Point cloud size ({point_cloud_size}) "
              f"doesn't match mask size ({mask_size})")
        return False
    return True


def resize_mask_to_match_pointcloud(
    mask: np.ndarray,
    target_size: int,
    pad_value: bool = False
) -> np.ndarray:
    """
    Resize mask to match target point cloud size.

    Args:
        mask: Original mask array
        target_size: Target size for the mask
        pad_value: Value to use for padding (if needed)

    Returns:
        Resized mask array
    """
    current_size = len(mask)
    
    if current_size == target_size:
        return mask
    elif current_size > target_size:
        # Truncate mask
        return mask[:target_size]
    else:
        # Pad mask
        padding_size = target_size - current_size
        padding = np.full(padding_size, pad_value, dtype=mask.dtype)
        return np.concatenate([mask, padding])


def combine_object_masks(masks_2d_list: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple 2D object masks into a single unified mask.

    Args:
        masks_2d_list: List of 2D boolean masks

    Returns:
        Combined boolean mask (same shape as input masks)
    """
    if not masks_2d_list:
        return np.array([], dtype=bool)

    # Initialize with first mask
    combined_mask = masks_2d_list[0].copy()

    # Combine with remaining masks using logical OR
    for mask_2d in masks_2d_list[1:]:
        if mask_2d.shape == combined_mask.shape:
            combined_mask |= mask_2d
        else:
            print(f"Warning: Mask shape mismatch - {mask_2d.shape} vs {combined_mask.shape}")

    return combined_mask


def get_object_bounding_box(
    mask_2d: np.ndarray,
    padding: int = 30
) -> tuple:
    """
    Get bounding box coordinates from 2D object mask.

    Args:
        mask_2d: 2D boolean mask
        padding: Padding to add around bounding box

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) or None if no objects found
    """
    if not np.any(mask_2d):
        return None

    # Find object pixels
    object_pixels = np.where(mask_2d)
    if len(object_pixels[0]) == 0:
        return None

    # Calculate bounding box
    min_y, max_y = object_pixels[0].min(), object_pixels[0].max()
    min_x, max_x = object_pixels[1].min(), object_pixels[1].max()

    # Apply padding
    height, width = mask_2d.shape
    min_x = max(0, min_x - padding)
    max_x = min(width - 1, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(height - 1, max_y + padding)

    # Validate bounding box
    if min_x >= max_x or min_y >= max_y:
        return None

    return min_x, max_x, min_y, max_y


def filter_mask_by_depth(
    mask_2d: np.ndarray,
    depth_image: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> np.ndarray:
    """
    Filter 2D mask based on depth constraints.

    Args:
        mask_2d: 2D boolean mask
        depth_image: Depth image (same shape as mask)
        min_depth: Minimum valid depth value
        max_depth: Maximum valid depth value

    Returns:
        Filtered boolean mask
    """
    if mask_2d.shape != depth_image.shape:
        print(f"Warning: Mask shape {mask_2d.shape} doesn't match depth shape {depth_image.shape}")
        return mask_2d

    # Create depth validity mask
    valid_depth_mask = (depth_image >= min_depth) & (depth_image <= max_depth)

    # Apply depth filtering to object mask
    filtered_mask = mask_2d & valid_depth_mask

    return filtered_mask


def calculate_mask_statistics(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for a boolean mask.

    Args:
        mask: Boolean mask array

    Returns:
        Dictionary containing mask statistics
    """
    total_points = len(mask)
    object_points = np.sum(mask)
    background_points = total_points - object_points

    stats = {
        'total_points': total_points,
        'object_points': int(object_points),
        'background_points': int(background_points),
        'object_ratio': float(object_points / total_points) if total_points > 0 else 0.0,
        'background_ratio': float(background_points / total_points) if total_points > 0 else 0.0
    }

    return stats


def apply_mask_erosion(
    mask_2d: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological erosion to 2D mask.

    Args:
        mask_2d: 2D boolean mask
        kernel_size: Size of erosion kernel
        iterations: Number of erosion iterations

    Returns:
        Eroded boolean mask
    """
    try:
        import cv2
        
        # Convert to uint8 for OpenCV
        mask_uint8 = mask_2d.astype(np.uint8) * 255
        
        # Create erosion kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply erosion
        eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)
        
        # Convert back to boolean
        return eroded > 0
        
    except ImportError:
        print("Warning: OpenCV not available, returning original mask")
        return mask_2d


def apply_mask_dilation(
    mask_2d: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological dilation to 2D mask.

    Args:
        mask_2d: 2D boolean mask
        kernel_size: Size of dilation kernel
        iterations: Number of dilation iterations

    Returns:
        Dilated boolean mask
    """
    try:
        import cv2
        
        # Convert to uint8 for OpenCV
        mask_uint8 = mask_2d.astype(np.uint8) * 255
        
        # Create dilation kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply dilation
        dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        
        # Convert back to boolean
        return dilated > 0
        
    except ImportError:
        print("Warning: OpenCV not available, returning original mask")
        return mask_2d


def create_circular_mask(
    center: tuple,
    radius: int,
    image_shape: tuple
) -> np.ndarray:
    """
    Create a circular mask centered at given coordinates.

    Args:
        center: (x, y) coordinates of circle center
        radius: Radius of the circle
        image_shape: (height, width) of the output mask

    Returns:
        Boolean mask with circular region set to True
    """
    height, width = image_shape
    center_x, center_y = center

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create circular mask
    mask = distance <= radius

    return mask


def validate_instance_attributes(
    instance_attributes: List[Dict],
    required_keys: List[str] = None
) -> bool:
    """
    Validate instance attribute data structure.

    Args:
        instance_attributes: List of instance attribute dictionaries
        required_keys: List of required keys in each dictionary

    Returns:
        True if all attributes are valid, False otherwise
    """
    if required_keys is None:
        required_keys = ['category_id', 'idx']

    for attr in instance_attributes:
        if not isinstance(attr, dict):
            return False
        
        for key in required_keys:
            if key not in attr:
                return False
            
            # Check for None values
            if attr[key] is None:
                return False

    return True
