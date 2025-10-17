"""
Point Cloud Processing Utilities for SceneLeapPro Dataset

This module provides utility functions for point cloud processing, including
RGB color mapping, cropping, downsampling, and mask operations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def add_rgb_to_pointcloud(
    pointcloud_xyz: np.ndarray,
    rgb_image: np.ndarray,
    camera
) -> np.ndarray:
    """
    Add RGB colors to point cloud by projecting to image plane.

    Args:
        pointcloud_xyz: Point cloud coordinates (N, 3)
        rgb_image: RGB image (H, W, 3)
        camera: Camera intrinsic parameters

    Returns:
        Point cloud with RGB colors (N, 6) - xyz+rgb
    """
    if pointcloud_xyz.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float32)

    points_3d = pointcloud_xyz.reshape(-1, 3)
    points_x, points_y, points_z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Filter points with valid depth
    valid_depth_mask = points_z > 1e-6
    colors = np.zeros((len(points_3d), 3), dtype=np.float32)

    if np.any(valid_depth_mask):
        # Project valid points to image plane
        valid_x = points_x[valid_depth_mask]
        valid_y = points_y[valid_depth_mask]
        valid_z = points_z[valid_depth_mask]

        u = (valid_x * camera.fx / valid_z + camera.cx).astype(int)
        v = (valid_y * camera.fy / valid_z + camera.cy).astype(int)

        # Check pixel bounds
        pixel_valid_mask = (
            (u >= 0) & (u < camera.width) &
            (v >= 0) & (v < camera.height)
        )

        if np.any(pixel_valid_mask):
            valid_u = u[pixel_valid_mask]
            valid_v = v[pixel_valid_mask]

            # Map colors back to point cloud
            valid_indices = np.where(valid_depth_mask)[0][pixel_valid_mask]
            colors[valid_indices] = rgb_image[valid_v, valid_u].astype(np.float32) / 255.0

    # Combine xyz and rgb
    return np.hstack([points_3d, colors])


def map_2d_mask_to_3d_pointcloud(
    point_cloud_xyz: np.ndarray,
    mask_2d: np.ndarray,
    camera
) -> np.ndarray:
    """
    Map 2D instance mask to 3D point cloud mask by projecting points to image plane.

    Args:
        point_cloud_xyz: Point cloud coordinates (N, 3)
        mask_2d: 2D boolean mask (H, W)
        camera: Camera object with intrinsic parameters

    Returns:
        Boolean mask for point cloud (N,)
    """
    if point_cloud_xyz.shape[0] == 0:
        return np.array([], dtype=bool)

    # Filter points with valid depth
    points_z = point_cloud_xyz[:, 2]
    valid_depth_mask = points_z > 1e-6
    object_mask_3d = np.zeros(len(point_cloud_xyz), dtype=bool)

    if not np.any(valid_depth_mask):
        return object_mask_3d

    # Project valid points to image plane
    valid_points = point_cloud_xyz[valid_depth_mask]
    valid_x, valid_y, valid_z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    # Calculate pixel coordinates
    u = (valid_x * camera.fx / valid_z + camera.cx).astype(int)
    v = (valid_y * camera.fy / valid_z + camera.cy).astype(int)

    # Check bounds
    height, width = mask_2d.shape
    pixel_valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    if np.any(pixel_valid_mask):
        valid_u = u[pixel_valid_mask]
        valid_v = v[pixel_valid_mask]

        # Check which projected points fall on the target object
        object_pixels_mask = mask_2d[valid_v, valid_u]

        # Map back to original point cloud indices
        valid_indices = np.where(valid_depth_mask)[0]
        pixel_valid_indices = valid_indices[pixel_valid_mask]
        object_mask_3d[pixel_valid_indices] = object_pixels_mask

    return object_mask_3d


def crop_point_cloud_to_objects(
    point_cloud_with_rgb: np.ndarray,
    instance_mask: np.ndarray,
    camera
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop point cloud to focus on object regions using instance mask.

    Args:
        point_cloud_with_rgb: Point cloud with RGB colors (N, 6)
        instance_mask: Instance segmentation mask (H, W)
        camera: Camera object with intrinsic parameters

    Returns:
        Tuple of (cropped_point_cloud, crop_indices)
        - cropped_point_cloud: Cropped point cloud with RGB colors (M, 6)
        - crop_indices: Boolean array indicating which points were kept (N,)
    """
    try:
        # Create binary mask for all objects (threshold > 9 to exclude background)
        all_objects_mask = instance_mask > 9

        # Check if any objects are detected
        if not np.any(all_objects_mask):
            print("Warning: No objects detected in instance mask, returning original point cloud")
            return point_cloud_with_rgb, np.ones(len(point_cloud_with_rgb), dtype=bool)

        # Find bounding rectangle of all objects
        object_pixels = np.where(all_objects_mask)
        if len(object_pixels[0]) == 0:
            return point_cloud_with_rgb, np.ones(len(point_cloud_with_rgb), dtype=bool)

        # Calculate bounding box with padding
        min_y, max_y = object_pixels[0].min(), object_pixels[0].max()
        min_x, max_x = object_pixels[1].min(), object_pixels[1].max()

        crop_padding = 25
        height, width = instance_mask.shape
        min_x = max(0, min_x - crop_padding)
        max_x = min(width - 1, max_x + crop_padding)
        min_y = max(0, min_y - crop_padding)
        max_y = min(height - 1, max_y + crop_padding)

        # Validate bounding box
        if min_x >= max_x or min_y >= max_y:
            print("Warning: Invalid bounding box after padding, returning original point cloud")
            return point_cloud_with_rgb, np.ones(len(point_cloud_with_rgb), dtype=bool)

        # Project point cloud and apply cropping
        crop_mask = get_crop_mask(point_cloud_with_rgb, camera, min_x, max_x, min_y, max_y)
        cropped_point_cloud = point_cloud_with_rgb[crop_mask]

        # Ensure minimum number of points
        if len(cropped_point_cloud) < 1000:
            print(f"Warning: Too few points after cropping ({len(cropped_point_cloud)}), returning original")
            return point_cloud_with_rgb, np.ones(len(point_cloud_with_rgb), dtype=bool)

        return cropped_point_cloud, crop_mask

    except Exception as e:
        print(f"Error in point cloud cropping: {e}, returning original point cloud")
        return point_cloud_with_rgb, np.ones(len(point_cloud_with_rgb), dtype=bool)


def get_crop_mask(
    point_cloud_with_rgb: np.ndarray,
    camera,
    min_x: int, max_x: int, min_y: int, max_y: int
) -> np.ndarray:
    """
    Get crop mask for point cloud based on bounding box.

    Args:
        point_cloud_with_rgb: Point cloud with RGB colors (N, 6)
        camera: Camera intrinsic parameters
        min_x, max_x, min_y, max_y: Bounding box coordinates

    Returns:
        Boolean mask indicating which points to keep
    """
    points_3d = point_cloud_with_rgb[:, :3]
    points_z = points_3d[:, 2]

    # Filter points with valid depth
    valid_depth_mask = points_z > 1e-6
    if not np.any(valid_depth_mask):
        return np.ones(len(point_cloud_with_rgb), dtype=bool)

    # Project valid points to image plane
    valid_points = points_3d[valid_depth_mask]
    valid_x, valid_y, valid_z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]

    u = (valid_x * camera.fx / valid_z + camera.cx)
    v = (valid_y * camera.fy / valid_z + camera.cy)

    # Filter points within bounding rectangle
    crop_mask_valid = (
        (u >= min_x) & (u <= max_x) &
        (v >= min_y) & (v <= max_y)
    )

    # Map back to original point cloud indices
    crop_mask = np.zeros(len(point_cloud_with_rgb), dtype=bool)
    valid_indices = np.where(valid_depth_mask)[0]
    crop_mask[valid_indices[crop_mask_valid]] = True

    return crop_mask


def remove_outliers_from_point_cloud(
    point_cloud_with_rgb: np.ndarray,
    object_mask: np.ndarray,
    method: str = "object_center_distance",
    distance_threshold_factor: float = 2.5,
    min_object_points: int = 50,
    statistical_nb_neighbors: int = 20,
    statistical_std_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from point cloud based on object mask and spatial distribution.

    Args:
        point_cloud_with_rgb: Point cloud with RGB colors (N, 6)
        object_mask: Object mask (N,) - True for object points
        method: Outlier removal method ("object_center_distance", "statistical", "hybrid")
        distance_threshold_factor: Factor for distance-based threshold (higher = more permissive)
        min_object_points: Minimum object points required to perform outlier removal
        statistical_nb_neighbors: Number of neighbors for statistical outlier removal
        statistical_std_ratio: Standard deviation ratio for statistical outlier removal

    Returns:
        Tuple of (filtered_point_cloud, filtered_mask)
    """
    if point_cloud_with_rgb.shape[0] == 0:
        return point_cloud_with_rgb, object_mask

    # Check if we have enough object points to establish a reliable reference
    object_indices = np.where(object_mask)[0]
    if len(object_indices) < min_object_points:
        # Not enough object points, skip outlier removal
        return point_cloud_with_rgb, object_mask

    xyz_points = point_cloud_with_rgb[:, :3]

    if method == "object_center_distance":
        inlier_mask = _remove_outliers_by_object_center_distance(
            xyz_points, object_mask, distance_threshold_factor
        )
    elif method == "statistical":
        inlier_mask = _remove_outliers_statistical(
            xyz_points, statistical_nb_neighbors, statistical_std_ratio
        )
    elif method == "hybrid":
        # Combine both methods - point must pass both filters
        distance_mask = _remove_outliers_by_object_center_distance(
            xyz_points, object_mask, distance_threshold_factor
        )
        statistical_mask = _remove_outliers_statistical(
            xyz_points, statistical_nb_neighbors, statistical_std_ratio
        )
        inlier_mask = distance_mask & statistical_mask
    else:
        raise ValueError(f"Unknown outlier removal method: {method}")

    # Apply the inlier mask
    filtered_pc = point_cloud_with_rgb[inlier_mask]
    filtered_mask = object_mask[inlier_mask]

    return filtered_pc, filtered_mask


def _remove_outliers_by_object_center_distance(
    xyz_points: np.ndarray,
    object_mask: np.ndarray,
    distance_threshold_factor: float = 2.5
) -> np.ndarray:
    """
    Remove outliers based on distance from object center.

    Args:
        xyz_points: Point coordinates (N, 3)
        object_mask: Object mask (N,)
        distance_threshold_factor: Factor for distance threshold

    Returns:
        Boolean mask indicating inlier points
    """
    object_indices = np.where(object_mask)[0]
    if len(object_indices) == 0:
        return np.ones(len(xyz_points), dtype=bool)

    # Calculate object center
    object_points = xyz_points[object_indices]
    object_center = np.mean(object_points, axis=0)

    # Calculate distances from object center
    distances = np.linalg.norm(xyz_points - object_center, axis=1)

    # Calculate adaptive threshold based on object point distribution
    object_distances = distances[object_indices]
    object_mean_dist = np.mean(object_distances)
    object_std_dist = np.std(object_distances)

    # Use a more robust threshold calculation
    # Base threshold on object distribution + some margin for table surface
    base_threshold = object_mean_dist + 2 * object_std_dist

    # Add extra margin for table surface (typically objects are on table)
    # Assume table extends roughly 0.5-1.0 meters beyond object center
    table_margin = max(0.5, object_mean_dist * 0.5)

    distance_threshold = base_threshold + table_margin
    distance_threshold *= distance_threshold_factor

    return distances <= distance_threshold


def _remove_outliers_statistical(
    xyz_points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> np.ndarray:
    """
    Remove statistical outliers based on local point density.

    Args:
        xyz_points: Point coordinates (N, 3)
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold

    Returns:
        Boolean mask indicating inlier points
    """
    try:
        # Try to use Open3D if available for more efficient computation
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_points)

        # Remove statistical outliers
        _, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        # Create boolean mask
        inlier_mask = np.zeros(len(xyz_points), dtype=bool)
        inlier_mask[inlier_indices] = True

        return inlier_mask

    except ImportError:
        # Fallback to numpy-based implementation
        return _remove_outliers_statistical_numpy(xyz_points, nb_neighbors, std_ratio)


def _remove_outliers_statistical_numpy(
    xyz_points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> np.ndarray:
    """
    Numpy-based statistical outlier removal (fallback when Open3D not available).

    Args:
        xyz_points: Point coordinates (N, 3)
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold

    Returns:
        Boolean mask indicating inlier points
    """
    if len(xyz_points) <= nb_neighbors:
        return np.ones(len(xyz_points), dtype=bool)

    if SCIPY_AVAILABLE:
        # Use scipy KDTree for efficient neighbor search
        tree = cKDTree(xyz_points)
        distances, _ = tree.query(xyz_points, k=nb_neighbors + 1)  # +1 because first is the point itself
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
    else:
        # Fallback to brute force method (slower but works without scipy)
        avg_distances = []
        for i, point in enumerate(xyz_points):
            # Calculate distances to all other points
            dists = np.linalg.norm(xyz_points - point, axis=1)
            # Get k nearest neighbors (excluding self)
            nearest_dists = np.partition(dists, nb_neighbors)[:nb_neighbors + 1]
            nearest_dists = nearest_dists[nearest_dists > 0]  # Exclude self-distance (0)
            if len(nearest_dists) >= nb_neighbors:
                avg_distances.append(np.mean(nearest_dists[:nb_neighbors]))
            else:
                avg_distances.append(np.mean(nearest_dists))
        avg_distances = np.array(avg_distances)

    # Calculate statistics
    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)

    # Points with average distance > mean + std_ratio * std are outliers
    threshold = mean_dist + std_ratio * std_dist
    inlier_mask = avg_distances <= threshold

    return inlier_mask


def downsample_point_cloud_with_mask(
    point_cloud_with_rgb: np.ndarray,
    object_mask: np.ndarray,
    max_points: int = 20000,
    enable_outlier_removal: bool = True,
    outlier_removal_method: str = "object_center_distance",
    distance_threshold_factor: float = 2.5,
    min_object_points_for_outlier_removal: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud and corresponding mask using stratified sampling.
    Preserves object/background ratio for better representation.

    Args:
        point_cloud_with_rgb: Point cloud with RGB colors (N, 6)
        object_mask: Object mask (N,)
        max_points: Maximum number of points to keep
        enable_outlier_removal: Whether to remove outliers before downsampling
        outlier_removal_method: Method for outlier removal ("object_center_distance", "statistical", "hybrid")
        distance_threshold_factor: Factor for distance-based outlier removal (higher = more permissive)
        min_object_points_for_outlier_removal: Minimum object points required to perform outlier removal

    Returns:
        Tuple of (downsampled_point_cloud, downsampled_mask)
    """
    num_points = point_cloud_with_rgb.shape[0]

    # Handle case where we have fewer points than required
    if num_points <= max_points:
        if num_points < max_points:
            padding_size = max_points - num_points
            pc_padding = np.zeros((padding_size, 6), dtype=point_cloud_with_rgb.dtype)
            mask_padding = np.zeros(padding_size, dtype=object_mask.dtype)
            return (
                np.vstack([point_cloud_with_rgb, pc_padding]),
                np.concatenate([object_mask, mask_padding])
            )
        return point_cloud_with_rgb, object_mask

    # Remove outliers before downsampling if enabled
    if enable_outlier_removal:
        point_cloud_with_rgb, object_mask = remove_outliers_from_point_cloud(
            point_cloud_with_rgb,
            object_mask,
            method=outlier_removal_method,
            distance_threshold_factor=distance_threshold_factor,
            min_object_points=min_object_points_for_outlier_removal
        )

        # Update num_points after outlier removal
        num_points = point_cloud_with_rgb.shape[0]

        # Re-check if we still have enough points after outlier removal
        if num_points <= max_points:
            if num_points < max_points:
                padding_size = max_points - num_points
                pc_padding = np.zeros((padding_size, 6), dtype=point_cloud_with_rgb.dtype)
                mask_padding = np.zeros(padding_size, dtype=object_mask.dtype)
                return (
                    np.vstack([point_cloud_with_rgb, pc_padding]),
                    np.concatenate([object_mask, mask_padding])
                )
            return point_cloud_with_rgb, object_mask

    # Stratified sampling to preserve object/background ratio
    object_indices = np.where(object_mask)[0]
    background_indices = np.where(~object_mask)[0]

    # Calculate sampling targets
    object_ratio = len(object_indices) / num_points if num_points > 0 else 0
    target_object_points = int(max_points * object_ratio)
    target_background_points = max_points - target_object_points

    # Sample indices
    selected_indices = sample_stratified_indices(
        object_indices, background_indices,
        target_object_points, target_background_points,
        max_points
    )

    return point_cloud_with_rgb[selected_indices], object_mask[selected_indices]


def sample_stratified_indices(
    object_indices: np.ndarray,
    background_indices: np.ndarray,
    target_object_points: int,
    target_background_points: int,
    max_points: int
) -> np.ndarray:
    """
    Sample indices using stratified sampling strategy.

    Args:
        object_indices: Indices of object points
        background_indices: Indices of background points
        target_object_points: Target number of object points
        target_background_points: Target number of background points
        max_points: Maximum total points

    Returns:
        Array of selected indices
    """
    selected_indices = []

    # Sample object points
    if len(object_indices) > 0 and target_object_points > 0:
        if len(object_indices) <= target_object_points:
            selected_indices.extend(object_indices)
        else:
            selected_object = np.random.choice(
                object_indices, target_object_points, replace=False
            )
            selected_indices.extend(selected_object)

    # Sample background points
    if len(background_indices) > 0 and target_background_points > 0:
        if len(background_indices) <= target_background_points:
            selected_indices.extend(background_indices)
        else:
            selected_background = np.random.choice(
                background_indices, target_background_points, replace=False
            )
            selected_indices.extend(selected_background)

    # Fill remaining slots if needed
    selected_indices = np.array(selected_indices, dtype=int)
    if len(selected_indices) < max_points:
        all_indices = np.concatenate([object_indices, background_indices])
        unused_indices = np.setdiff1d(all_indices, selected_indices)
        if len(unused_indices) > 0:
            remaining_points = max_points - len(selected_indices)
            additional = np.random.choice(
                unused_indices,
                min(remaining_points, len(unused_indices)),
                replace=False
            )
            selected_indices = np.concatenate([selected_indices, additional])

    # Shuffle and limit to max_points
    np.random.shuffle(selected_indices)
    return selected_indices[:max_points]


def crop_point_cloud_to_objects_with_mask(
    point_cloud_with_rgb: np.ndarray,
    object_mask: np.ndarray,
    instance_mask: np.ndarray,
    camera
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop point cloud and corresponding object mask to focus on object regions.

    Args:
        point_cloud_with_rgb: Point cloud with RGB colors (N, 6)
        object_mask: Object mask for point cloud (N,)
        instance_mask: Instance segmentation mask (H, W)
        camera: Camera object with intrinsic parameters

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Cropped point cloud, Cropped object mask)
    """
    # Crop point cloud using instance mask
    cropped_pc, crop_indices = crop_point_cloud_to_objects(point_cloud_with_rgb, instance_mask, camera)

    # Apply the same crop indices to the object mask
    if len(object_mask) == len(point_cloud_with_rgb):
        cropped_object_mask = object_mask[crop_indices]
        return cropped_pc, cropped_object_mask
    else:
        # If original mask size didn't match, create new default mask
        print(f"Warning: Object mask size ({len(object_mask)}) doesn't match point cloud size ({len(point_cloud_with_rgb)})")
        return cropped_pc, np.zeros(len(cropped_pc), dtype=bool)
