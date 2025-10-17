import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import (euler_angles_to_matrix, matrix_to_euler_angles,
                                quaternion_to_matrix, matrix_to_quaternion,
                                quaternion_invert, quaternion_multiply)
# 尝试导入 open3d，如果失败则SOR功能不可用
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D library not found. Statistical Outlier Removal will not be available.")

# Supporting functions
def downsample_point_cloud(point_cloud, num_points=100000):
    if point_cloud.shape[0] > num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[indices]
    return point_cloud

def downsample_point_cloud_other(point_cloud, num_points=100000, object_weight=0.6):
    object_mask = point_cloud[:, 3] > 0.5
    object_points = point_cloud[object_mask]
    background_points = point_cloud[~object_mask]
    
    if object_points.shape[0] == 0:
        # 如果没有目标物体点，直接下采样背景
        return downsample_point_cloud(background_points, num_points)
    
    object_center = np.mean(object_points[:, :3], axis=0)
    distances = np.linalg.norm(background_points[:, :3] - object_center, axis=1)
    distance_weights = 1 / (1 + distances)
    
    num_object_points = min(int(num_points * object_weight), object_points.shape[0])
    num_background_points = num_points - num_object_points
    
    if object_points.shape[0] > num_object_points:
        object_indices = np.random.choice(object_points.shape[0], num_object_points, replace=False)
        sampled_object_points = object_points[object_indices]
    else:
        sampled_object_points = object_points
    
    if background_points.shape[0] > num_background_points:
        background_indices = np.random.choice(
            background_points.shape[0], 
            num_background_points, 
            replace=False, 
            p=distance_weights / np.sum(distance_weights)
        )
        sampled_background_points = background_points[background_indices]
    else:
        sampled_background_points = background_points
    
    sampled_point_cloud = np.vstack((sampled_object_points, sampled_background_points))
    
    if sampled_point_cloud.shape[0] < num_points:
        additional_indices = np.random.choice(sampled_point_cloud.shape[0], num_points - sampled_point_cloud.shape[0], replace=True)
        additional_points = sampled_point_cloud[additional_indices]
        sampled_point_cloud = np.vstack((sampled_point_cloud, additional_points))
    
    return sampled_point_cloud

def get_points_in_aabb(points, min_coords, max_coords):
    """
    Helper function to get points within an Axis-Aligned Bounding Box (AABB).
    points: Nxk array, where first 3 columns are x, y, z.
    min_coords: (3,) array for min x, y, z.
    max_coords: (3,) array for max x, y, z.
    Returns: Boolean mask of shape (N,)
    """
    return np.all((points[:, :3] >= min_coords) & (points[:, :3] <= max_coords), axis=1)

def remove_object_outliers_sor(object_points_xyz, nb_neighbors=20, std_ratio=2.0):
    """
    Removes outliers from object points using Statistical Outlier Removal.
    Requires Open3D.
    Args:
        object_points_xyz (np.ndarray): Nx3 array of object points.
        nb_neighbors (int): Number of neighbors to consider.
        std_ratio (float): Standard deviation ratio.
    Returns:
        np.ndarray: Cleaned Nx3 object points.
    """
    if not OPEN3D_AVAILABLE:
        print("Warning: Open3D not available, skipping SOR.")
        return object_points_xyz
    if object_points_xyz.shape[0] < nb_neighbors: # Not enough points for SOR
        print(f"Warning: Not enough object points ({object_points_xyz.shape[0]}) for SOR with {nb_neighbors} neighbors. Skipping.")
        return object_points_xyz

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_points_xyz)
    # cl是 cleaned_pcd, ind 是 inlier_indices
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    # print(f"SOR: Original object points: {object_points_xyz.shape[0]}, after SOR: {len(ind)}")
    return object_points_xyz[ind]


def downsample_point_cloud_focused(point_cloud, num_points_to_sample=10000,
                                   aabb_padding=0.2, object_priority_ratio=0.3,
                                   clean_object_points=True, 
                                   sor_nb_neighbors=20, sor_std_ratio=2.0):
    """
    Downsamples a point cloud by focusing on the region around a target object,
    with an option to clean object points before AABB calculation.

    Args:
        point_cloud (np.ndarray): The input point cloud, shape (N, 4).
                                  Assumes last column is the object mask (1 for object, 0 for background).
        num_points_to_sample (int): The total number of points to sample.
        aabb_padding (float): How much to expand the object's AABB in meters to define the "near space".
        object_priority_ratio (float): Desired ratio of object points in the sampled output (0.0 to 1.0).
        clean_object_points (bool): If True, attempts to remove outliers from object points before AABB calculation.
        sor_nb_neighbors (int): Parameter for Statistical Outlier Removal (if used).
        sor_std_ratio (float): Parameter for Statistical Outlier Removal (if used).

    Returns:
        np.ndarray: The downsampled point cloud, shape (K, 4), where K <= num_points_to_sample.
    """
    if point_cloud.shape[0] == 0:
        return np.empty((0, 4), dtype=point_cloud.dtype)

    # 1. Separate object and background points
    object_mask_bool = point_cloud[:, 3] > 0.5
    all_object_points_with_mask = point_cloud[object_mask_bool]
    all_background_points_with_mask = point_cloud[~object_mask_bool]

    if all_object_points_with_mask.shape[0] == 0:
        print("Warning: No object points found in the input point cloud.")
        # Fallback: randomly sample from the whole scene if no object
        if point_cloud.shape[0] > num_points_to_sample:
            indices = np.random.choice(point_cloud.shape[0], num_points_to_sample, replace=False)
            return point_cloud[indices]
        return point_cloud

    # --- New Step: Clean Object Points (Optional) ---
    object_points_for_aabb_calc = all_object_points_with_mask # Default to use all marked object points
    if clean_object_points and OPEN3D_AVAILABLE:
        if all_object_points_with_mask.shape[0] > 0: # Check if there are any object points to clean
            # Extract XYZ for SOR, keep original full points for later sampling
            object_xyz_for_sor = all_object_points_with_mask[:, :3]
            cleaned_object_xyz = remove_object_outliers_sor(object_xyz_for_sor,
                                                            nb_neighbors=sor_nb_neighbors,
                                                            std_ratio=sor_std_ratio)
            
            if cleaned_object_xyz.shape[0] < 10: # Heuristic: if too few points remain, SOR might have been too aggressive
                print(f"Warning: SOR resulted in very few object points ({cleaned_object_xyz.shape[0]}). Reverting to original object points for AABB calculation.")
                # object_points_for_aabb_calc remains all_object_points_with_mask
            elif cleaned_object_xyz.shape[0] > 0 : # Ensure some points are left
                 # We need to create a temporary N_cleaned x 4 array for AABB calculation
                 # This is a bit tricky because we only have cleaned XYZ, not their original masks (though they are all object)
                 # For AABB calculation, only XYZ matters.
                 # For actual sampling later, we will use the original all_object_points_with_mask
                 # but filter them using the cleaned AABB.
                 # Alternative: Re-filter all_object_points_with_mask to keep only points whose XYZ are in cleaned_object_xyz.
                 # This is more robust.
                
                # Find indices of cleaned points in the original object_xyz_for_sor
                # This can be slow for large clouds; a KDTree approach would be faster for matching.
                # For simplicity here, if SOR was effective and points are from original, we can assume
                # the AABB from cleaned_object_xyz is good enough.
                # The actual object points for sampling later (object_points_in_roi)
                # will still come from all_object_points_with_mask filtered by this new AABB.
                
                # Create a temporary array for AABB calculation based on cleaned points
                # This is only used for min/max, so mask value doesn't matter here.
                temp_cleaned_for_aabb = np.zeros((cleaned_object_xyz.shape[0], 4), dtype=point_cloud.dtype)
                temp_cleaned_for_aabb[:, :3] = cleaned_object_xyz
                object_points_for_aabb_calc = temp_cleaned_for_aabb
            else: # No points left after cleaning
                print("Warning: SOR removed all object points. Reverting to original object points for AABB calculation.")
                # object_points_for_aabb_calc remains all_object_points_with_mask

        else: # No object points to clean
            pass # object_points_for_aabb_calc remains all_object_points_with_mask

    # 2. Calculate object AABB using (potentially cleaned) object points
    # Ensure there are points to calculate AABB from
    if object_points_for_aabb_calc.shape[0] == 0:
        print("Error: No object points available (even before/after cleaning) to calculate AABB. Using uncleaned points for AABB.")
        # Fallback to original object points if cleaning removed everything or there were none for cleaning.
        if all_object_points_with_mask.shape[0] == 0:
             # This case should have been caught earlier, but as a safeguard:
            print("Critical Error: No object points at all. Cannot proceed with focused sampling.")
            # Fallback to random sampling of the whole scene as a last resort
            if point_cloud.shape[0] > num_points_to_sample:
                indices = np.random.choice(point_cloud.shape[0], num_points_to_sample, replace=False)
                return point_cloud[indices]
            return point_cloud
        obj_min_coords = np.min(all_object_points_with_mask[:, :3], axis=0)
        obj_max_coords = np.max(all_object_points_with_mask[:, :3], axis=0)
    else:
        obj_min_coords = np.min(object_points_for_aabb_calc[:, :3], axis=0)
        obj_max_coords = np.max(object_points_for_aabb_calc[:, :3], axis=0)


    # 3. Expand AABB to define "near space" ROI
    roi_min_coords = obj_min_coords - aabb_padding
    roi_max_coords = obj_max_coords + aabb_padding

    # 4. Filter *original* all_object_points and all_background_points within the ROI
    # We use the original points here to ensure we get the mask values correctly
    # and to avoid issues if SOR slightly shifted points.
    object_points_in_roi_mask = get_points_in_aabb(all_object_points_with_mask, roi_min_coords, roi_max_coords)
    object_points_in_roi = all_object_points_with_mask[object_points_in_roi_mask]

    background_points_in_roi_mask = get_points_in_aabb(all_background_points_with_mask, roi_min_coords, roi_max_coords)
    background_points_in_roi = all_background_points_with_mask[background_points_in_roi_mask]
    
    # ... (rest of the sampling logic from step 5 onwards remains the same) ...

    # 5. Determine target number of points for object and background
    num_desired_object_pts = int(num_points_to_sample * object_priority_ratio)
    # num_desired_background_pts = num_points_to_sample - num_desired_object_pts # Calculated later

    # 6. Sample object points from ROI
    if object_points_in_roi.shape[0] > num_desired_object_pts:
        obj_indices = np.random.choice(object_points_in_roi.shape[0], num_desired_object_pts, replace=False)
        sampled_object_points = object_points_in_roi[obj_indices]
    else:
        sampled_object_points = object_points_in_roi 

    # 7. Adjust desired background points and sample background points from ROI
    actual_num_object_pts = sampled_object_points.shape[0]
    num_background_pts_to_sample_now = num_points_to_sample - actual_num_object_pts
    num_background_pts_to_sample_now = max(0, num_background_pts_to_sample_now) # Ensure non-negative

    if background_points_in_roi.shape[0] > num_background_pts_to_sample_now:
        if num_background_pts_to_sample_now > 0:
            bg_indices = np.random.choice(background_points_in_roi.shape[0], num_background_pts_to_sample_now, replace=False)
            sampled_background_points = background_points_in_roi[bg_indices]
        else:
            sampled_background_points = np.empty((0,4), dtype=point_cloud.dtype)
    else:
        sampled_background_points = background_points_in_roi

    # 8. Combine sampled points
    final_sampled_points_list = []
    if sampled_object_points.shape[0] > 0:
        final_sampled_points_list.append(sampled_object_points)
    if sampled_background_points.shape[0] > 0:
        final_sampled_points_list.append(sampled_background_points)

    if not final_sampled_points_list:
        print("Warning: ROI resulted in no points to sample. Consider adjusting AABB padding or checking object mask.")
        # Fallback strategy if ROI is empty or yields too few points
        # Option 1: Return empty
        # return np.empty((0, 4), dtype=point_cloud.dtype)
        # Option 2: Fallback to sampling object points from original full object cloud (if any)
        if all_object_points_with_mask.shape[0] > 0:
            print("Fallback: Sampling from original object points as ROI was empty/insufficient.")
            if all_object_points_with_mask.shape[0] > num_points_to_sample:
                 obj_indices = np.random.choice(all_object_points_with_mask.shape[0], num_points_to_sample, replace=False)
                 return all_object_points_with_mask[obj_indices]
            return all_object_points_with_mask
        return np.empty((0, 4), dtype=point_cloud.dtype)

    final_sampled_points = np.vstack(final_sampled_points_list)
    
    # 9. (Optional) Handle shortfall if strictly num_points_to_sample is required
    # if final_sampled_points.shape[0] < num_points_to_sample and final_sampled_points.shape[0] > 0:
    #     num_to_add = num_points_to_sample - final_sampled_points.shape[0]
    #     additional_indices = np.random.choice(final_sampled_points.shape[0], num_to_add, replace=True)
    #     final_sampled_points = np.vstack((final_sampled_points, final_sampled_points[additional_indices]))

    return final_sampled_points

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth.astype(np.float32) / camera.depth_scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud

def transform_point_cloud(points_in_camera_frame, cam_R_model_to_camera, cam_t_model_to_camera):
    # This function transforms points from the camera frame to the object's model frame.
    # cam_R_model_to_camera (R_mc) and cam_t_model_to_camera (t_mc) define T_model_to_camera.
    # We need T_camera_to_model = inv(T_model_to_camera).
    # R_cam_to_model = cam_R_model_to_camera.T (R_mc.T)
    # t_cam_to_model = - (cam_R_model_to_camera.T @ cam_t_model_to_camera) (-R_mc.T @ t_mc)
    # So, P_model = R_cam_to_model @ P_camera + t_cam_to_model

    R_mc = cam_R_model_to_camera
    t_mc = cam_t_model_to_camera # Expected as (3,) or (3,1)

    R_cm = R_mc.T
    t_cm = -np.dot(R_mc.T, t_mc.reshape(3,1)).flatten() # Ensure t_mc is col vector for matmul, then flatten t_cm

    # points_in_camera_frame is (N,3). R_cm is (3,3). t_cm is (3,)
    # P_model = (P_camera @ R_cm.T) + t_cm  <- if P_camera are row vectors
    # P_model = R_cm @ P_camera_cols + t_cm_col <- if P_camera are col vectors
    # Using (P_camera @ R_cm.T) + t_cm for row vector convention
    points_in_model_frame = np.dot(points_in_camera_frame, R_cm.T) + t_cm.reshape(1,3)
    return points_in_model_frame

def transform_hand_pose(hand_pose, cam_R_m2c, cam_t_m2c):
    qpos, euler_angles, translation = hand_pose[:22], hand_pose[22:25], hand_pose[25:28]
    
    grasp_rot_matrix = R.from_euler('xyz', euler_angles.numpy()).as_matrix()
    grasp_rot_camera = np.dot(cam_R_m2c, grasp_rot_matrix)
    grasp_trans_camera = np.dot(cam_R_m2c, translation.numpy()) + cam_t_m2c
    grasp_euler_camera = R.from_matrix(grasp_rot_camera).as_euler('xyz')
    
    euler_camera = torch.tensor(grasp_euler_camera, dtype=torch.float32)
    trans_camera = torch.tensor(grasp_trans_camera, dtype=torch.float32)
    return torch.cat((trans_camera, euler_camera, qpos), dim=0)

def transform_hand_pose_se3(hand_pose, cam_R_m2c, cam_t_m2c):
    qpos, euler_angles, translation = hand_pose[:22], hand_pose[22:25], hand_pose[25:28]
    
    grasp_rot_matrix = R.from_euler('xyz', euler_angles.numpy()).as_matrix()
    grasp_rot_camera = np.dot(cam_R_m2c, grasp_rot_matrix)
    grasp_trans_camera = np.dot(cam_R_m2c, translation.numpy()) + cam_t_m2c
    grasp_euler_camera = R.from_matrix(grasp_rot_camera).as_euler('xyz')
    
    euler_camera = torch.tensor(grasp_euler_camera, dtype=torch.float32)
    trans_camera = torch.tensor(grasp_trans_camera, dtype=torch.float32)
    se3 = torch.eye(4, dtype=torch.float32)
    se3[:3, :3] = torch.tensor(grasp_rot_camera, dtype=torch.float32)
    se3[:3, 3] = trans_camera
    return torch.cat((trans_camera, euler_camera, qpos), dim=0), se3

def transform_hand_pose_batch(hand_poses, cam_R_m2c, cam_t_m2c):
    if isinstance(cam_R_m2c, np.ndarray):
        cam_R_m2c = torch.tensor(cam_R_m2c, dtype=torch.float32)
    if isinstance(cam_t_m2c, np.ndarray):
        cam_t_m2c = torch.tensor(cam_t_m2c, dtype=torch.float32)

    cam_R_m2c = cam_R_m2c.to(hand_poses.device)
    cam_t_m2c = cam_t_m2c.to(hand_poses.device)
    
    qpos, euler_angles, translation = hand_poses[:, :22], hand_poses[:, 22:25], hand_poses[:, 25:28]
    grasp_rot_matrix = euler_angles_to_matrix(-euler_angles, convention='XYZ')
    grasp_rot_matrix = torch.transpose(grasp_rot_matrix, 1, 2)
    grasp_rot_camera = torch.matmul(cam_R_m2c, grasp_rot_matrix)
    grasp_trans_camera = torch.matmul(cam_R_m2c, translation.unsqueeze(-1)).squeeze(-1) + cam_t_m2c
    grasp_euler_camera = matrix_to_euler_angles(grasp_rot_camera, convention='XYZ')
    return torch.cat((grasp_trans_camera.float(), grasp_euler_camera.float(), qpos), dim=1)

class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, depth_scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale * 1000.0 