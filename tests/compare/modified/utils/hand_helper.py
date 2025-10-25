import json
import math

import numpy as np
import pytorch3d.transforms as transforms
import torch
import torch.nn.functional as F
from pytorchse3.se3 import se3_log_map

JSON_STATS_FILE_PATH = "assets/formatch_overall_hand_pose_dimension_statistics_by_mode.json"
POSE_STATS = None
JOINT_ANGLE_DIM = 16

def load_pose_stats_if_needed():
    """Loads pose statistics from the JSON file if not already loaded."""
    global POSE_STATS
    if POSE_STATS is None:
        try:
            with open(JSON_STATS_FILE_PATH, 'r') as f:
                raw_stats = json.load(f)
            POSE_STATS = {}
            for mode_key, stats_list in raw_stats.items():
                POSE_STATS[mode_key] = {item['dimension_label']: item for item in stats_list}
        except FileNotFoundError:
            print(f"CRITICAL WARNING: Statistics file {JSON_STATS_FILE_PATH} not found. Normalization will likely fail or be incorrect.")
            POSE_STATS = {}
        except json.JSONDecodeError:
            print(f"CRITICAL WARNING: Error decoding JSON from {JSON_STATS_FILE_PATH}. Normalization will be incorrect.")
            POSE_STATS = {}
        except Exception as e:
            print(f"CRITICAL WARNING: An unexpected error occurred while loading statistics: {e}")
            POSE_STATS = {}

load_pose_stats_if_needed()

def get_min_max_from_stats(mode, dimension_labels, device, dtype, tensor_type=torch.Tensor):
    """Helper function to retrieve min and max values for given dimension labels from loaded stats."""
    if POSE_STATS is None or not POSE_STATS:
        raise RuntimeError("Pose statistics are not loaded. Cannot proceed with normalization.")
    if mode not in POSE_STATS:
        raise ValueError(f"Mode '{mode}' not found in loaded statistics. Available modes: {list(POSE_STATS.keys())}")

    mins = []
    maxs = []
    for label in dimension_labels:
        if label not in POSE_STATS[mode]:
            raise ValueError(f"Dimension label '{label}' not found in mode '{mode}'. Check statistics file.")
        mins.append(POSE_STATS[mode][label]['min'])
        maxs.append(POSE_STATS[mode][label]['max'])
    
    if tensor_type == torch.Tensor:
        return torch.tensor(mins, device=device, dtype=dtype), torch.tensor(maxs, device=device, dtype=dtype)
    elif tensor_type == np.ndarray:
        return np.array(mins, dtype=dtype), np.array(maxs, dtype=dtype)
    else:
        raise TypeError(f"Unsupported tensor_type: {tensor_type}")

# Rotation normalization functions
def normalize_rot6d_torch(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot

def normalize_np(x):
    x_n = np.linalg.norm(x, axis=-1, keepdims=True)
    x_n = x_n.clip(min=1e-8)
    x = x / x_n
    return x

def normalize_rot6d_numpy(rot):
    if rot.shape[-1] == 3:
        unflatten = True
        undim = True
        ori_shape = rot.shape[:-2]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    elif len(rot.shape) > 2:
        unflatten = False
        undim = True
        ori_shape = rot.shape[:-1]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    else:
        unflatten = False
        undim = False
        ori_shape = None
    a1, a2 = rot[:, :3], rot[:, 3:]
    b1 = normalize_np(a1)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = normalize_np(b2)
    rot = np.concatenate([b1, b2], axis=-1)
    if unflatten:
        rot = rot.reshape(ori_shape + (2, 3))
    elif undim:
        rot = rot.reshape(ori_shape + (6, ))
    return rot

def normalize_rot6d(rot):
    if isinstance(rot, torch.Tensor):
        return normalize_rot6d_torch(rot)
    elif isinstance(rot, np.ndarray):
        return normalize_rot6d_numpy(rot)
    else:
        raise NotImplementedError

# Constants
NORM_UPPER = 1.0
NORM_LOWER = -1.0

ROT_DIM_DICT = {
    'quat': 4,
    'axis': 3,
    'euler': 3,
    'r6d': 6,
    'map': 3,
}

# Translation normalization functions
def normalize_trans_torch(hand_t, mode, rot_type_for_map_distinction='not_map'):
    if rot_type_for_map_distinction == 'map':
        labels = [f'se3_log_map_dim_{i}' for i in range(3)]
    else:
        labels = ['translation_x', 'translation_y', 'translation_z']
    t_min, t_max = get_min_max_from_stats(mode, labels, hand_t.device, hand_t.dtype)
    
    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t_scaled_0_1 = torch.div((hand_t - t_min), t_range)
    t_final_normalized = t_scaled_0_1 * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    
    t = t_final_normalized

    return t

def denormalize_trans_torch(hand_t, mode, rot_type_for_map_distinction='not_map'):
    if rot_type_for_map_distinction == 'map':
        labels = [f'se3_log_map_dim_{i}' for i in range(3)]
    else:
        labels = ['translation_x', 'translation_y', 'translation_z']
    t_min, t_max = get_min_max_from_stats(mode, labels, hand_t.device, hand_t.dtype)

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t = hand_t + ((NORM_UPPER - NORM_LOWER) / 2.0)
    t /= (NORM_UPPER - NORM_LOWER)
    t = t * t_range + t_min

    return t

def normalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction='not_map'):
    if rot_type_for_map_distinction == 'map':
        labels = [f'se3_log_map_dim_{i}' for i in range(3)]
    else:
        labels = ['translation_x', 'translation_y', 'translation_z']
    t_min, t_max = get_min_max_from_stats(mode, labels, None, hand_t.dtype, tensor_type=np.ndarray)
    
    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8

    t = (hand_t - t_min) / t_range
    t = t * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    return t

def denormalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction='not_map'):
    if rot_type_for_map_distinction == 'map':
        labels = [f'se3_log_map_dim_{i}' for i in range(3)]
    else:
        labels = ['translation_x', 'translation_y', 'translation_z']
    t_min, t_max = get_min_max_from_stats(mode, labels, None, hand_t.dtype, tensor_type=np.ndarray)

    t_range = t_max - t_min
    t_range[t_range == 0] = 1e-8
    
    t = hand_t + ((NORM_UPPER - NORM_LOWER) / 2.0)
    t /= (NORM_UPPER - NORM_LOWER)
    t = t * t_range + t_min
    return t

# Joint parameter normalization functions
def normalize_param_torch(hand_param, mode):
    labels = [f'joint_angle_{i}' for i in range(JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(mode, labels, hand_param.device, hand_param.dtype)
    
    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = torch.div((hand_param - p_min), p_range)
    p = p * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    return p

def denormalize_param_torch(hand_param, mode):
    labels = [f'joint_angle_{i}' for i in range(JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(mode, labels, hand_param.device, hand_param.dtype)

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = hand_param + ((NORM_UPPER - NORM_LOWER) / 2.0)
    p /= (NORM_UPPER - NORM_LOWER)
    p = p * p_range + p_min

    return p

def normalize_param_numpy(hand_param, mode):
    labels = [f'joint_angle_{i}' for i in range(JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(mode, labels, None, hand_param.dtype, tensor_type=np.ndarray)

    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = (hand_param - p_min) / p_range
    p = p * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    return p

def denormalize_param_numpy(hand_param, mode):
    labels = [f'joint_angle_{i}' for i in range(JOINT_ANGLE_DIM)]
    p_min, p_max = get_min_max_from_stats(mode, labels, None, hand_param.dtype, tensor_type=np.ndarray)
    
    p_range = p_max - p_min
    p_range[p_range == 0] = 1e-8

    p = hand_param + ((NORM_UPPER - NORM_LOWER) / 2.0)
    p /= (NORM_UPPER - NORM_LOWER)
    p = p * p_range + p_min
    return p

# Rotation normalization functions
def _get_rot_labels(rot_type):
    if rot_type == 'quat':
        return ['quaternion_w', 'quaternion_x', 'quaternion_y', 'quaternion_z']
    elif rot_type == 'r6d':
        return [f'r6d_dim_{i}' for i in range(6)]
    elif rot_type == 'map':
        return [f'se3_log_map_dim_{i}' for i in range(3, 6)]
    elif rot_type == 'axis':
        print(f"Warning: Using generic 'axis_angle_dim_0/1/2' for rot_type 'axis'. Ensure these labels are in your JSON or update mapping.")
        return [f'axis_angle_dim_{i}' for i in range(3)]
    elif rot_type == 'euler':
        print(f"Warning: Using generic 'euler_angle_dim_0/1/2' for rot_type 'euler'. Ensure these labels are in your JSON or update mapping.")
        return [f'euler_angle_dim_{i}' for i in range(3)]
    else:
        raise ValueError(f"Unsupported rotation type for generating dimension labels: {rot_type}")

def normalize_rot_torch(hand_r, rot_type, mode):
    if rot_type == 'quat':
        return hand_r

    labels = _get_rot_labels(rot_type)
    r_min, r_max = get_min_max_from_stats(mode, labels, hand_r.device, hand_r.dtype)
    
    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8
    
    r = torch.div((hand_r - r_min), r_range)
    r = r * (NORM_UPPER - NORM_LOWER) + NORM_LOWER

    return r

def denormalize_rot_torch(hand_r, rot_type, mode):
    if rot_type == 'quat':
        return hand_r

    labels = _get_rot_labels(rot_type)
    r_min, r_max = get_min_max_from_stats(mode, labels, hand_r.device, hand_r.dtype)

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = hand_r + ((NORM_UPPER - NORM_LOWER) / 2.0)
    r /= (NORM_UPPER - NORM_LOWER)
    r = r * r_range + r_min

    return r

def normalize_rot_numpy(hand_r, rot_type, mode):
    if rot_type == 'quat':
        return hand_r

    labels = _get_rot_labels(rot_type)
    r_min, r_max = get_min_max_from_stats(mode, labels, None, hand_r.dtype, tensor_type=np.ndarray)

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = (hand_r - r_min) / r_range
    r = r * (NORM_UPPER - NORM_LOWER) + NORM_LOWER
    return r

def denormalize_rot_numpy(hand_r, rot_type, mode):
    if rot_type == 'quat':
        return hand_r
        
    labels = _get_rot_labels(rot_type)
    r_min, r_max = get_min_max_from_stats(mode, labels, None, hand_r.dtype, tensor_type=np.ndarray)

    r_range = r_max - r_min
    r_range[r_range == 0] = 1e-8

    r = hand_r + ((NORM_UPPER - NORM_LOWER) / 2.0)
    r /= (NORM_UPPER - NORM_LOWER)
    r = r * r_range + r_min
    return r

# Main pose normalization functions
def norm_hand_pose(hand_pose, rot_type, mode):
    """Normalize hand pose parameters"""
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    if isinstance(hand_pose, torch.Tensor):
        hand_t, hand_param, hand_r = hand_pose.split((3, JOINT_ANGLE_DIM, rot_dim), dim=-1)
        
        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t = normalize_trans_torch(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r = normalize_rot_torch(hand_r, rot_type, mode)
        hand_param = normalize_param_torch(hand_param, mode)
        hand = torch.cat([hand_t, hand_param, hand_r], dim=-1)
    elif isinstance(hand_pose, np.ndarray):
        hand_t, hand_param, hand_r = np.split(hand_pose, [3, 3 + JOINT_ANGLE_DIM], axis=-1)
        
        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t = normalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r = normalize_rot_numpy(hand_r, rot_type, mode)
        hand_param = normalize_param_numpy(hand_param, mode)
        hand = np.concatenate([hand_t, hand_param, hand_r], axis=-1)
    else:
        raise NotImplementedError

    return hand

def denorm_hand_pose(hand_pose, rot_type, mode):
    """Denormalize hand pose parameters"""
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    if isinstance(hand_pose, torch.Tensor):
        hand_t, hand_param, hand_r = hand_pose.split((3, JOINT_ANGLE_DIM, rot_dim), dim=-1)

        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t = denormalize_trans_torch(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r = denormalize_rot_torch(hand_r, rot_type, mode)
        hand_param = denormalize_param_torch(hand_param, mode)
        hand = torch.cat([hand_t, hand_param, hand_r], dim=-1)
    elif isinstance(hand_pose, np.ndarray):
        hand_t, hand_param, hand_r = np.split(hand_pose, [3, 3 + JOINT_ANGLE_DIM], axis=-1)

        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t = denormalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r = denormalize_rot_numpy(hand_r, rot_type, mode)
        hand_param = denormalize_param_numpy(hand_param, mode)
        hand = np.concatenate([hand_t, hand_param, hand_r], axis=-1)
    else:
        raise NotImplementedError
    
    return hand

def norm_hand_pose_robust(hand_pose, rot_type, mode):
    """
    Normalize hand pose parameters. Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim]
    - Multi grasp: [B, num_grasps, pose_dim]
    """
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    
    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose
            
        hand_t, hand_param, hand_r = hand_pose_reshaped.split((3, JOINT_ANGLE_DIM, rot_dim), dim=-1)
        
        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t_norm = normalize_trans_torch(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r_norm = normalize_rot_torch(hand_r, rot_type, mode)
        hand_param_norm = normalize_param_torch(hand_param, mode)
        hand_norm_reshaped = torch.cat([hand_t_norm, hand_param_norm, hand_r_norm], dim=-1)
        
        if len(orig_shape) == 3:
            hand = hand_norm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_norm_reshaped
            
    elif isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose
            
        hand_t, hand_param, hand_r = np.split(hand_pose_reshaped, [3, 3 + JOINT_ANGLE_DIM], axis=-1)

        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t_norm = normalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r_norm = normalize_rot_numpy(hand_r, rot_type, mode)
        hand_param_norm = normalize_param_numpy(hand_param, mode)
        hand_norm_reshaped = np.concatenate([hand_t_norm, hand_param_norm, hand_r_norm], axis=-1)
        
        if len(orig_shape) == 3:
            hand = hand_norm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_norm_reshaped
    else:
        raise NotImplementedError

    return hand

def denorm_hand_pose_robust(hand_pose, rot_type, mode):
    """
    Denormalize hand pose parameters. Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim]
    - Multi grasp: [B, num_grasps, pose_dim]
    """
    rot_dim = ROT_DIM_DICT[rot_type.lower()]
    
    if isinstance(hand_pose, torch.Tensor):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose
            
        hand_t, hand_param, hand_r = hand_pose_reshaped.split((3, JOINT_ANGLE_DIM, rot_dim), dim=-1)

        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t_denorm = denormalize_trans_torch(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r_denorm = denormalize_rot_torch(hand_r, rot_type, mode)
        hand_param_denorm = denormalize_param_torch(hand_param, mode)
        hand_denorm_reshaped = torch.cat([hand_t_denorm, hand_param_denorm, hand_r_denorm], dim=-1)
        
        if len(orig_shape) == 3:
            hand = hand_denorm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_denorm_reshaped
            
    elif isinstance(hand_pose, np.ndarray):
        orig_shape = hand_pose.shape
        if len(orig_shape) == 3:
            hand_pose_reshaped = hand_pose.reshape(-1, orig_shape[-1])
        else:
            hand_pose_reshaped = hand_pose

        hand_t, hand_param, hand_r = np.split(hand_pose_reshaped, [3, 3 + JOINT_ANGLE_DIM], axis=-1)

        map_distinction = 'map' if rot_type == 'map' else 'not_map'
        hand_t_denorm = denormalize_trans_numpy(hand_t, mode, rot_type_for_map_distinction=map_distinction)
        hand_r_denorm = denormalize_rot_numpy(hand_r, rot_type, mode)
        hand_param_denorm = denormalize_param_numpy(hand_param, mode)
        hand_denorm_reshaped = np.concatenate([hand_t_denorm, hand_param_denorm, hand_r_denorm], axis=-1)
        
        if len(orig_shape) == 3:
            hand = hand_denorm_reshaped.reshape(orig_shape[0], orig_shape[1], -1)
        else:
            hand = hand_denorm_reshaped
    else:
        raise NotImplementedError
    
    return hand

# Batch processing functions
def _process_batch_pose_logic(se3_batch, hand_model_pose_batch, rot_type, mode):
    """
    Processes a batch of SE(3) poses and hand model poses, unifying single and multi-grasp formats.
    - Single grasp input: hand_model_pose_batch [B, 23], se3_batch [B, 4, 4]
    - Multi grasp input: hand_model_pose_batch [B, num_grasps, 23], se3_batch [B, num_grasps, 4, 4]
    Returns consistently shaped 3D tensors: [B, num_grasps, pose_dim]
    """
    # Unify input shapes to [B, num_grasps, ...]
    if hand_model_pose_batch.dim() == 2:
        hand_model_pose_batch = hand_model_pose_batch.unsqueeze(1)
        se3_batch = se3_batch.unsqueeze(1)

    B, num_grasps = hand_model_pose_batch.shape[:2]

    # Reshape for batch processing
    se3_flat = se3_batch.view(B * num_grasps, 4, 4)
    pose_flat = hand_model_pose_batch.view(B * num_grasps, -1)

    # Process using existing logic
    matrix_batch = se3_flat[:, :3, :3]

    if rot_type == 'quat':
        rot_representation = transforms.matrix_to_quaternion(matrix_batch)
    elif rot_type == 'r6d':
        rot_representation = transforms.matrix_to_rotation_6d(matrix_batch)
    elif rot_type == 'axis':
        rot_representation = transforms.matrix_to_axis_angle(matrix_batch)
    elif rot_type == 'euler':
        rot_representation = transforms.matrix_to_euler_angles(matrix_batch, convention="XYZ")
    elif rot_type == 'map':
        log_map_full = se3_log_map(se3_flat)
        rot_representation = log_map_full[:, 3:]
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")

    input_translation_part = pose_flat[:, :3]
    input_joint_angle_part = pose_flat[:, 3 : 3 + JOINT_ANGLE_DIM]

    if rot_type == 'map':
        processed_translation_part = log_map_full[:, :3]
    else:
        processed_translation_part = input_translation_part

    processed_hand_pose_flat = torch.cat(
        [processed_translation_part,
         input_joint_angle_part,
         rot_representation],
        dim=1
    )

    # Normalize using robust function
    norm_pose_flat = norm_hand_pose_robust(processed_hand_pose_flat, rot_type=rot_type, mode=mode)

    # Reshape back to unified 3D format
    pose_dim = processed_hand_pose_flat.shape[-1]
    processed_hand_pose = processed_hand_pose_flat.view(B, num_grasps, pose_dim)
    norm_pose = norm_pose_flat.view(B, num_grasps, pose_dim)

    return norm_pose, processed_hand_pose


def process_hand_pose_train(data, rot_type, mode):
    se3 = data['se3']
    hand_model_pose_input = data['hand_model_pose']

    norm_pose, processed_hand_pose = _process_batch_pose_logic(se3, hand_model_pose_input, rot_type, mode)
    
    data['norm_pose'] = norm_pose
    data['hand_model_pose'] = processed_hand_pose
    
    return data

def process_hand_pose_single(data, rot_type, mode):
    """Process single hand pose data
    Args:
        data (dict): Dictionary containing:
            - se3 (torch.Tensor): Shape (4, 4) transformation matrix
            - hand_model_pose (torch.Tensor): Shape (23,) hand pose parameters (T, 16J, 4Q)
        rot_type (str): Rotation representation type
        mode (str): Normalization statistics mode
    Returns:
        dict: Processed data with normalized pose
    """
    se3_single = data['se3'].unsqueeze(0)
    hand_model_pose_single = data['hand_model_pose'].unsqueeze(0)
    
    norm_pose_batched, processed_hand_pose_batched = _process_batch_pose_logic(
        se3_single, hand_model_pose_single, rot_type, mode
    )
    
    data['norm_pose'] = norm_pose_batched.squeeze(0)
    data['hand_model_pose'] = processed_hand_pose_batched.squeeze(0)
    
    return data

def process_hand_pose_test(data, rot_type, mode):
    """
    Process data returned from ForMatchSceneLeapDataset, performing normalization and rotation representation transformation.
    Supports both list format and tensor format for multi-grasp data.

    Args:
        data (dict): Dictionary containing:
            - hand_model_pose: [B, num_grasps, 23] tensor or list of tensors
            - se3: [B, num_grasps, 4, 4] tensor or list of tensors
        rot_type (str): Rotation representation type
        mode (str): Processing mode

    Returns:
        dict: Processed data dictionary containing:
            - hand_model_pose: Processed hand pose
            - norm_pose: Normalized pose
    """
    if isinstance(data.get('hand_model_pose'), list):
        # Handle list format (legacy support)
        processed_data = []
        for i, (poses, se3s) in enumerate(zip(data['hand_model_pose'], data['se3'])):
            # poses: [num_grasps, 23], se3s: [num_grasps, 4, 4]
            item_data = {'hand_model_pose': poses, 'se3': se3s}
            processed_item = process_hand_pose(item_data, rot_type, mode)
            processed_data.append(processed_item)

        # Reorganize data
        data['norm_pose'] = [item['norm_pose'] for item in processed_data]
        data['hand_model_pose'] = [item['hand_model_pose'] for item in processed_data]

    else:
        # Standard tensor format: [B, num_grasps, 23] and [B, num_grasps, 4, 4]
        if 'se3' not in data or 'hand_model_pose' not in data:
            return data

        # Calculate valid mask for non-zero poses
        valid_mask = (data['hand_model_pose'].abs().sum(dim=-1) > 0)

        # Process using updated _process_batch_pose_logic that supports multi-grasp
        norm_pose_processed, processed_hand_pose_full = _process_batch_pose_logic(
            data['se3'], data['hand_model_pose'], rot_type, mode
        )

        # Apply valid mask to filter out zero poses
        valid_mask = valid_mask.unsqueeze(-1)
        processed_hand_pose_final = processed_hand_pose_full * valid_mask
        norm_pose_final = norm_pose_processed * valid_mask

        data['hand_model_pose'] = processed_hand_pose_final
        data['norm_pose'] = norm_pose_final

    return data

def _process_batch_pose(se3, hand_model_pose, rot_type, mode):
    """
    Helper for process_hand_pose when input is a standard batch.
    hand_model_pose is the new 23-dim pose.
    """
    return _process_batch_pose_logic(se3, hand_model_pose, rot_type, mode)

def process_hand_pose(data, rot_type, mode):
    """
    Process a batch of data from DataLoader.
    Supports both single and multi-grasp formats:
    - Single grasp: se3 [B, 4, 4], hand_model_pose [B, 23]
    - Multi grasp: se3 [B, num_grasps, 4, 4], hand_model_pose [B, num_grasps, 23]
    - List format: hand_model_pose as list of tensors

    Automatically selects processing method based on input data type.
    """
    if isinstance(data, dict) and not isinstance(data.get('hand_model_pose'), list):
        if 'se3' not in data or 'hand_model_pose' not in data:
            logging.warning("Missing required keys 'se3' or 'hand_model_pose' in input data. Skipping pose processing.")
            return data
        
        se3 = data['se3']
        hand_model_pose_input = data['hand_model_pose']
        
        norm_pose, processed_hand_pose = _process_batch_pose(se3, hand_model_pose_input, rot_type, mode)
        
        data['norm_pose'] = norm_pose
        data['hand_model_pose'] = processed_hand_pose
        
    elif isinstance(data, dict) and isinstance(data.get('hand_model_pose'), list):
        num_items = len(data['hand_model_pose'])
        if num_items == 0:
            data['norm_pose'] = []
            data['hand_model_pose'] = []
            return data

        norm_poses_list = [None] * num_items
        processed_hand_poses_list = [None] * num_items

        for i in range(num_items):
            se3_i = data['se3'][i]
            hand_model_pose_i = data['hand_model_pose'][i]
            
            norm_pose_i_batched, processed_hand_pose_i_batched = _process_batch_pose(
                se3_i, hand_model_pose_i, rot_type, mode
            )
            norm_poses_list[i] = norm_pose_i_batched
            processed_hand_poses_list[i] = processed_hand_pose_i_batched
            
        data['norm_pose'] = norm_poses_list
        data['hand_model_pose'] = processed_hand_poses_list
            
    else:
        raise ValueError(f"Unsupported data type for process_hand_pose: {type(data)}. Expected dict or dict of lists.")
    
    return data

def process_se3(data, rot_type, mode):
    matrix = data['se3'][:, :3, :3] 
    
    if rot_type == 'quat':
        rot_representation = transforms.matrix_to_quaternion(matrix)
    elif rot_type == 'r6d':
        rot_representation = transforms.matrix_to_rotation_6d(matrix)
    elif rot_type == 'axis':
        rot_representation = transforms.matrix_to_axis_angle(matrix)
    elif rot_type == 'euler':
        rot_representation = transforms.matrix_to_euler_angles(matrix, convention="XYZ")
    elif rot_type == 'map':
        se3_full_log_map = se3_log_map(data['se3'])
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")
    
    if rot_type == 'map':
        se3_rep_for_norm = se3_full_log_map
    else:
        trans = data['se3'][:, :3, 3]
        se3_rep_for_norm = torch.cat([trans, rot_representation], dim=-1)

    trans_to_norm = se3_rep_for_norm[..., :3]
    rot_to_norm = se3_rep_for_norm[..., 3:]

    map_distinction_for_trans = 'map' if rot_type == 'map' else 'not_map'
    trans_norm = normalize_trans_torch(trans_to_norm, mode, rot_type_for_map_distinction=map_distinction_for_trans)
    
    rot_norm = normalize_rot_torch(rot_to_norm, rot_type, mode) 
    
    se3_rep_norm = torch.cat([trans_norm, rot_norm], dim=-1)
    data['se3_rep_norm'] = se3_rep_norm

    return data

def decompose_hand_pose(hand_pose, rot_type='quat'):
    """
    Decompose hand pose into global translation, global rotation, and joint angles.
    Input hand_pose is the "processed_hand_pose" format: T (3) | J (JOINT_ANGLE_DIM) | R (rot_dim)
    """
    global_translation = hand_pose[:, :3]
    
    qpos = hand_pose[:, 3 : 3 + JOINT_ANGLE_DIM]
    
    rot_part_start_idx = 3 + JOINT_ANGLE_DIM
    
    if rot_type == 'quat':
        quat = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 4]
        global_rotation = transforms.quaternion_to_matrix(quat)
    elif rot_type == 'r6d':
        r6d = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 6]
        global_rotation = transforms.rotation_6d_to_matrix(r6d)
    elif rot_type == 'axis':
        axis_angle = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.axis_angle_to_matrix(axis_angle)
    elif rot_type == 'euler':
        euler = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.euler_angles_to_matrix(euler, convention="XYZ")
    elif rot_type == 'map':
        axis_angle_from_map = hand_pose[:, rot_part_start_idx : rot_part_start_idx + 3]
        global_rotation = transforms.axis_angle_to_matrix(axis_angle_from_map)
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")
    
    return global_translation, global_rotation, qpos