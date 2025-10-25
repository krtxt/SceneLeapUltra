from typing import Dict, Tuple, Union

import numpy as np
import pytorch3d.ops
import pytorch3d.structures
import scipy.spatial
import torch
from torch import Tensor

# try:
#     from csdf import compute_sdf, index_vertices_by_faces
# except ImportError:
#     from torchsdf import index_vertices_by_faces, compute_sdf
from torchsdf import compute_sdf, index_vertices_by_faces


def cal_distance_to_mesh(
    query_points: Tensor,
    vertices: Tensor,  # dtype=torch.float32
    faces: Tensor,  # dtype=torch.long
    scale: float = 0.1,
    with_closest_points: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """
    Calculate signed distances from query points to a mesh surface.

    Args:
        query_points: Query points tensor [B, N, 3]
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        scale: Scale factor for SDF computation
        with_closest_points: Whether to return closest points

    Returns:
        distance_physical: Signed distances [B, N]
        normals_physical: Surface normals [B, N, 3]
        closest_points_physical: Closest points [B, N, 3] (if with_closest_points=True)
    """
    device = query_points.device

    # vertices are now pre-scaled to physical world.
    # scale parameter defines the operational scale for compute_sdf.
    # Both query_points and vertices will be brought to this operational scale.

    operational_scale_factor = 1.0 / scale

    # 保证所有张量在同一设备上
    query_points_op_scale = query_points * operational_scale_factor
    vertices_op_scale = (vertices.to(device)) * operational_scale_factor
    faces = faces.to(device)

    face_verts_op_scale = index_vertices_by_faces(vertices_op_scale, faces)

    batch_size, n_points, _ = query_points.shape

    # Compute SDF
    dis, dis_signs, normals_ts, clst_points_ts = compute_sdf(
        query_points_op_scale.reshape(-1, 3), face_verts_op_scale
    )

    # Calculate closest points if requested
    if with_closest_points:
        # Use the closest points from SDF computation
        closest_points_op_scale = clst_points_ts

    dis = torch.sqrt(dis + 1e-8) * (-dis_signs)

    # Scale results back to physical world scale
    distance_physical = (dis * scale).reshape(batch_size, n_points)
    normals_physical = (normals_ts * dis_signs.unsqueeze(1)).reshape(
        batch_size, n_points, 3
    )  # Normals are unit vectors, their orientation is affected by dis_signs

    if with_closest_points:
        closest_points_physical = (closest_points_op_scale * scale).reshape(
            batch_size, n_points, 3
        )
        return distance_physical, normals_physical, closest_points_physical

    return distance_physical, normals_physical


def cal_q1(
    cfg: Dict,
    hand_model,
    vertices: Tensor,  # dtype=torch.float32
    faces: Tensor,  # dtype=torch.long
    scale: float,
    hand_pose: Tensor,
) -> Union[Tensor, float]:
    """
    Calculate Q1 metric for one object and multiple grasps.

    Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim] -> returns [B] tensor
    - Multi grasp: [B, num_grasps, pose_dim] -> returns [B, num_grasps] tensor

    Args:
        cfg: Configuration dictionary
        hand_model: Hand model instance
        vertices: Object mesh vertices [N_verts, 3]
        faces: Object mesh faces [N_faces, 3]
        scale: Scale factor
        hand_pose: Hand pose tensor [B, pose_dim] or [B, num_grasps, pose_dim]

    Returns:
        Q1 metrics tensor [B] or [B, num_grasps]
    """
    device = hand_pose.device
    original_shape = hand_pose.shape

    # Handle input format detection and preprocessing
    if hand_pose.dim() == 1:
        hand_pose = hand_pose.unsqueeze(0)
        is_single_sample = True
        is_multi_grasp = False
    elif hand_pose.dim() == 2:
        # Single grasp format: [B, pose_dim]
        is_single_sample = False
        is_multi_grasp = False
        batch_size = hand_pose.shape[0]
        num_grasps = 1
    elif hand_pose.dim() == 3:
        # Multi grasp format: [B, num_grasps, pose_dim]
        is_single_sample = False
        is_multi_grasp = True
        batch_size = hand_pose.shape[0]
        num_grasps = hand_pose.shape[1]
        # Flatten for batch processing: [B*num_grasps, pose_dim]
        hand_pose = hand_pose.view(-1, hand_pose.shape[-1])
    else:
        raise ValueError(
            f"hand_pose must be 1D, 2D or 3D tensor, got {hand_pose.dim()}D"
        )

    # Calculate hand pose decomposition for all samples
    rot_type = cfg.get("rot_type") if hasattr(cfg, "get") else cfg["rot_type"]
    global_translation, global_rotation, qpos = hand_model.decompose_hand_pose(
        hand_pose, rot_type=rot_type
    )
    current_status = hand_model.chain.forward_kinematics(qpos)

    # Process each sample to calculate Q1
    q1_results = []
    effective_batch_size = hand_pose.shape[0]

    for batch_idx in range(effective_batch_size):
        contact_points_object = []
        contact_normals = []

        # Collect contact points from each link for current sample
        for link_name in hand_model.mesh:
            if len(hand_model.mesh[link_name]["surface_points"]) == 0:
                continue

            surface_points = current_status[link_name].transform_points(
                hand_model.mesh[link_name]["surface_points"]
            )
            # Extract current sample's transformation
            curr_rotation = global_rotation[batch_idx : batch_idx + 1]  # [1, 3, 3]
            curr_translation = global_translation[batch_idx : batch_idx + 1]  # [1, 3]

            # Ensure surface_points has correct batch dimension
            if surface_points.dim() == 2:
                surface_points = surface_points.unsqueeze(0)  # [1, N, 3]
            elif surface_points.dim() == 3 and surface_points.shape[0] != 1:
                surface_points = surface_points[batch_idx : batch_idx + 1]  # [1, N, 3]

            surface_points = surface_points @ curr_rotation.transpose(
                -2, -1
            ) + curr_translation.unsqueeze(1)

            distances, normals, closest_points = cal_distance_to_mesh(
                surface_points, vertices, faces, scale=scale, with_closest_points=True
            )

            nms = cfg.get("nms") if hasattr(cfg, "get") else cfg["nms"]
            thres_contact = (
                cfg.get("thres_contact")
                if hasattr(cfg, "get")
                else cfg["thres_contact"]
            )

            if nms:
                nearest_point_index = distances.argmax()
                if -distances[0, nearest_point_index] < thres_contact:
                    contact_points_object.append(closest_points[0, nearest_point_index])
                    contact_normals.append(normals[0, nearest_point_index])
            else:
                contact_idx = (-distances < thres_contact).nonzero().reshape(-1)
                if len(contact_idx) != 0:
                    for idx in contact_idx:
                        contact_points_object.append(closest_points[0, idx])
                        contact_normals.append(normals[0, idx])

        # Handle case with no contact points
        if len(contact_points_object) == 0:
            contact_points_object.append(
                torch.tensor([0, 0, 0], dtype=torch.float, device=device)
            )
            contact_normals.append(
                torch.tensor([1, 0, 0], dtype=torch.float, device=device)
            )

        contact_points_object_np = torch.stack(contact_points_object).cpu().numpy()
        contact_normals_np = torch.stack(contact_normals).cpu().numpy()
        n_contact = len(contact_points_object_np)

        if (
            np.isnan(contact_points_object_np).any()
            or np.isnan(contact_normals_np).any()
        ):
            q1_results.append(0.0)
            continue

        # Calculate contact forces
        u1 = np.stack(
            [
                -contact_normals_np[:, 1],
                contact_normals_np[:, 0],
                np.zeros([n_contact], dtype=np.float32),
            ],
            axis=1,
        )
        u2 = np.stack(
            [
                np.ones([n_contact], dtype=np.float32),
                np.zeros([n_contact], dtype=np.float32),
                np.zeros([n_contact], dtype=np.float32),
            ],
            axis=1,
        )

        u = np.where(np.linalg.norm(u1, axis=1, keepdims=True) > 1e-8, u1, u2)
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        v = np.cross(u, contact_normals_np)

        m = cfg.get("m") if hasattr(cfg, "get") else cfg["m"]
        mu = cfg.get("mu") if hasattr(cfg, "get") else cfg["mu"]
        lambda_torque = (
            cfg.get("lambda_torque") if hasattr(cfg, "get") else cfg["lambda_torque"]
        )

        theta = np.linspace(0, 2 * np.pi, m, endpoint=False).reshape(m, 1, 1)
        contact_forces = (
            contact_normals_np + mu * (np.cos(theta) * u + np.sin(theta) * v)
        ).reshape(-1, 3)

        # Calculate wrench space
        origin = np.array([0, 0, 0], dtype=np.float32)
        wrenches = np.concatenate(
            [
                np.concatenate(
                    [
                        contact_forces,
                        lambda_torque
                        * np.cross(
                            np.tile(contact_points_object_np - origin, (m, 1)),
                            contact_forces,
                        ),
                    ],
                    axis=1,
                ),
                np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float32),
            ],
            axis=0,
        )

        try:
            wrench_space = scipy.spatial.ConvexHull(wrenches)
        except scipy.spatial._qhull.QhullError:
            q1_results.append(0.0)
            continue

        # Calculate Q1 metric
        q1 = np.array([1], dtype=np.float32)
        for equation in wrench_space.equations:
            q1 = np.minimum(q1, np.abs(equation[6]) / np.linalg.norm(equation[:6]))

        q1_results.append(q1.item())

    # Convert results to tensor and reshape according to input format
    q1_tensor = torch.tensor(q1_results, dtype=torch.float32, device=device)

    if is_single_sample:
        return q1_tensor.item()
    elif is_multi_grasp:
        # Reshape back to [B, num_grasps]
        return q1_tensor.view(batch_size, num_grasps)
    else:
        # Single grasp format: [B]
        return q1_tensor


def sample_surface_points(
    vertices: Tensor,  # dtype=torch.float32
    faces: Tensor,  # dtype=torch.long
    num_samples: int = 2000,
) -> Tensor:
    """Sample points from mesh surface using farthest point sampling"""
    mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
    dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(
        mesh, num_samples=100 * num_samples
    )
    surface_points = pytorch3d.ops.sample_farthest_points(
        dense_point_cloud, K=num_samples
    )[0][0].to(dtype=torch.float32, device=vertices.device)

    return surface_points.unsqueeze(0)


def cal_pen(
    cfg: Dict,
    hand_model,
    vertices: Tensor,  # dtype=torch.float32
    faces: Tensor,  # dtype=torch.long
    scale: float,
    hand_pose: Tensor,
    num_samples: int = 2000,
    skip_links: list = [],
) -> Union[Tensor, float]:
    """
    Calculate penetration depth between hand and object.

    Supports both single and multi-grasp formats:
    - Single grasp: [B, pose_dim] -> returns [B] tensor
    - Multi grasp: [B, num_grasps, pose_dim] -> returns [B, num_grasps] tensor

    Args:
        cfg: Configuration dictionary
        hand_model: Hand model instance
        vertices: Object mesh vertices [N_verts, 3]
        faces: Object mesh faces [N_faces, 3]
        scale: Scale factor
        hand_pose: Hand pose tensor [B, pose_dim] or [B, num_grasps, pose_dim]
        num_samples: Number of surface points to sample
        skip_links: List of link names to skip

    Returns:
        Penetration depth tensor [B] or [B, num_grasps]
    """
    device = hand_pose.device
    original_shape = hand_pose.shape

    # Handle input format detection and preprocessing
    if hand_pose.dim() == 1:
        hand_pose = hand_pose.unsqueeze(0)
        is_single_sample = True
        is_multi_grasp = False
    elif hand_pose.dim() == 2:
        # Single grasp format: [B, pose_dim]
        is_single_sample = False
        is_multi_grasp = False
        batch_size = hand_pose.shape[0]
        num_grasps = 1
    elif hand_pose.dim() == 3:
        # Multi grasp format: [B, num_grasps, pose_dim]
        is_single_sample = False
        is_multi_grasp = True
        batch_size = hand_pose.shape[0]
        num_grasps = hand_pose.shape[1]
        # Flatten for batch processing: [B*num_grasps, pose_dim]
        hand_pose = hand_pose.view(-1, hand_pose.shape[-1])
    else:
        raise ValueError(
            f"hand_pose must be 1D, 2D or 3D tensor, got {hand_pose.dim()}D"
        )

    # Sample surface points from object mesh (same for all grasps)
    surface_points = sample_surface_points(vertices, faces, num_samples=num_samples)
    # 保证在与 hand_pose 相同的设备上
    object_surface_points = surface_points.to(device)  # Already at physical scale

    # Calculate hand pose decomposition for all samples
    rot_type = cfg.get("rot_type") if hasattr(cfg, "get") else cfg["rot_type"]
    global_translation, global_rotation, qpos = hand_model.decompose_hand_pose(
        hand_pose, rot_type=rot_type
    )
    current_status = hand_model.chain.forward_kinematics(qpos)

    # Transform object points to hand coordinate frame for all samples
    # object_surface_points: [1, N, 3], global_translation: [B*num_grasps, 3], global_rotation: [B*num_grasps, 3, 3]
    effective_batch_size = hand_pose.shape[0]
    object_surface_points_expanded = object_surface_points.expand(
        effective_batch_size, -1, -1
    )  # [B*num_grasps, N, 3]
    x = (
        object_surface_points_expanded - global_translation.unsqueeze(1)
    ) @ global_rotation

    distances_per_link = []

    for link_name in hand_model.mesh:
        if link_name in skip_links:
            continue

        matrix = current_status[link_name].get_matrix()  # [B*num_grasps, 4, 4]
        x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
        x_local_flat = x_local.reshape(-1, 3)

        if "radius" in hand_model.mesh[link_name]:
            radius = hand_model.mesh[link_name]["radius"]
            dis_local = radius - x_local_flat.norm(dim=1)
        elif "face_verts" in hand_model.mesh[link_name]:
            face_verts = hand_model.mesh[link_name]["face_verts"]
            dis_local_sq, dis_signs, normals_ts, clst_points_ts = compute_sdf(
                x_local_flat, face_verts
            )
            dis_local = torch.sqrt(dis_local_sq + 1e-8) * (-dis_signs)
        else:
            batch_size_x, num_points_x = x.shape[0], x.shape[1]
            dis_local = torch.full(
                (batch_size_x * num_points_x,),
                -float("inf"),
                device=x_local_flat.device,
                dtype=x_local_flat.dtype,
            )

        distances_per_link.append(dis_local.reshape(x.shape[0], x.shape[1]))

    # Calculate maximum penetration across all links for each sample
    if distances_per_link:
        distances = torch.max(torch.stack(distances_per_link, dim=0), dim=0)[
            0
        ]  # [B*num_grasps, N]
        penetration_per_sample = torch.clamp(
            distances.max(dim=1)[0], min=0.0
        )  # [B*num_grasps]
    else:
        penetration_per_sample = torch.zeros(effective_batch_size, device=device)

    # Reshape results according to input format
    if is_single_sample:
        return penetration_per_sample.item()
    elif is_multi_grasp:
        # Reshape back to [B, num_grasps]
        return penetration_per_sample.view(batch_size, num_grasps)
    else:
        # Single grasp format: [B]
        return penetration_per_sample
