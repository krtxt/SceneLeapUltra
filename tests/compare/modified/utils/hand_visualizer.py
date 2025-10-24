from typing import Optional, TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import trimesh as tm

from utils.hand_constants import SELF_PENETRATION_POINT_RADIUS
from utils.point_utils import transform_points

if TYPE_CHECKING:
    from utils.hand_model import HandModel


class HandVisualizer:
    def __init__(self, hand_model: "HandModel"):
        """Initialize with hand model."""
        # Visualization methods will check pose when used
        self.hand_model = hand_model

    def get_plotly_data(
        self,
        i: int,
        opacity: float = 0.5,
        color: str = "lightblue",
        with_contact_points: bool = False,
        with_contact_candidates: bool = False,
        with_surface_points: bool = False,
        with_penetration_keypoints: bool = False,
        pose: Optional[np.ndarray] = None,
        visual: bool = False,
    ) -> list:
        """Get Plotly visualization data for hand model."""
        hm = self.hand_model

        # Check if HandModel has been initialized with pose parameters
        if hm.hand_pose is None:
            raise ValueError(
                "HandModel has not been initialized with a pose. "
                "Call `set_parameters` or `__call__` on the hand model instance before using visualization methods."
            )

        if pose is None:
            pose = np.eye(4, dtype=np.float32)
        assert pose.shape == (4, 4), f"pose shape: {pose.shape}"

        data = []
        assert hm.current_status is not None
        assert hm.global_translation is not None
        assert hm.global_rotation is not None
        for link_name in hm.mesh:
            v = hm.current_status[link_name].transform_points(
                hm.mesh[link_name]["visual_vertices"]
                if visual and "visual_vertices" in hm.mesh[link_name]
                else hm.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]

            # Process unified multi-grasp format
            # Convert i to (batch_idx, grasp_idx)
            batch_idx = i // hm.num_grasps
            grasp_idx = i % hm.num_grasps
            global_rotation = hm.global_rotation[batch_idx, grasp_idx]  # (3, 3)
            global_translation = hm.global_translation[batch_idx, grasp_idx]  # (3,)

            v = v @ global_rotation.T + global_translation
            v = v.detach().cpu().numpy()
            f = (
                (
                    hm.mesh[link_name]["visual_faces"]
                    if visual and "visual_faces" in hm.mesh[link_name]
                    else hm.mesh[link_name]["faces"]
                )
                .detach()
                .cpu()
            )
            v = transform_points(T=pose, points=v)
            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                    name="hand",
                )
            )
        if with_contact_points:
            assert hm.contact_points is not None
            contact_points = hm.contact_points[i].detach().cpu().numpy()
            contact_points = transform_points(T=pose, points=contact_points)
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=10),
                    name="contact points",
                )
            )
        if with_contact_candidates:
            contact_candidates = hm.get_contact_candidates()[i].detach().cpu().numpy()
            contact_candidates = transform_points(T=pose, points=contact_candidates)
            data.append(
                go.Scatter3d(
                    x=contact_candidates[:, 0],
                    y=contact_candidates[:, 1],
                    z=contact_candidates[:, 2],
                    mode="markers",
                    marker=dict(color="blue", size=5),
                    name="contact candidates",
                )
            )
        if with_surface_points:
            surface_points = hm.get_surface_points()[i].detach().cpu().numpy()
            surface_points = transform_points(T=pose, points=surface_points)
            data.append(
                go.Scatter3d(
                    x=surface_points[:, 0],
                    y=surface_points[:, 1],
                    z=surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=2),
                    name="surface points",
                )
            )

        if with_penetration_keypoints:
            penetration_keypoints = (
                hm.get_penetration_keypoints()[i].detach().cpu().numpy()
            )
            penetration_keypoints = transform_points(
                T=pose, points=penetration_keypoints
            )
            data.append(
                go.Scatter3d(
                    x=penetration_keypoints[:, 0],
                    y=penetration_keypoints[:, 1],
                    z=penetration_keypoints[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=3),
                    name="penetration_keypoints",
                )
            )
            for ii in range(penetration_keypoints.shape[0]):
                penetration_keypoint = penetration_keypoints[ii]
                assert penetration_keypoint.shape == (
                    3,
                ), f"{penetration_keypoint.shape}"
                mesh = tm.primitives.Capsule(
                    radius=SELF_PENETRATION_POINT_RADIUS, height=0
                )
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                data.append(
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.5,
                        name="penetration_keypoints_mesh",
                    )
                )

        return data

    def get_trimesh_data(self, i: int) -> tm.Trimesh:
        """
        Get full mesh

        Returns
        -------
        data: tm.Trimesh
        """
        hm = self.hand_model

        # Check if HandModel has been initialized with pose parameters
        if hm.hand_pose is None:
            raise ValueError(
                "HandModel has not been initialized with a pose. "
                "Call `set_parameters` or `__call__` on the hand model instance before using visualization methods."
            )

        data = tm.Trimesh()
        for link_name in hm.mesh:
            v = hm.current_status[link_name].transform_points(
                hm.mesh[link_name]["vertices"]
            )
            if len(v.shape) == 3:
                v = v[i]

            # Process unified multi-grasp format
            # Convert i to (batch_idx, grasp_idx)
            batch_idx = i // hm.num_grasps
            grasp_idx = i % hm.num_grasps
            global_rotation = hm.global_rotation[batch_idx, grasp_idx]  # (3, 3)
            global_translation = hm.global_translation[batch_idx, grasp_idx]  # (3,)

            v = v @ global_rotation.T + global_translation
            v = v.detach().cpu()
            f = hm.mesh[link_name]["faces"].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f)
        return data 