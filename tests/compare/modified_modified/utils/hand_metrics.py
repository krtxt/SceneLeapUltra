from typing import TYPE_CHECKING
import torch

from .hand_constants import QPOS_SLICE, SELF_PENETRATION_POINT_RADIUS

if TYPE_CHECKING:
    from .hand_model import HandModel


class HandMetrics:
    def __init__(self, hand_model: "HandModel"):
        """Initialize with hand model."""
        if hand_model.hand_pose is None:
            raise ValueError("Hand model not posed. Call set_parameters or __call__ first.")
        self.hand_model = hand_model

    def cal_self_penetration_energy(self) -> torch.Tensor:
        """Compute self-penetration energy."""
        hm = self.hand_model
        batch_size = hm.global_translation.shape[0]
        points = hm.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        link_indices = hm.global_index_to_link_index_penetration.clone().repeat(
            batch_size, 1
        )
        transforms = torch.zeros(
            batch_size, hm.n_keypoints, 4, 4, dtype=torch.float, device=hm.device
        )
        for link_name in hm.mesh:
            mask = link_indices == hm.link_name_to_link_index[link_name]
            cur = (
                hm.current_status[link_name]
                .get_matrix()
                .unsqueeze(1)
                .expand(batch_size, hm.n_keypoints, 4, 4)
            )
            transforms[mask] = cur[mask]
        points = torch.cat(
            [
                points,
                torch.ones(
                    batch_size,
                    hm.n_keypoints,
                    1,
                    dtype=torch.float,
                    device=hm.device,
                ),
            ],
            dim=2,
        )
        points = (transforms @ points.unsqueeze(3))[:, :, :3, 0]
        points = points @ hm.global_rotation.transpose(
            1, 2
        ) + hm.global_translation.unsqueeze(1)
        dis = (points.unsqueeze(1) - points.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
        dis = torch.where(
            dis < 1e-6, 1e6 * torch.ones_like(dis), dis
        )  # Ignore self-distance

        spen = (
            SELF_PENETRATION_POINT_RADIUS * 2 - dis
        )  # Each point is a sphere, this measures amount of penetration
        E_spen = torch.where(spen > 0, spen, torch.zeros_like(spen))
        return E_spen.sum((1, 2))

    def cal_joint_limit_energy(self) -> torch.Tensor:
        """
        Calculate joint limit energy

        Returns:
        E_joints: (N,) torch.Tensor
        """
        hm = self.hand_model
        joint_angles = hm.hand_pose[:, QPOS_SLICE]
        joint_limit_energy = torch.sum(
            (joint_angles > hm.joints_upper)
            * (joint_angles - hm.joints_upper),
            dim=-1,
        ) + torch.sum(
            (joint_angles < hm.joints_lower)
            * (hm.joints_lower - joint_angles),
            dim=-1,
        )
        return joint_limit_energy

    def cal_finger_finger_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-finger distance energy

        Returns
        -------
        E_ff: (N,) torch.Tensor
        """
        hm = self.hand_model
        if hm.contact_points is None:
            raise ValueError(
                "Contact points are not set. Cannot calculate finger-finger distance."
            )
        batch_size = hm.contact_points.shape[0]
        finger_finger_distance_energy = (
            -torch.cdist(hm.contact_points, hm.contact_points, p=2)
            .reshape(batch_size, -1)
            .sum(dim=-1)
        )
        return finger_finger_distance_energy

    def cal_finger_palm_distance_energy(self) -> torch.Tensor:
        """
        Calculate finger-palm distance energy

        Returns
        -------
        E_fp: (N,) torch.Tensor
        """
        hm = self.hand_model
        if hm.contact_points is None:
            raise ValueError(
                "Contact points are not set. Cannot calculate finger-palm distance."
            )
        palm_position = hm.global_translation[:, None, :]
        palm_finger_distance_energy = (
            -(palm_position - hm.contact_points).norm(dim=-1).sum(dim=-1)
        )
        return palm_finger_distance_energy

    def cal_table_penetration(
        self, table_pos: torch.Tensor, table_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate table penetration energy

        Args
        ----
        table_pos: (B, 3) torch.Tensor
            position of table surface
        table_normal: (B, 3) torch.Tensor
            normal of table

        Returns
        -------
        E_tpen: (B,) torch.Tensor
            table penetration energy
        """
        hm = self.hand_model
        # Two methods: use sampled points or meshes
        B1, D1 = table_pos.shape
        B2, D2 = table_normal.shape
        assert B1 == B2
        assert D1 == D2 == 3

        sampled_points_world_frame = hm.get_surface_points()
        B, N, D = sampled_points_world_frame.shape
        assert B == B1
        assert D == 3

        # Positive = above table, negative = below table
        signed_distance_from_table = torch.sum(
            (sampled_points_world_frame - table_pos.unsqueeze(1))
            * table_normal.unsqueeze(1),
            dim=-1,
        )

        penetration = torch.clamp(signed_distance_from_table, max=0.0)
        penetration = -penetration
        assert penetration.shape == (B, N)

        return penetration.sum(-1) 