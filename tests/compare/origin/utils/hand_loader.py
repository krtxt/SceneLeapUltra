import json
import os
import pathlib
from typing import Dict, List, Union

import numpy as np
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
from urdf_parser_py.urdf import Box, Mesh, Robot, Sphere

try:
    from pytorch3d.ops import knn_points
    from pytorch3d.ops import \
        sample_farthest_points as pytorch3d_sample_farthest_points
    from pytorch3d.ops import \
        sample_points_from_meshes as pytorch3d_sample_points_from_meshes
    from pytorch3d.structures import Meshes as PyTorch3DMeshes
    _PYTORCH3D_AVAILABLE = True
except ImportError:
    _PYTORCH3D_AVAILABLE = False

from torchsdf import index_vertices_by_faces

from utils.path_utils import get_assets_folder


class HandLoader:
    def __init__(
        self,
        urdf_path: pathlib.Path,
        contact_points_path: pathlib.Path,
        penetration_points_path: pathlib.Path,
        n_surface_points: int,
        device: Union[str, torch.device],
        joint_names: List[str],
    ):
        self.urdf_path = urdf_path
        self.contact_points_path = contact_points_path
        self.penetration_points_path = penetration_points_path
        self.n_surface_points = n_surface_points
        self.device = device
        self.joint_names = joint_names
        self._already_warned_kaolin = False

    @staticmethod
    def get_mesh_file_path(urdf_path: pathlib.Path, mesh_filename: str) -> str:
        # Remove package:// prefix
        mesh_filename = (
            mesh_filename
            if not mesh_filename.startswith("package://")
            else mesh_filename.split("package://")[1]
        )

        # Check for the file
        file_path1 = os.path.join(os.path.dirname(urdf_path), mesh_filename)
        file_path2 = os.path.join(
            os.path.dirname(os.path.dirname(urdf_path)), mesh_filename
        )
        if not os.path.exists(file_path1) and not os.path.exists(file_path2):
            raise FileNotFoundError(f"Could not find file: {file_path1} or {file_path2}")

        if os.path.exists(file_path1) and os.path.exists(file_path2):
            raise AssertionError(
                f"Found file in both locations: {file_path1} and {file_path2}"
            )

        if os.path.exists(file_path1):
            return file_path1

        if os.path.exists(file_path2):
            return file_path2

        raise FileNotFoundError(f"Could not find file: {file_path1} or {file_path2}")

    def load(self) -> Dict[str, Union[pk.Chain, Dict, int, torch.Tensor]]:
        device = self.device

        chain = pk.build_chain_from_urdf(open(self.urdf_path).read()).to(
            dtype=torch.float, device=device
        )
        assert set(chain.get_joint_parameter_names()) == set(self.joint_names), (
            f"Only in self.chain: {set(chain.get_joint_parameter_names()) - set(self.joint_names)}\n"
            + f"Only in self.joint_names: {set(self.joint_names) - set(chain.get_joint_parameter_names())}"
        )
        assert chain.get_joint_parameter_names() == self.joint_names, (
            "Mismatch between ordering of joint names, this is an important and subtle bug to catch, as the order matters \n"
            + f"self.chain.get_joint_parameter_names() = {chain.get_joint_parameter_names()} \n"
            + f"self.joint_names = {self.joint_names} \n"
            + f"Different idxs: {[i for i, (a, b) in enumerate(zip(chain.get_joint_parameter_names(), self.joint_names)) if a != b]}"
        )

        robot = Robot.from_xml_file(self.urdf_path)
        n_dofs = len(chain.get_joint_parameter_names())

        penetration_points = json.load(open(self.penetration_points_path, "r"))
        contact_points = json.load(open(self.contact_points_path, "r"))

        mesh_dict = {}
        areas = {}
        for link in robot.links:
            if link.collision is None:
                print(f"In {self.urdf_path}, link {link.name} has no collision")
                continue

            mesh_dict[link.name] = {}

            # load collision mesh
            assert (
                len(link.collisions) == 1
            ), f"In {self.urdf_path}, link {link.name} has {len(link.collisions)} collisions, expected 1"

            collision = link.collision
            if isinstance(collision.geometry, Sphere):
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                mesh_dict[link.name]["radius"] = collision.geometry.radius
            elif isinstance(collision.geometry, Box):
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(
                    get_assets_folder() / "box" / "meshes" / "box.obj",
                    process=False,
                )
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            elif isinstance(collision.geometry, Mesh):
                # print(
                #     "WARNING: Collision geometry uses Mesh, need to check if this works"
                # )
                link_mesh = tm.load_mesh(
                    self.get_mesh_file_path(
                        urdf_path=self.urdf_path, mesh_filename=collision.geometry.filename
                    ),
                    process=False,
                )
                link_mesh.apply_scale(
                    np.array(collision.geometry.scale)
                    if collision.geometry.scale is not None
                    else np.ones(3)
                )
            else:
                raise ValueError(
                    f"Unknown collision geometry: {type(collision.geometry)}"
                )
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=device
            )
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if (
                hasattr(collision.geometry, "scale")
                and collision.geometry.scale is None
            ):
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(
                getattr(collision.geometry, "scale", [1, 1, 1]),
                dtype=torch.float,
                device=device,
            )
            translation = torch.tensor(
                getattr(collision.origin, "xyz", [0, 0, 0]),
                dtype=torch.float,
                device=device,
            )
            rotation = torch.tensor(
                transforms3d.euler.euler2mat(
                    *getattr(collision.origin, "rpy", [0, 0, 0])
                ),
                dtype=torch.float,
                device=device,
            )
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            mesh_dict[link.name].update(
                {
                    "vertices": vertices,
                    "faces": faces,
                }
            )

            if "radius" not in mesh_dict[link.name]:
                try:
                    mesh_dict[link.name]["face_verts"] = index_vertices_by_faces(
                        vertices, faces
                    )
                except ModuleNotFoundError:
                    if (
                        not hasattr(self, "_already_warned_kaolin")
                        or not self._already_warned_kaolin
                    ):
                        print("WARNING: kaolin not found, not computing face_verts")
                        self._already_warned_kaolin = True

            # load visual mesh
            if len(link.visuals) > 0:
                assert (
                    len(link.visuals) == 1
                ), f"In {self.urdf_path}, link {link.name} has {len(link.visuals)} visuals, expected 1"

                visual = link.visual
                link_mesh = tm.load_mesh(
                    self.get_mesh_file_path(
                        urdf_path=self.urdf_path, mesh_filename=visual.geometry.filename
                    ),
                )

                if isinstance(link_mesh, tm.Scene):
                    link_mesh = link_mesh.dump(concatenate=False)
                    if len(link_mesh) == 0:
                        raise ValueError(
                            f"In {self.urdf_path}, link {link.name} has visual {visual.geometry.filename} of type Scene but no meshes"
                        )
                    link_mesh = tm.util.concatenate(link_mesh)

                visual_vertices = torch.tensor(
                    link_mesh.vertices, dtype=torch.float, device=device
                )
                visual_faces = torch.tensor(
                    link_mesh.faces, dtype=torch.long, device=device
                )
                if hasattr(visual.geometry, "scale") and visual.geometry.scale is None:
                    visual.geometry.scale = [1, 1, 1]
                visual_scale = torch.tensor(
                    getattr(visual.geometry, "scale", [1, 1, 1]),
                    dtype=torch.float,
                    device=device,
                )
                visual_translation = torch.tensor(
                    getattr(visual.origin, "xyz", [0, 0, 0]),
                    dtype=torch.float,
                    device=device,
                )
                visual_rotation = torch.tensor(
                    transforms3d.euler.euler2mat(
                        *getattr(visual.origin, "rpy", [0, 0, 0])
                    ),
                    dtype=torch.float,
                    device=device,
                )
                visual_vertices = visual_vertices * visual_scale
                visual_vertices = (
                    visual_vertices @ visual_rotation.T + visual_translation
                )
                mesh_dict[link.name].update(
                    {
                        "visual_vertices": visual_vertices,
                        "visual_faces": visual_faces,
                    }
                )

            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(
                contact_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            penetration_keypoints = torch.tensor(
                penetration_points[link.name], dtype=torch.float32, device=device
            ).reshape(-1, 3)
            mesh_dict[link.name].update(
                {
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                }
            )
            areas[link.name] = tm.Trimesh(
                vertices.cpu().numpy(), faces.cpu().numpy()
            ).area.item()

        joints_lower = torch.tensor(
            [
                joint.limit.lower
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )
        joints_upper = torch.tensor(
            [
                joint.limit.upper
                for joint in robot.joints
                if joint.joint_type == "revolute"
            ],
            dtype=torch.float,
            device=device,
        )

        self._sample_surface_points(self.n_surface_points, mesh_dict, areas)
        return {
            "chain": chain,
            "mesh": mesh_dict,
            "areas": areas,
            "n_dofs": n_dofs,
            "joints_lower": joints_lower,
            "joints_upper": joints_upper,
        }

    def _sample_surface_points(self, n_surface_points: int, mesh_dict: Dict, areas: Dict) -> None:
        device = self.device

        if n_surface_points == 0:
            for link_name in mesh_dict:
                mesh_dict[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
            return

        if not _PYTORCH3D_AVAILABLE:
            print("WARNING: PyTorch3D is not available. Skipping surface point sampling.")
            for link_name in mesh_dict:
                mesh_dict[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
            return

        total_area = sum(areas.values())
        num_samples = dict(
            [
                (link_name, int(areas[link_name] / total_area * n_surface_points))
                for link_name in mesh_dict
            ]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(
            num_samples.values()
        )
        for link_name in mesh_dict:
            if num_samples[link_name] == 0:
                mesh_dict[link_name]["surface_points"] = torch.tensor(
                    [], dtype=torch.float, device=device
                ).reshape(0, 3)
                continue

            mesh = PyTorch3DMeshes(
                mesh_dict[link_name]["vertices"].unsqueeze(0),
                mesh_dict[link_name]["faces"].unsqueeze(0),
            )
            dense_point_cloud = pytorch3d_sample_points_from_meshes(
                mesh, num_samples=100 * num_samples[link_name]
            )
            surface_points = pytorch3d_sample_farthest_points(
                dense_point_cloud, K=num_samples[link_name]
            )[0][0]
            surface_points = surface_points.to(dtype=torch.float, device=device)
            mesh_dict[link_name]["surface_points"] = surface_points 