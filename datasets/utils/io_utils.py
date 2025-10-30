"""
I/O Utilities for SceneLeapPro Dataset

This module provides utility functions for loading images, meshes, and other data files
used by the SceneLeapPro dataset classes.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple, Any

import cv2
import torch
from pytorch3d.io import load_obj
try:
    # TexturesUV 用于为 PyTorch3D Meshes 附着纹理
    from pytorch3d.renderer import TexturesUV  # type: ignore
except Exception:  # 兜底：运行环境可能不含渲染模块
    TexturesUV = None  # type: ignore


@lru_cache(maxsize=None)
def load_depth_image(path: str) -> Optional[cv2.Mat]:
    """
    Load depth image from file path with caching.

    Args:
        path: Path to depth image file

    Returns:
        Loaded depth image or None if loading failed
    """
    if not os.path.exists(path):
        return None
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth


@lru_cache(maxsize=None)
def load_rgb_image(path: str) -> Optional[cv2.Mat]:
    """
    Load RGB image from file path with caching.

    Args:
        path: Path to RGB image file

    Returns:
        Loaded RGB image (converted from BGR) or None if loading failed
    """
    if not os.path.exists(path):
        return None
    rgb = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    # Convert BGR to RGB
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


@lru_cache(maxsize=None)
def load_mask_image(path: str) -> Optional[cv2.Mat]:
    """
    Load mask image from file path with caching.

    Args:
        path: Path to mask image file

    Returns:
        Loaded mask image or None if loading failed
    """
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return mask


@lru_cache(maxsize=None)
def load_object_mesh(
    obj_root_dir: str, object_code: str, mesh_scale: float
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load object mesh from file path with caching.

    Args:
        obj_root_dir: Root directory containing object meshes
        object_code: Object code identifier
        mesh_scale: Scale factor to apply to mesh vertices

    Returns:
        Tuple of (vertices, faces) tensors or (None, None) if loading failed
    """
    mesh_path = os.path.join(obj_root_dir, object_code, "mesh", "simplified.obj")
    if not os.path.exists(mesh_path):
        return None, None
    try:
        verts, faces, _ = load_obj(mesh_path, load_textures=False)
        verts = verts * mesh_scale
        return verts, faces.verts_idx
    except Exception as e:
        return None, None


@lru_cache(maxsize=None)
def load_object_mesh_with_textures(
    obj_root_dir: str, object_code: str, mesh_scale: float
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Any]]:
    """
    加载物体 mesh（含纹理信息，若可用）。

    返回:
        - verts: (V,3) 经过 `mesh_scale` 缩放后的顶点
        - faces: (F,3) 面索引
        - textures: 可直接传入 PyTorch3D `Meshes(textures=...)` 的 `TexturesUV`，若不可用则为 None

    注意:
        - 纹理坐标与贴图与顶点的刚体/缩放变换无关，因此外部可安全地在返回的 verts 上做位姿/居中等操作。
    """
    mesh_path = os.path.join(obj_root_dir, object_code, "mesh", "simplified.obj")
    if not os.path.exists(mesh_path):
        return None, None, None

    try:
        verts, faces, aux = load_obj(mesh_path, load_textures=True)
        verts = verts * mesh_scale

        faces_idx = faces.verts_idx
        textures = None

        # 当渲染模块可用且OBJ内含有纹理坐标与贴图时，构造 TexturesUV
        if TexturesUV is not None and hasattr(faces, "textures_idx") and getattr(faces, "textures_idx") is not None:
            faces_uvs = faces.textures_idx
            verts_uvs = getattr(aux, "verts_uvs", None)
            tex_images = getattr(aux, "texture_images", None)

            if verts_uvs is not None and tex_images:
                # 纹理贴图字段在不同版本 PyTorch3D 中可能是 dict 或 list，做鲁棒处理
                if isinstance(tex_images, dict):
                    first_img = next(iter(tex_images.values()))
                elif isinstance(tex_images, (list, tuple)) and len(tex_images) > 0:
                    first_img = tex_images[0]
                else:
                    first_img = None

                if first_img is not None:
                    # 统一转为 torch.Tensor，范围一般为[0,1]
                    if not torch.is_tensor(first_img):
                        import numpy as np  # 局部导入，避免不必要依赖
                        if hasattr(first_img, "numpy"):
                            first_img = torch.from_numpy(first_img.numpy())
                        elif isinstance(first_img, np.ndarray):
                            first_img = torch.from_numpy(first_img)
                        else:
                            # 无法识别的对象，放弃纹理
                            first_img = None

                    if first_img is not None:
                        # 统一到 [0,1] 范围，并添加 batch 维度
                        maps = first_img.to(torch.float32)
                        if maps.max() > 1.0:
                            maps = maps / 255.0
                        maps = maps.unsqueeze(0)
                        textures = TexturesUV(
                            maps=maps,
                            faces_uvs=[faces_uvs],
                            verts_uvs=[verts_uvs],
                        )

        return verts, faces_idx, textures
    except Exception:
        # 回退到无纹理加载
        try:
            verts, faces, _ = load_obj(mesh_path, load_textures=False)
            verts = verts * mesh_scale
            return verts, faces.verts_idx, None
        except Exception:
            return None, None, None


def validate_file_paths(*paths: str) -> bool:
    """
    Validate that all provided file paths exist.

    Args:
        *paths: Variable number of file paths to validate

    Returns:
        True if all paths exist, False otherwise
    """
    return all(os.path.exists(path) for path in paths)


def get_depth_view_indices(depth_dir: str) -> list:
    """
    Get sorted list of depth view indices from depth directory.

    Args:
        depth_dir: Directory containing depth images

    Returns:
        Sorted list of depth view indices
    """
    depth_view_indices = []
    if os.path.exists(depth_dir) and os.path.isdir(depth_dir):
        for f_name in sorted(os.listdir(depth_dir)):
            if f_name.endswith(".png"):
                try:
                    depth_view_indices.append(int(os.path.splitext(f_name)[0]))
                except ValueError:
                    pass
    return depth_view_indices


def construct_file_paths(scene_dir: str, depth_view_index: int) -> Tuple[str, str, str]:
    """
    Construct file paths for depth, RGB, and instance mask images.

    Args:
        scene_dir: Scene directory path
        depth_view_index: Depth view index

    Returns:
        Tuple of (depth_path, rgb_path, instance_mask_path)
    """
    depth_filename = f"{depth_view_index:06d}.png"
    depth_path = os.path.join(scene_dir, "train_pbr/000000/depth", depth_filename)
    rgb_path = os.path.join(scene_dir, "train_pbr/000000/rgb", depth_filename)
    instance_mask_path = os.path.join(scene_dir, "InstanceMask", depth_filename)
    return depth_path, rgb_path, instance_mask_path


def load_scene_images(
    scene_dir: str, depth_view_index: int
) -> Tuple[Optional[cv2.Mat], Optional[cv2.Mat], Optional[cv2.Mat]]:
    """
    Load all scene images (depth, RGB, instance mask) for a given view.

    Args:
        scene_dir: Scene directory path
        depth_view_index: Depth view index

    Returns:
        Tuple of (depth_image, rgb_image, instance_mask_image) or None values if loading failed
    """
    depth_path, rgb_path, instance_mask_path = construct_file_paths(
        scene_dir, depth_view_index
    )

    depth_image = load_depth_image(depth_path)
    rgb_image = load_rgb_image(rgb_path)
    instance_mask_image = load_mask_image(instance_mask_path)

    return depth_image, rgb_image, instance_mask_image


def clear_image_cache():
    """Clear all cached images to free memory."""
    load_depth_image.cache_clear()
    load_rgb_image.cache_clear()
    load_mask_image.cache_clear()
    load_object_mesh.cache_clear()
