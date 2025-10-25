import sys

sys.path.append("./third_party/pointnet2/")

import unittest

import torch
import torch.nn as nn
from pointnet2_modules import PointnetSAModuleVotes
from torch.functional import Tensor


class Pointnet2Backbone(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            Now expects 3 for RGB features (xyz coordinates are handled separately).

       Note:
            Input point cloud format: (B, N, 6) where 6 = xyz + rgb
            The _break_up_pc method separates xyz (first 3 channels) and
            rgb features (last 3 channels) automatically.
    """

    def __init__(self, cfg):
        super().__init__()
        
        # Output dimension (for interface compatibility with PTv3)
        self.output_dim = 512

        self.sa1 = PointnetSAModuleVotes(
            npoint=cfg.layer1.npoint,
            radius=cfg.layer1.radius_list[0],
            nsample=cfg.layer1.nsample_list[0],
            mlp=cfg.layer1.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=cfg.layer2.npoint,
            radius=cfg.layer2.radius_list[0],
            nsample=cfg.layer2.nsample_list[0],
            mlp=cfg.layer2.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=cfg.layer3.npoint,
            radius=cfg.layer3.radius_list[0],
            nsample=cfg.layer3.nsample_list[0],
            mlp=cfg.layer3.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=cfg.layer4.npoint,
            radius=cfg.layer4.radius_list[0],
            nsample=cfg.layer4.nsample_list[0],
            mlp=cfg.layer4.mlp_list,
            use_xyz=cfg.use_xyz,
            normalize_xyz=cfg.normalize_xyz
        )
        if cfg.use_pooling:
            self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.use_pooling = cfg.use_pooling

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: Tensor):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(Tensor)
                (B, N, 6) tensor - Point cloud with xyz + rgb
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, r, g, b)

            Returns
            ----------
                xyz: float32 Tensor of shape (B, K, 3)
                features: float32 Tensor of shape (B, D, K)
                inds: int64 Tensor of shape (B, K) values in [0, N-1]
        """
        xyz, features = self._break_up_pc(pointcloud)
        xyz, features, fps_inds = self.sa1(xyz, features)
        xyz, features, fps_inds = self.sa2(xyz, features)
        xyz, features, fps_inds = self.sa3(xyz, features)
        xyz, features, fps_inds = self.sa4(xyz, features)
        if self.use_pooling:
            features = self.gap(features).squeeze(-1)  # 移除最后一个维度，从[B, 512, 1]变为[B, 512]
        return xyz, features

