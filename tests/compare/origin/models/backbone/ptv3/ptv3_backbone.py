"""
PTv3 Backbone Adapter for SceneLeapPlus
适配PTv3模型以符合项目的backbone接口规范
"""

import torch
import torch.nn as nn
import numpy as np
from .ptv3 import PointTransformerV3
from ..pointnet2_utils import farthest_point_sample, index_points


class PTv3Backbone(nn.Module):
    """
    PTv3 Backbone适配器，用于将PTv3模型适配到SceneLeapPlus项目中

    输入格式: (B, N, C) - xyz + 可选的其他特征
    输出格式: xyz: (B, K, 3), features: (B, 512, K)
    """

    def __init__(self, cfg):
        super().__init__()

        # 从配置中提取参数，设置合理的默认值
        self.grid_size = getattr(cfg, 'grid_size', 0.02)  # 体素化网格大小
        self.use_pooling = getattr(cfg, 'use_pooling', False)
        # 新增：限制上下文点数量，避免注意力内存爆炸
        self.max_context_points = getattr(cfg, 'max_context_points', None)

        # 获取输入通道数，默认为6 (xyz + rgb)
        in_channels = getattr(cfg, 'in_channels', 6)

        # 修正: PTv3的in_channels应该只计算非坐标特征
        feat_channels = in_channels - 3 if in_channels > 3 else 1

        # PTv3模型配置
        ptv3_config = {
            'in_channels': feat_channels,  # 只传递非坐标特征的通道数
            'order': getattr(cfg, 'order', ("z", "z-trans")),  # 简化序列化顺序
            'stride': getattr(cfg, 'stride', (2, 2, 2, 2)),
            'enc_depths': getattr(cfg, 'enc_depths', (2, 2, 2, 6, 2)),
            'enc_channels': getattr(cfg, 'enc_channels', (32, 64, 128, 256, 512)),
            'enc_num_head': getattr(cfg, 'enc_num_head', (2, 4, 8, 16, 32)),
            'enc_patch_size': getattr(cfg, 'enc_patch_size', (1024, 1024, 1024, 1024, 1024)),
            'dec_depths': getattr(cfg, 'dec_depths', (2, 2, 2, 2)),
            'dec_channels': getattr(cfg, 'dec_channels', (64, 64, 128, 256)),
            'dec_num_head': getattr(cfg, 'dec_num_head', (4, 4, 8, 16)),
            'dec_patch_size': getattr(cfg, 'dec_patch_size', (1024, 1024, 1024, 1024)),
            'mlp_ratio': getattr(cfg, 'mlp_ratio', 4),
            'qkv_bias': getattr(cfg, 'qkv_bias', True),
            'attn_drop': getattr(cfg, 'attn_drop', 0.0),
            'proj_drop': getattr(cfg, 'proj_drop', 0.0),
            'drop_path': getattr(cfg, 'drop_path', 0.1),
            'cls_mode': False,  # 关闭分类模式，返回点特征而不是全局特征
            'shuffle_orders': getattr(cfg, 'shuffle_orders', False),  # 测试时不随机化
        }

        # 创建PTv3模型
        self.ptv3 = PointTransformerV3(**ptv3_config)

        # 特征投影层，将PTv3输出投影到512维
        # 修正：PTv3的输出维度由解码器决定，而不是编码器。使用dec_channels[0]。
        self.feature_proj = nn.Linear(ptv3_config['dec_channels'][0], 512)

        # 如果使用pooling，添加全局平均池化
        if self.use_pooling:
            self.gap = nn.AdaptiveAvgPool1d(1)

        # 子采样策略：random | fps
        self.sampling_strategy = getattr(cfg, 'sampling_strategy', 'random')

    def _prepare_ptv3_input(self, pointcloud):
        """
        将输入点云转换为PTv3所需的格式

        Args:
            pointcloud: (B, N, C) tensor - xyz + 可选的其他特征

        Returns:
            data_dict: PTv3所需的数据字典
        """
        B, N, C = pointcloud.shape
        device = pointcloud.device

        # 分离xyz和特征
        coord = pointcloud[..., :3]  # (B, N, 3)

        # 修正: feat应该只包含非坐标特征
        if C > 3:
            feat = pointcloud[..., 3:]
            C_feat = feat.shape[2]
        else:
            # 如果没有额外特征，创建一个全1的虚拟特征
            feat = torch.ones((B, N, 1), dtype=torch.float32, device=pointcloud.device)
            C_feat = 1

        # 创建batch索引
        batch_indices = []
        for b in range(B):
            batch_indices.append(torch.full((N,), b, dtype=torch.long, device=device))
        batch = torch.cat(batch_indices, dim=0)  # (B*N,)

        # 重塑数据
        coord = coord.reshape(-1, 3)  # (B*N, 3)
        feat = feat.reshape(-1, C_feat)   # (B*N, C_feat)

        # 计算网格坐标（体素化）
        grid_coord = torch.floor(coord / self.grid_size).long()

        # 创建offset（每个batch的点数累积和）
        offset = torch.cumsum(torch.tensor([N] * B, device=device, dtype=torch.long), dim=0)

        data_dict = {
            'coord': coord,
            'feat': feat,
            'grid_coord': grid_coord,
            'batch': batch,
            'offset': offset,
            'grid_size': self.grid_size
        }

        return data_dict

    def _subsample_per_batch(self, features_bchw, xyz_bnh3, strategy: str = "random"):
        """对每个 batch 进行子采样，限制 K 不超过 max_context_points。
        features_bchw: (B, 512, K)
        xyz_bnh3: (B, K, 3)
        strategy: "random" | "fps"
        返回裁剪/采样后的张量。
        """
        if self.max_context_points is None:
            return features_bchw, xyz_bnh3
        _, _, K = features_bchw.shape
        if K <= self.max_context_points:
            return features_bchw, xyz_bnh3

        if strategy == "fps":
            # 使用 FPS 在坐标上选取代表性点
            B = xyz_bnh3.shape[0]
            # 取采样数 S
            S = self.max_context_points
            # 计算每个 batch 的 FPS 索引
            fps_idx = farthest_point_sample(xyz_bnh3, S)  # (B, S)
            # 索引坐标
            xyz_bnh3 = index_points(xyz_bnh3, fps_idx)    # (B, S, 3)
            # 同步索引特征: 将特征转置到 (B, K, C) 再 index，再转回 (B, C, S)
            features_bhk = features_bchw.transpose(1, 2)   # (B, K, 512)
            features_bhk = index_points(features_bhk, fps_idx)  # (B, S, 512)
            features_bchw = features_bhk.transpose(1, 2)   # (B, 512, S)
            return features_bchw, xyz_bnh3
        else:
            # 随机子采样
            idx = torch.randperm(K, device=features_bchw.device)[: self.max_context_points]
            features_bchw = features_bchw[:, :, idx]
            xyz_bnh3 = xyz_bnh3[:, idx, :]
            return features_bchw, xyz_bnh3

    def _extract_features_and_coords(self, ptv3_output):
        """
        从PTv3输出中提取特征和坐标

        Args:
            ptv3_output: PTv3的输出特征 (B*N, C)

        Returns:
            xyz: (B, K, 3) - 采样后的坐标
            features: (B, 512, K) - 特征
        """
        # 获取点的数量
        B = len(ptv3_output.offset) if hasattr(ptv3_output, 'offset') else 1
        if B == 1:
            N = ptv3_output.feat.shape[0]
        else:
            N = (ptv3_output.offset[0]).item() if B > 1 else ptv3_output.feat.shape[0]

        # 获取特征和坐标
        features = self.feature_proj(ptv3_output.feat)  # (B*N, 512)
        xyz = ptv3_output.coord  # (B*N, 3)

        # 重新组织为批次格式
        if B > 1:
            # 分割成批次
            feature_list = []
            xyz_list = []
            start_idx = 0
            for i in range(B):
                end_idx = ptv3_output.offset[i].item() if i < len(ptv3_output.offset) else features.shape[0]
                feature_list.append(features[start_idx:end_idx].transpose(0, 1).unsqueeze(0))  # (1, 512, N_i)
                xyz_list.append(xyz[start_idx:end_idx].unsqueeze(0))  # (1, N_i, 3)
                start_idx = end_idx

            # 对齐为相同 K（以首个 batch 为参照），随后做上限裁剪
            target_K = feature_list[0].shape[2]
            features_batch = []
            xyz_batch = []
            for i in range(B):
                feat = feature_list[i]
                coord = xyz_list[i]
                current_K = feat.shape[2]
                if current_K > target_K:
                    feat = feat[:, :, :target_K]
                    coord = coord[:, :target_K, :]
                elif current_K < target_K:
                    pad_size = target_K - current_K
                    feat = torch.cat([feat, feat[:, :, -1:].repeat(1, 1, pad_size)], dim=2)
                    coord = torch.cat([coord, coord[:, -1:, :].repeat(1, pad_size, 1)], dim=1)
                features_batch.append(feat)
                xyz_batch.append(coord)
            features = torch.cat(features_batch, dim=0)  # (B, 512, K)
            xyz = torch.cat(xyz_batch, dim=0)  # (B, K, 3)
        else:
            # 单个批次情况
            K = features.shape[0]
            features = features.transpose(0, 1).unsqueeze(0)  # (1, 512, K)
            xyz = xyz.unsqueeze(0)  # (1, K, 3)

        # 新增：对 (B, 512, K)/(B, K, 3) 按 max_context_points 进行子采样
        features, xyz = self._subsample_per_batch(features, xyz, strategy=self.sampling_strategy)

        return xyz, features

    def forward(self, pointcloud):
        """
        前向传播

        Args:
            pointcloud: (B, N, C) tensor - xyz + 可选的其他特征

        Returns:
            xyz: (B, K, 3) - 采样后的坐标
            features: (B, 512, K) 或 (B, 512) - 特征（取决于use_pooling）
        """
        # 准备PTv3输入
        data_dict = self._prepare_ptv3_input(pointcloud)

        # PTv3前向传播
        ptv3_output = self.ptv3(data_dict)

        # 提取特征和坐标
        xyz, features = self._extract_features_and_coords(ptv3_output)

        # 如果使用全局池化，将特征池化为[B, 512]
        if self.use_pooling:
            # features: (B, 512, K) -> (B, 512, 1) -> (B, 512)
            features = self.gap(features).squeeze(-1)

        return xyz, features
