"""
PointNext Backbone Wrapper for SceneLeapUltra

Wraps the official PointNext implementation from OpenPoints to provide
a unified interface compatible with PointNet2, PTv3 and the decoder models.

Model source: from openpoints.models.backbone.pointnext import PointNextEncoder
"""
import logging
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Add third_party path if needed
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
openpoints_path = os.path.join(project_root, "third_party")
if openpoints_path not in sys.path:
    sys.path.insert(0, openpoints_path)

try:
    from openpoints.models.backbone.pointnext import PointNextEncoder
    POINTNEXT_AVAILABLE = True
    logging.info("PointNext imported successfully")
except ImportError as e:
    logging.warning(f"PointNext not available: {e}")
    logging.warning("Please ensure OpenPoints dependencies are installed or use PTv3/PointNet2 instead")
    POINTNEXT_AVAILABLE = False
    PointNextEncoder = None


class PointNextBackbone(nn.Module):
    """
    PointNext Backbone wrapper for SceneLeapUltra.
    
    Provides a unified interface compatible with PointNet2 and PTv3:
    - Input: (B, N, C) where C = xyz(3) + optional_features
    - Output: (xyz, features) where xyz is (B, K, 3), features is (B, out_dim, K)
    
    PointNext is a unified point cloud analysis model that:
    - Uses efficient hierarchical architecture with InvResMLP blocks
    - Supports both classification and segmentation tasks
    - Provides better performance than PointNet++
    
    Args:
        cfg: Configuration object with PointNext parameters
            - num_points: Number of input points (default: 8192)
            - num_tokens: Number of output tokens K (default: 128)
            - out_dim: Output feature dimension (default: 512)
            - width: Base channel width (default: 32)
            - blocks: Number of InvResMLP blocks per stage (list, default: [1,1,1,1,1])
            - strides: Stride for each stage (list, default: [1,4,4,4,4])
            - use_res: Use residual connections (default: True)
            - radius: Ball query radius (default: 0.1)
            - nsample: Number of samples per ball query (default: 32)
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        if not POINTNEXT_AVAILABLE:
            raise ImportError(
                "PointNext not available. Please install OpenPoints: "
                "cd third_party/openpoints && pip install -e ."
            )
        
        self.logger = logging.getLogger(__name__)
        
        # Basic parameters
        self.num_points = getattr(cfg, 'num_points', 8192)
        self.num_tokens = getattr(cfg, 'num_tokens', 128)
        self.out_dim = getattr(cfg, 'out_dim', 512)
        
        # PointNext architecture parameters
        self.width = getattr(cfg, 'width', 32)
        self.blocks = list(getattr(cfg, 'blocks', [1, 1, 1, 1, 1]))
        self.strides = list(getattr(cfg, 'strides', [1, 4, 4, 4, 4]))
        
        # Local aggregation parameters
        self.use_res = getattr(cfg, 'use_res', True)
        self.radius = getattr(cfg, 'radius', 0.1)
        self.nsample = getattr(cfg, 'nsample', 32)
        
        # Input/output configuration
        self.in_channels = getattr(cfg, 'input_feature_dim', 3)  # xyz only by default
        self.num_classes = self.out_dim  # For segmentation mode
        
        # Compatibility flags
        self.use_xyz = getattr(cfg, 'use_xyz', True)
        self.normalize_xyz = getattr(cfg, 'normalize_xyz', True)
        
        # Build PointNext encoder
        self.encoder = self._build_pointnext_encoder(cfg)
        
        # FPS sampling for token extraction
        self.use_fps = getattr(cfg, 'use_fps', True)
        
        # Output projection to ensure correct dimension
        encoder_out_dim = self._get_encoder_output_dim()
        if encoder_out_dim != self.out_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(encoder_out_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
                nn.ReLU(inplace=True)
            )
            self.logger.info(
                f"Added output projection: {encoder_out_dim} -> {self.out_dim}"
            )
        else:
            self.output_projection = nn.Identity()
        
        self.logger.info(
            f"Initialized PointNextBackbone: "
            f"num_points={self.num_points}, num_tokens={self.num_tokens}, "
            f"out_dim={self.out_dim}, width={self.width}, "
            f"blocks={self.blocks}, strides={self.strides}"
        )
        
        # Debug info for potential errors
        self.logger.debug(
            f"[PointNext] Input channels: {self.in_channels}, "
            f"Encoder output dim: {encoder_out_dim}, "
            f"Final output dim: {self.out_dim}"
        )
    
    def _get_encoder_output_dim(self):
        """Calculate encoder output dimension based on architecture."""
        # PointNext doubles width after each downsampling (stride != 1)
        # Replicate the logic from PointNextEncoder.__init__
        width = self.width
        for stride in self.strides:
            if stride != 1:
                width *= 2
        return width
    
    def _build_pointnext_encoder(self, cfg):
        """Build PointNext encoder based on configuration."""
        try:
            # Additional parameters
            sa_layers = getattr(cfg, 'sa_layers', 1)
            sa_use_res = getattr(cfg, 'sa_use_res', False)
            expansion = getattr(cfg, 'expansion', 4)
            reduction = getattr(cfg, 'reduction', 4)
            
            # Aggregation args
            aggr_args = {
                'feature_type': 'dp_fj',
                'reduction': 'max'
            }
            
            # Group args - need to use EasyDict for attribute access
            from easydict import EasyDict
            group_args = EasyDict({'NAME': 'ballquery'})
            
            # Sampler selection
            sampler = getattr(cfg, 'sampler', 'fps')
            
            # Build encoder with parameters
            encoder = PointNextEncoder(
                in_channels=self.in_channels,
                width=self.width,
                blocks=self.blocks,
                strides=self.strides,
                nsample=self.nsample,
                radius=self.radius,
                aggr_args=aggr_args,
                group_args=group_args,
                sa_layers=sa_layers,
                sa_use_res=sa_use_res,
                norm_args={'norm': 'bn'},
                act_args={'act': 'relu'},
                conv_args={},  # Empty dict instead of None
                expansion=expansion,
                sampler=sampler,  # 'fps' or 'random'
            )
            
            self.logger.info(
                f"Built PointNext encoder: in_channels={self.in_channels}, "
                f"width={self.width}, stages={len(self.blocks)}, "
                f"sa_layers={sa_layers}, expansion={expansion}"
            )
            
            return encoder
            
        except Exception as e:
            self.logger.error(f"Failed to build PointNext encoder: {e}")
            self.logger.error(
                f"Config: width={self.width}, blocks={self.blocks}, "
                f"strides={self.strides}, in_channels={self.in_channels}"
            )
            raise
    
    def _fps_sample(self, xyz: torch.Tensor, features: torch.Tensor, 
                    num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Farthest Point Sampling to extract K tokens from N points.
        
        Args:
            xyz: (B, N, 3) coordinates
            features: (B, N, D) features
            num_samples: K, number of tokens to sample
        
        Returns:
            sampled_xyz: (B, K, 3)
            sampled_features: (B, K, D)
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        if num_samples >= N:
            self.logger.warning(
                f"num_samples ({num_samples}) >= N ({N}), returning all points"
            )
            return xyz, features
        
        # Farthest Point Sampling
        try:
            from pointnet2_ops import pointnet2_utils
            # FPS expects (B, N, 3)
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_samples)  # (B, K)
            
            # Gather sampled points and features
            # fps_idx: (B, K) -> (B, K, 1) -> (B, K, 3) for xyz
            fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, 3)
            sampled_xyz = torch.gather(xyz, 1, fps_idx_expanded)  # (B, K, 3)
            
            # (B, K, 1) -> (B, K, D) for features
            fps_idx_expanded = fps_idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            sampled_features = torch.gather(features, 1, fps_idx_expanded)  # (B, K, D)
            
            return sampled_xyz, sampled_features
            
        except ImportError:
            self.logger.warning(
                "pointnet2_ops not available, using random sampling instead"
            )
            # Fallback to random sampling
            indices = torch.randperm(N, device=device)[:num_samples]
            indices = indices.unsqueeze(0).expand(B, -1)  # (B, K)
            
            indices_xyz = indices.unsqueeze(-1).expand(-1, -1, 3)
            sampled_xyz = torch.gather(xyz, 1, indices_xyz)
            
            indices_feat = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            sampled_features = torch.gather(features, 1, indices_feat)
            
            return sampled_xyz, sampled_features
    
    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of PointNext backbone.
        
        Args:
            pos: (B, N, C) point cloud tensor
                 C can be 3 (xyz), 4 (xyz+mask), 6 (xyz+rgb), or 7 (xyz+rgb+mask)
        
        Returns:
            xyz: (B, K, 3) sampled point coordinates
            features: (B, D, K) sampled point features (D = out_dim)
        """
        B, N, C = pos.shape
        device = pos.device
        
        # Split coordinates and features
        xyz = pos[..., :3].contiguous()  # (B, N, 3)
        
        if C > 3:
            # Has additional features (rgb, mask, or both)
            features = pos[..., 3:].contiguous()  # (B, N, C-3)
        else:
            # Only xyz, no additional features
            features = None
        
        # Normalize xyz if required
        if self.normalize_xyz:
            # Center and normalize
            xyz_mean = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            xyz_centered = xyz - xyz_mean
            xyz_std = xyz_centered.std(dim=1, keepdim=True).clamp(min=1e-6)
            xyz_normalized = xyz_centered / xyz_std
        else:
            xyz_normalized = xyz
        
        # Prepare input for PointNext
        # PointNext expects (B, C, N) format for features
        if features is not None:
            # Concatenate xyz with features if use_xyz is True
            if self.use_xyz:
                point_features = torch.cat([xyz_normalized, features], dim=-1)  # (B, N, 3+F)
            else:
                point_features = features  # (B, N, F)
        else:
            # Only xyz as features
            point_features = xyz_normalized  # (B, N, 3)
        
        # Transpose to (B, C, N) for PointNext
        point_features = point_features.transpose(1, 2).contiguous()  # (B, C, N)
        
        # Debug info
        self.logger.debug(
            f"[PointNext] Input shape: pos={pos.shape}, "
            f"xyz_normalized={xyz_normalized.shape}, "
            f"point_features={point_features.shape}"
        )
        
        # Forward through PointNext encoder
        try:
            # PointNext encoder returns (p_list, f_list)
            # p_list: [(B, N, 3), (B, N1, 3), ..., (B, Nk, 3)]
            # f_list: [(B, C0, N), (B, C1, N1), ..., (B, Ck, Nk)]
            p_list, f_list = self.encoder(xyz_normalized, point_features)
            
            # Use the last stage features
            encoded_xyz = p_list[-1]  # (B, Nk, 3)
            encoded_features = f_list[-1]  # (B, Ck, Nk)
            
            self.logger.debug(
                f"[PointNext] Encoder output shape: xyz={encoded_xyz.shape}, "
                f"features={encoded_features.shape}"
            )
            
        except Exception as e:
            self.logger.error(f"PointNext encoder failed: {e}")
            self.logger.error(
                f"Input shapes - xyz: {xyz_normalized.shape}, "
                f"features: {point_features.shape}"
            )
            import traceback
            traceback.print_exc()
            raise
        
        # Transpose features back to (B, Nk, D)
        encoded_features = encoded_features.transpose(1, 2).contiguous()  # (B, Nk, D)
        
        # Get the number of points from encoder output
        Nk = encoded_xyz.shape[1]
        
        # Sample K tokens using FPS if needed
        if self.use_fps and self.num_tokens < Nk:
            sampled_xyz, sampled_features = self._fps_sample(
                encoded_xyz, encoded_features, self.num_tokens
            )
            K = self.num_tokens
        else:
            # Use all points or simple stride sampling
            if self.num_tokens < Nk:
                stride = Nk // self.num_tokens
                indices = torch.arange(0, Nk, stride, device=device)[:self.num_tokens]
                sampled_xyz = encoded_xyz[:, indices, :]  # (B, K, 3)
                sampled_features = encoded_features[:, indices, :]  # (B, K, D)
                K = sampled_xyz.shape[1]
            else:
                sampled_xyz = encoded_xyz
                sampled_features = encoded_features
                K = Nk
        
        # Apply output projection if needed
        sampled_features = self.output_projection(sampled_features)  # (B, K, out_dim)
        
        # Transpose features to (B, out_dim, K) to match PointNet2 interface
        output_features = sampled_features.transpose(1, 2).contiguous()  # (B, out_dim, K)
        
        self.logger.debug(
            f"[PointNext] Output shapes: xyz={sampled_xyz.shape}, "
            f"features={output_features.shape}"
        )
        
        # Sanity check
        assert sampled_xyz.shape == (B, K, 3), \
            f"xyz shape mismatch: expected ({B}, {K}, 3), got {sampled_xyz.shape}"
        assert output_features.shape == (B, self.out_dim, K), \
            f"features shape mismatch: expected ({B}, {self.out_dim}, {K}), got {output_features.shape}"
        
        return sampled_xyz, output_features


# Unit test
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    
    from omegaconf import OmegaConf
    
    # Test configuration
    cfg_dict = {
        'name': 'pointnext',
        'num_points': 8192,
        'num_tokens': 128,
        'out_dim': 512,
        'width': 32,
        'blocks': [1, 1, 1, 1, 1],
        'strides': [1, 4, 4, 4, 4],
        'use_res': True,
        'radius': 0.1,
        'nsample': 32,
        'input_feature_dim': 3,
        'use_xyz': True,
        'normalize_xyz': True,
        'use_fps': True,
    }
    
    cfg = OmegaConf.create(cfg_dict)
    
    print("=" * 80)
    print("Testing PointNext Backbone")
    print("=" * 80)
    
    # Create model
    model = PointNextBackbone(cfg)
    model.eval()
    
    # Test input
    batch_size = 2
    num_points = 8192
    
    # Test case 1: xyz only
    print("\n[Test 1] Input: (B, N, 3) - xyz only")
    pos_xyz = torch.randn(batch_size, num_points, 3)
    xyz_out, feat_out = model(pos_xyz)
    print(f"Input shape: {pos_xyz.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 128, 3)
    assert feat_out.shape == (batch_size, 512, 128)
    print("✓ Test 1 passed!")
    
    # Test case 2: xyz + rgb
    print("\n[Test 2] Input: (B, N, 6) - xyz + rgb")
    pos_xyzrgb = torch.randn(batch_size, num_points, 6)
    xyz_out, feat_out = model(pos_xyzrgb)
    print(f"Input shape: {pos_xyzrgb.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 128, 3)
    assert feat_out.shape == (batch_size, 512, 128)
    print("✓ Test 2 passed!")
    
    # Test case 3: Different output dimensions
    print("\n[Test 3] Custom output dimension: 256")
    cfg_dict['out_dim'] = 256
    cfg_dict['num_tokens'] = 256
    cfg = OmegaConf.create(cfg_dict)
    model2 = PointNextBackbone(cfg)
    model2.eval()
    
    pos_xyz = torch.randn(batch_size, num_points, 3)
    xyz_out, feat_out = model2(pos_xyz)
    print(f"Input shape: {pos_xyz.shape}")
    print(f"Output xyz shape: {xyz_out.shape}")
    print(f"Output features shape: {feat_out.shape}")
    assert xyz_out.shape == (batch_size, 256, 3)
    assert feat_out.shape == (batch_size, 256, 256)
    print("✓ Test 3 passed!")
    
    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)

