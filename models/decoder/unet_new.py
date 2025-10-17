import torch
import torch.nn as nn
import logging
from typing import Dict
from einops import rearrange

from models.utils.diffusion_utils import (
    timestep_embedding,
    ResBlock,
    SpatialTransformer,
    GraspNet,
)
from models.backbone import build_backbone
from models.utils.text_encoder import TextConditionProcessor, PosNegTextEncoder

class UNetModel(nn.Module):
    """
    U-Net model for predicting noise in a grasp synthesis diffusion process.

    This model takes a noisy grasp pose, timestep, and conditions (scene point cloud, text)
    to predict the added noise, using ResBlocks and Spatial Transformers.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        # Input Dimension Config
        if not hasattr(cfg, 'rot_type'):
            raise ValueError("'rot_type' must be specified in the config.")
        
        rot_to_dim = {'quat': 23, 'r6d': 25}
        if cfg.rot_type not in rot_to_dim:
            raise ValueError(f"Unsupported rot_type '{cfg.rot_type}'. Must be one of {list(rot_to_dim.keys())}")
        self.d_x = rot_to_dim[cfg.rot_type]

        # Model Architecture Config
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.use_position_embedding = cfg.use_position_embedding

        # Transformer Config
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim

        # Conditioning Modules
        # Adjust backbone config based on use_object_mask
        backbone_cfg = self._adjust_backbone_config(cfg.backbone, cfg.use_object_mask)
        self.scene_model = build_backbone(backbone_cfg)
        self.text_encoder = None  # Lazily initialized
        self.text_processor = TextConditionProcessor(
            text_dim=512,
            context_dim=self.context_dim,
            dropout=self.transformer_dropout,
            use_negative_prompts=getattr(cfg, 'use_negative_prompts', True),
        )
        self.grasp_encoder = GraspNet(input_dim=self.d_x, output_dim=512)

        # Text Conditioning Config
        self.use_text_condition = cfg.use_text_condition
        self.text_dropout_prob = cfg.text_dropout_prob
        self.use_negative_prompts = getattr(cfg, 'use_negative_prompts', True)
        self.use_object_mask = cfg.use_object_mask

        # Note: Grasp count control is handled in diffusion core, not here

        self._build_model_layers(cfg)

    def _adjust_backbone_config(self, backbone_cfg, use_object_mask):
        """
        Adjusts the backbone configuration based on use_object_mask setting.
        Supports both PointNet2 and PTv3 backbones.

        Args:
            backbone_cfg: Original backbone configuration
            use_object_mask: Whether to use object mask as additional input

        Returns:
            Modified backbone configuration with correct input dimensions
        """
        import copy
        adjusted_cfg = copy.deepcopy(backbone_cfg)

        # Calculate total input dimension:
        # - XYZ coordinates: 3 channels (handled automatically by PointNet2, explicitly by PTv3)
        # - RGB features: 3 channels
        # - Object mask (if enabled): +1 channel
        total_input_dim = 6 + (1 if use_object_mask else 0)  # xyz + rgb + optional mask
        feature_input_dim = 3 + (1 if use_object_mask else 0)  # rgb + optional mask (for PointNet2)

        backbone_name = getattr(adjusted_cfg, 'name', '').lower()

        if backbone_name == 'pointnet2':
            # For PointNet2: adjust the first layer's mlp_list first parameter
            # PointNet2 automatically handles xyz coordinates when use_xyz=True
            if (hasattr(adjusted_cfg, 'layer1') and
                hasattr(adjusted_cfg.layer1, 'mlp_list') and
                len(adjusted_cfg.layer1.mlp_list) > 0):
                mlp_list = list(adjusted_cfg.layer1.mlp_list)
                mlp_list[0] = feature_input_dim  # RGB + optional mask
                adjusted_cfg.layer1.mlp_list = mlp_list

        elif backbone_name == 'ptv3':
            # For PTv3: adjust the in_channels parameter
            # PTv3 expects the total input dimension including xyz coordinates
            adjusted_cfg.in_channels = total_input_dim  # xyz + rgb + optional mask

        return adjusted_cfg

    def _build_model_layers(self, cfg):
        """Constructs the core layers of the network."""
        time_embed_dim = self.d_model * getattr(cfg, 'time_embed_mult', 4)
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input layer for fused features
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        )

        # Main processing blocks
        self.layers = nn.ModuleList()
        for _ in range(self.nblocks):
            self.layers.append(ResBlock(self.d_model, time_embed_dim, self.resblock_dropout, self.d_model))
            self.layers.append(
                SpatialTransformer(
                    self.d_model, self.transformer_num_heads, self.transformer_dim_head,
                    depth=self.transformer_depth, dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff, context_dim=self.d_model
                )
            )

        # Output layer
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=min(32, self.d_model), num_channels=self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, kernel_size=1),
        )

    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        Applies the model to a batch of noisy inputs.

        Args:
            x_t: The noisy input grasp pose, shape (B, C) or (B, num_grasps, C).
            ts: The batch of timesteps, shape (B,).
            data: A dictionary containing conditional information.

        Returns:
            The predicted noise or denoised target.
        """
        # Handle input dimensions
        if x_t.dim() == 2:
            return self._forward_single_grasp(x_t, ts, data)  # For backward compatibility
        elif x_t.dim() == 3:
            return self._forward_multi_grasp(x_t, ts, data)
        else:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}. Expected 2 or 3.")

    def _forward_single_grasp(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """Handles the single-grasp format for backward compatibility."""
        scene_cond = data["scene_cond"]
        text_cond = data.get("text_cond")
        x_t = x_t.unsqueeze(1)

        # Encode grasp and text
        grasp_embedding = self.grasp_encoder(x_t.squeeze(1))
        grasp_text_embedding = grasp_embedding + text_cond if text_cond is not None and self.use_text_condition else grasp_embedding

        # Construct context (scene + text)
        context_tokens = torch.cat([scene_cond, text_cond.unsqueeze(1)], dim=1) if text_cond is not None and self.use_text_condition else scene_cond

        # Prepare UNet input
        h = rearrange(grasp_text_embedding.unsqueeze(1), 'b l c -> b c l')
        t_emb = self.time_embed(timestep_embedding(ts, self.d_model))

        # U-Net Path
        h = self.in_layers(h)
        if self.use_position_embedding:
            B, C, L = h.shape
            pos_q = torch.arange(L, dtype=h.dtype, device=h.device)
            h = h + timestep_embedding(pos_q, C).permute(1, 0).unsqueeze(0)

        for i in range(self.nblocks):
            h = self.layers[i * 2](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=context_tokens)

        # Output
        h = self.out_layers(h)
        return h.rearrange('b c l -> b l c').squeeze(1)

    def _forward_multi_grasp(self, x_t: torch.Tensor, ts: torch.Tensor, data: Dict) -> torch.Tensor:
        """Handles the multi-grasp format."""
        scene_cond = data["scene_cond"]
        text_cond = data.get("text_cond")
        B, num_grasps, _ = x_t.shape

        # UNet processes whatever grasp count it receives from diffusion core

        # Encode grasp and text
        grasp_embedding = self.grasp_encoder(x_t)
        if text_cond is not None and self.use_text_condition:
            grasp_text_embedding = grasp_embedding + text_cond.unsqueeze(1).expand(-1, num_grasps, -1)
        else:
            grasp_text_embedding = grasp_embedding

        # Construct context (scene + text)
        context_tokens = torch.cat([scene_cond, text_cond.unsqueeze(1)], dim=1) if text_cond is not None and self.use_text_condition else scene_cond

        # Time embedding
        t_emb = self.time_embed(timestep_embedding(ts, self.d_model))

        # Process grasps in parallel
        return self._process_multi_grasp_unet(grasp_text_embedding, context_tokens, t_emb, B, num_grasps)

    def _process_multi_grasp_unet(self, grasp_features, context_tokens, t_emb, B, num_grasps):
        """UNet backbone for parallel processing of multiple grasps."""
        context_len = context_tokens.shape[1]

        # Reshape for batch processing;避免不必要的拷贝，尽量使用view/reshape
        h = grasp_features.reshape(B * num_grasps, self.d_model, 1)
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, num_grasps, -1).reshape(B * num_grasps, -1)
        context_expanded = context_tokens.unsqueeze(1).expand(-1, num_grasps, -1, -1).reshape(B * num_grasps, context_len, self.d_model)

        # UNet backbone
        h = self.in_layers(h)
        if self.use_position_embedding:
            _, C, L = h.shape
            pos_q = torch.arange(L, dtype=h.dtype, device=h.device)
            h = h + timestep_embedding(pos_q, C).permute(1, 0).unsqueeze(0)

        for i in range(self.nblocks):
            h = self.layers[i * 2](h, t_emb_expanded)
            h = self.layers[i * 2 + 1](h, context=context_expanded)

        # Output layer and reshape back
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c').squeeze(1)
        return h.view(B, num_grasps, -1)

    def condition(self, data: Dict) -> Dict:
        """
        Pre-computes and processes all conditional features (scene, text).

        Args:
            data: The raw data from the dataloader.

        Returns:
            A dictionary with processed "scene_cond" and "text_cond".
        """
        # 1. Scene Feature Extraction
        # Handle object mask inclusion in scene point cloud
        if self.use_object_mask and 'object_mask' in data:
            scene_pc = data['scene_pc'].to(torch.float32)
            object_mask = data['object_mask'].to(torch.float32).unsqueeze(-1)
            pos = torch.cat([scene_pc, object_mask], dim=-1)
        else:
            pos = data['scene_pc'].to(torch.float32)
        
        _, scene_feat = self.scene_model(pos)
        scene_feat = scene_feat.permute(0, 2, 1).contiguous()  # (B, N, C)

        condition_dict = {"scene_cond": scene_feat, "text_cond": None, "text_mask": None}
        if self.use_negative_prompts:
            condition_dict.update({"neg_pred": None, "neg_text_features": None})

        # 2. Text Feature Extraction (if enabled)
        if not (self.use_text_condition and 'positive_prompt' in data):
            return condition_dict

        try:
            self._ensure_text_encoder()
            b = scene_feat.shape[0]

            # Encode positive and negative prompts
            pos_text_features = self.text_encoder.encode_positive(data['positive_prompt'])
            neg_text_features = self.text_encoder.encode_negative(data['negative_prompts']) if self.use_negative_prompts and 'negative_prompts' in data else None

            # Apply text dropout during training
            text_mask = torch.bernoulli(torch.full((b, 1), 1.0 - self.text_dropout_prob, device=pos_text_features.device)) if self.training else torch.ones(b, 1, device=pos_text_features.device)
            
            # Process text features
            scene_embedding = torch.mean(scene_feat, dim=1)
            pos_text_features_out, neg_pred = self.text_processor(pos_text_features, neg_text_features, scene_embedding)

            condition_dict.update({"text_cond": pos_text_features_out * text_mask, "text_mask": text_mask})
            if self.use_negative_prompts:
                condition_dict.update({
                    "neg_pred": neg_pred,  # Predicted negative prompt for CFG
                    "neg_text_features": neg_text_features, # Original negative features for loss
                })

        except Exception as e:
            logging.warning(f"Text encoding failed: {e}. Falling back to scene-only conditioning.")
            condition_dict.update({k: None for k in condition_dict if k != "scene_cond"})

        return condition_dict

    # --- Device Management for Lazy-Loaded Encoder ---

    def _get_device(self):
        """Infers the model's device from its parameters."""
        return next(self.parameters()).device

    def _ensure_text_encoder(self):
        """Initializes the text encoder on the correct device if it doesn't exist."""
        if self.text_encoder is None:
            device = self._get_device()
            self.text_encoder = PosNegTextEncoder(device=device)
            self.text_encoder.to(device)
            logging.info(f"Text encoder lazily initialized on device: {device}")
        else:
            current_device = self._get_device()
            if self.text_encoder.device != current_device:
                self.text_encoder.to(current_device)
                logging.info(f"Text encoder moved to device: {current_device}")

    def to(self, *args, **kwargs):
        """Overrides `to()` to ensure the text encoder is also moved."""
        super().to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

