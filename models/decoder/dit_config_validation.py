"""
Configuration validation utilities for DiT (Diffusion Transformer) model.

This module provides validation functions to ensure DiT configuration parameters
are properly set and compatible with the model requirements.
"""

import logging
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig


class DiTConfigValidationError(Exception):
    """Custom exception for DiT configuration validation errors."""
    pass


def validate_dit_config(cfg: DictConfig) -> bool:
    """
    Validate DiT model configuration parameters.
    
    Args:
        cfg: DiT configuration object
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        DiTConfigValidationError: If configuration is invalid
    """
    try:
        # Required parameters
        required_params = [
            'name', 'rot_type', 'd_model', 'num_layers', 'num_heads', 'd_head',
            'dropout', 'max_sequence_length', 'use_learnable_pos_embedding',
            'time_embed_dim', 'time_embed_mult', 'use_adaptive_norm',
            'use_text_condition', 'text_dropout_prob', 'use_negative_prompts',
            'use_object_mask', 'use_rgb', 'attention_dropout', 'cross_attention_dropout',
            'ff_mult', 'ff_dropout', 'gradient_checkpointing', 'use_flash_attention'
        ]
        
        # Check required parameters exist
        missing_params = []
        for param in required_params:
            if not hasattr(cfg, param):
                missing_params.append(param)
        
        if missing_params:
            raise DiTConfigValidationError(
                f"Missing required DiT configuration parameters: {missing_params}"
            )
        
        # Validate model name
        if cfg.name.lower() != 'dit':
            raise DiTConfigValidationError(
                f"Invalid model name '{cfg.name}'. Expected 'dit'"
            )
        
        # Validate rotation type
        valid_rot_types = ['quat', 'r6d']
        if cfg.rot_type not in valid_rot_types:
            raise DiTConfigValidationError(
                f"Invalid rot_type '{cfg.rot_type}'. Must be one of {valid_rot_types}"
            )
        
        # Validate architecture parameters
        if cfg.d_model <= 0 or cfg.d_model % cfg.num_heads != 0:
            raise DiTConfigValidationError(
                f"d_model ({cfg.d_model}) must be positive and divisible by num_heads ({cfg.num_heads})"
            )
        
        if cfg.num_layers <= 0:
            raise DiTConfigValidationError(
                f"num_layers ({cfg.num_layers}) must be positive"
            )
        
        if cfg.num_heads <= 0:
            raise DiTConfigValidationError(
                f"num_heads ({cfg.num_heads}) must be positive"
            )
        
        if cfg.d_head <= 0:
            raise DiTConfigValidationError(
                f"d_head ({cfg.d_head}) must be positive"
            )
        
        # Validate dropout parameters
        dropout_params = [
            ('dropout', cfg.dropout),
            ('attention_dropout', cfg.attention_dropout),
            ('cross_attention_dropout', cfg.cross_attention_dropout),
            ('ff_dropout', cfg.ff_dropout),
            ('text_dropout_prob', cfg.text_dropout_prob)
        ]
        
        for param_name, param_value in dropout_params:
            if not (0.0 <= param_value <= 1.0):
                raise DiTConfigValidationError(
                    f"{param_name} ({param_value}) must be between 0.0 and 1.0"
                )
        
        # Validate sequence length
        if cfg.max_sequence_length <= 0:
            raise DiTConfigValidationError(
                f"max_sequence_length ({cfg.max_sequence_length}) must be positive"
            )
        
        # Validate timestep embedding parameters
        if cfg.time_embed_dim <= 0:
            raise DiTConfigValidationError(
                f"time_embed_dim ({cfg.time_embed_dim}) must be positive"
            )
        
        if cfg.time_embed_mult <= 0:
            raise DiTConfigValidationError(
                f"time_embed_mult ({cfg.time_embed_mult}) must be positive"
            )
        
        # Validate feed-forward multiplier
        if cfg.ff_mult <= 0:
            raise DiTConfigValidationError(
                f"ff_mult ({cfg.ff_mult}) must be positive"
            )
        
        # Validate boolean parameters
        boolean_params = [
            'use_learnable_pos_embedding', 'use_adaptive_norm', 'use_text_condition',
            'use_negative_prompts', 'use_object_mask', 'use_rgb', 'gradient_checkpointing',
            'use_flash_attention'
        ]
        
        for param in boolean_params:
            if not isinstance(getattr(cfg, param), bool):
                raise DiTConfigValidationError(
                    f"{param} must be a boolean value"
                )
        
        # Validate backbone configuration if present
        if hasattr(cfg, 'backbone'):
            _validate_backbone_config(cfg.backbone)
        
        logging.info("DiT configuration validation passed")
        return True
        
    except DiTConfigValidationError:
        raise
    except Exception as e:
        raise DiTConfigValidationError(f"Unexpected error during DiT config validation: {str(e)}")


def _validate_backbone_config(backbone_cfg: DictConfig) -> None:
    """
    Validate backbone configuration for DiT model.
    
    Args:
        backbone_cfg: Backbone configuration object
        
    Raises:
        DiTConfigValidationError: If backbone configuration is invalid
    """
    if not hasattr(backbone_cfg, 'name'):
        raise DiTConfigValidationError("Backbone configuration missing 'name' parameter")
    
    # Currently only PointNet2 is supported
    if backbone_cfg.name.lower() != 'pointnet2':
        raise DiTConfigValidationError(
            f"Unsupported backbone '{backbone_cfg.name}'. Currently only 'pointnet2' is supported"
        )
    
    # Validate PointNet2 specific parameters
    required_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in required_layers:
        if not hasattr(backbone_cfg, layer_name):
            raise DiTConfigValidationError(
                f"PointNet2 backbone missing required layer: {layer_name}"
            )
        
        layer_cfg = getattr(backbone_cfg, layer_name)
        
        # Check required layer parameters
        required_layer_params = ['npoint', 'radius_list', 'nsample_list', 'mlp_list']
        for param in required_layer_params:
            if not hasattr(layer_cfg, param):
                raise DiTConfigValidationError(
                    f"PointNet2 {layer_name} missing required parameter: {param}"
                )
        
        # Validate parameter values
        if layer_cfg.npoint <= 0:
            raise DiTConfigValidationError(
                f"PointNet2 {layer_name} npoint ({layer_cfg.npoint}) must be positive"
            )
        
        # Handle both list and single value formats for radius_list
        radius_list = layer_cfg.radius_list
        if not isinstance(radius_list, (list, tuple)):
            radius_list = [radius_list]
        if len(radius_list) == 0:
            raise DiTConfigValidationError(
                f"PointNet2 {layer_name} radius_list must be non-empty"
            )
        
        # Handle both list and single value formats for nsample_list
        nsample_list = layer_cfg.nsample_list
        if not isinstance(nsample_list, (list, tuple)):
            nsample_list = [nsample_list]
        if len(nsample_list) == 0:
            raise DiTConfigValidationError(
                f"PointNet2 {layer_name} nsample_list must be non-empty"
            )
        
        # Handle both list and single value formats for mlp_list
        mlp_list = layer_cfg.mlp_list
        if not isinstance(mlp_list, (list, tuple)):
            mlp_list = [mlp_list]
        if len(mlp_list) == 0:
            raise DiTConfigValidationError(
                f"PointNet2 {layer_name} mlp_list must be non-empty"
            )


def validate_dit_compatibility_with_diffuser(dit_cfg: DictConfig, diffuser_cfg: DictConfig) -> bool:
    """
    Validate DiT configuration compatibility with diffuser configuration.
    
    Args:
        dit_cfg: DiT configuration object
        diffuser_cfg: Diffuser configuration object
        
    Returns:
        bool: True if configurations are compatible
        
    Raises:
        DiTConfigValidationError: If configurations are incompatible
    """
    # Check rot_type consistency
    if dit_cfg.rot_type != diffuser_cfg.rot_type:
        raise DiTConfigValidationError(
            f"DiT rot_type ({dit_cfg.rot_type}) doesn't match diffuser rot_type ({diffuser_cfg.rot_type})"
        )
    
    # Check negative prompts consistency
    if hasattr(diffuser_cfg, 'use_negative_prompts'):
        if dit_cfg.use_negative_prompts != diffuser_cfg.use_negative_prompts:
            logging.warning(
                f"DiT use_negative_prompts ({dit_cfg.use_negative_prompts}) differs from "
                f"diffuser use_negative_prompts ({diffuser_cfg.use_negative_prompts}). "
                f"Using DiT setting."
            )
    
    # Check object mask consistency
    if hasattr(diffuser_cfg, 'use_object_mask'):
        if dit_cfg.use_object_mask != diffuser_cfg.use_object_mask:
            logging.warning(
                f"DiT use_object_mask ({dit_cfg.use_object_mask}) differs from "
                f"diffuser use_object_mask ({diffuser_cfg.use_object_mask}). "
                f"Using DiT setting."
            )
    
    # Check use_rgb consistency
    if hasattr(diffuser_cfg, 'use_rgb'):
        if dit_cfg.use_rgb != diffuser_cfg.use_rgb:
            logging.warning(
                f"DiT use_rgb ({dit_cfg.use_rgb}) differs from "
                f"diffuser use_rgb ({diffuser_cfg.use_rgb}). "
                f"Using DiT setting."
            )
    
    logging.info("DiT-Diffuser compatibility validation passed")
    return True


def get_dit_config_summary(cfg: DictConfig) -> Dict[str, Any]:
    """
    Get a summary of DiT configuration for logging/debugging.
    
    Args:
        cfg: DiT configuration object
        
    Returns:
        Dict containing key configuration parameters
    """
    return {
        'model_name': cfg.name,
        'rot_type': cfg.rot_type,
        'architecture': {
            'd_model': cfg.d_model,
            'num_layers': cfg.num_layers,
            'num_heads': cfg.num_heads,
            'd_head': cfg.d_head,
            'max_sequence_length': cfg.max_sequence_length
        },
        'conditioning': {
            'use_text_condition': cfg.use_text_condition,
            'use_negative_prompts': cfg.use_negative_prompts,
            'use_object_mask': cfg.use_object_mask,
            'use_rgb': cfg.use_rgb,
            'text_dropout_prob': cfg.text_dropout_prob
        },
        'optimization': {
            'dropout': cfg.dropout,
            'attention_dropout': cfg.attention_dropout,
            'gradient_checkpointing': cfg.gradient_checkpointing,
            'use_flash_attention': cfg.use_flash_attention
        }
    }