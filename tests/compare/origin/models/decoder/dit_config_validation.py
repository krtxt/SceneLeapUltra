"""
Configuration validation utilities for DiT (Diffusion Transformer) model.

This module provides validation functions to ensure DiT configuration parameters
are properly set and compatible with the model requirements.
"""

import logging
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, ListConfig


class DiTConfigValidationError(Exception):
    """Custom exception for DiT configuration validation errors."""
    pass


REQUIRED_PARAMS = [
    'name', 'rot_type', 'd_model', 'num_layers', 'num_heads', 'd_head',
    'dropout', 'max_sequence_length', 'use_learnable_pos_embedding',
    'time_embed_dim', 'time_embed_mult', 'use_adaptive_norm',
    'use_text_condition', 'text_dropout_prob', 'use_negative_prompts',
    'use_object_mask', 'use_rgb', 'attention_dropout', 'cross_attention_dropout',
    'ff_mult', 'ff_dropout', 'gradient_checkpointing', 'use_flash_attention'
]

VALID_ROT_TYPES = {'quat', 'r6d'}

DROPOUT_PARAMS = [
    'dropout', 'attention_dropout', 'cross_attention_dropout',
    'ff_dropout', 'text_dropout_prob'
]

BOOLEAN_PARAMS = [
    'use_learnable_pos_embedding', 'use_adaptive_norm', 'use_text_condition',
    'use_negative_prompts', 'use_object_mask', 'use_rgb',
    'gradient_checkpointing', 'use_flash_attention'
]


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
        _require_attributes(cfg, REQUIRED_PARAMS)
        _ensure_condition(cfg.name.lower() == 'dit', f"Invalid model name '{cfg.name}'. Expected 'dit'")
        _ensure_in_set(cfg.rot_type, VALID_ROT_TYPES, "rot_type")

        _ensure_positive(cfg, ['d_model', 'num_layers', 'num_heads', 'd_head',
                               'max_sequence_length', 'time_embed_dim', 'time_embed_mult', 'ff_mult'])
        _ensure_divisible(cfg.d_model, cfg.num_heads, "d_model", "num_heads")

        _ensure_in_range(cfg, DROPOUT_PARAMS, 0.0, 1.0)
        _ensure_booleans(cfg, BOOLEAN_PARAMS)

        if hasattr(cfg, 'backbone'):
            _validate_backbone_config(cfg.backbone)

        logging.info("DiT configuration validation passed")
        return True

    except DiTConfigValidationError:
        raise
    except Exception as e:
        raise DiTConfigValidationError(f"Unexpected error during DiT config validation: {str(e)}") from e


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

    backbone_name = backbone_cfg.name.lower()
    validators = {
        'pointnet2': _validate_pointnet2_backbone,
        'ptv3': _validate_ptv3_backbone,
    }

    if backbone_name not in validators:
        raise DiTConfigValidationError(
            f"Unsupported backbone '{backbone_cfg.name}'. Supported backbones: 'pointnet2', 'ptv3'"
        )

    validators[backbone_name](backbone_cfg)


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


def _require_attributes(cfg: Any, attributes: List[str], prefix: str = "DiT configuration") -> None:
    missing = [attr for attr in attributes if not hasattr(cfg, attr)]
    if missing:
        raise DiTConfigValidationError(f"{prefix} missing required parameters: {missing}")


def _ensure_condition(condition: bool, message: str) -> None:
    if not condition:
        raise DiTConfigValidationError(message)


def _ensure_in_set(value: Any, valid_values: set, field_name: str) -> None:
    if value not in valid_values:
        raise DiTConfigValidationError(
            f"Invalid {field_name} '{value}'. Must be one of {sorted(valid_values)}"
        )


def _ensure_positive(cfg: Any, fields: List[str]) -> None:
    for field in fields:
        value = getattr(cfg, field)
        if value <= 0:
            raise DiTConfigValidationError(f"{field} ({value}) must be positive")


def _ensure_divisible(value: int, divisor: int, value_name: str, divisor_name: str) -> None:
    if value <= 0 or divisor <= 0 or value % divisor != 0:
        raise DiTConfigValidationError(
            f"{value_name} ({value}) must be positive and divisible by {divisor_name} ({divisor})"
        )


def _ensure_in_range(cfg: Any, fields: List[str], min_value: float, max_value: float) -> None:
    for field in fields:
        value = getattr(cfg, field)
        if not (min_value <= value <= max_value):
            raise DiTConfigValidationError(
                f"{field} ({value}) must be between {min_value} and {max_value}"
            )


def _ensure_booleans(cfg: Any, fields: List[str]) -> None:
    for field in fields:
        value = getattr(cfg, field)
        if not isinstance(value, bool):
            raise DiTConfigValidationError(f"{field} must be a boolean value")


def _validate_pointnet2_backbone(backbone_cfg: DictConfig) -> None:
    required_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    _require_attributes(backbone_cfg, required_layers, prefix="PointNet2 backbone")

    required_layer_params = ['npoint', 'radius_list', 'nsample_list', 'mlp_list']

    for layer_name in required_layers:
        layer_cfg = getattr(backbone_cfg, layer_name)
        layer_prefix = f"PointNet2 {layer_name}"
        _require_attributes(layer_cfg, required_layer_params, prefix=layer_prefix)

        if layer_cfg.npoint <= 0:
            raise DiTConfigValidationError(f"{layer_prefix} npoint ({layer_cfg.npoint}) must be positive")

        for param in ['radius_list', 'nsample_list', 'mlp_list']:
            values = _coerce_to_sequence(getattr(layer_cfg, param), f"{layer_prefix} {param}")
            if len(values) == 0:
                raise DiTConfigValidationError(f"{layer_prefix} {param} must be non-empty")


def _validate_ptv3_backbone(backbone_cfg: DictConfig) -> None:
    _ensure_optional_positive_int(backbone_cfg, 'in_channels', "PTv3 in_channels")
    _ensure_optional_positive_number(backbone_cfg, 'grid_size', "PTv3 grid_size")
    _ensure_optional_positive_int(backbone_cfg, 'max_context_points', "PTv3 max_context_points")

    sequence_fields = [
        'enc_depths', 'enc_channels', 'enc_num_head', 'enc_patch_size',
        'dec_depths', 'dec_channels', 'dec_num_head', 'dec_patch_size'
    ]
    for field in sequence_fields:
        if hasattr(backbone_cfg, field):
            values = _coerce_to_sequence(getattr(backbone_cfg, field), f"PTv3 {field}", allow_scalar=False)
            if len(values) == 0:
                raise DiTConfigValidationError(f"PTv3 {field} must be non-empty if provided")


def _ensure_optional_positive_int(cfg: Any, field: str, label: str) -> None:
    if hasattr(cfg, field):
        value = getattr(cfg, field)
        if not isinstance(value, int) or value <= 0:
            raise DiTConfigValidationError(f"{label} ({value}) must be a positive integer if provided")


def _ensure_optional_positive_number(cfg: Any, field: str, label: str) -> None:
    if hasattr(cfg, field):
        value = getattr(cfg, field)
        if not isinstance(value, (int, float)) or value <= 0:
            raise DiTConfigValidationError(f"{label} ({value}) must be a positive number if provided")


def _coerce_to_sequence(value: Any, field_name: str, allow_scalar: bool = True) -> List[Any]:
    if isinstance(value, (list, tuple, ListConfig)):
        sequence = list(value)
    elif allow_scalar:
        sequence = [value]
    else:
        raise DiTConfigValidationError(f"{field_name} must be a non-empty sequence if provided")

    if any(isinstance(item, str) for item in sequence):
        raise DiTConfigValidationError(f"{field_name} must not contain string elements")
    return sequence
