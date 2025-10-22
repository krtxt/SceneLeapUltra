import pytest
from omegaconf import OmegaConf

from models.decoder.dit_config_validation import (
    validate_dit_config,
    DiTConfigValidationError,
)


def _build_minimal_cfg():
    return OmegaConf.create({
        "name": "dit",
        "rot_type": "quat",
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "d_head": 16,
        "dropout": 0.1,
        "max_sequence_length": 32,
        "use_learnable_pos_embedding": True,
        "time_embed_dim": 32,
        "time_embed_mult": 2,
        "use_adaptive_norm": True,
        "use_text_condition": True,
        "text_dropout_prob": 0.2,
        "use_negative_prompts": True,
        "use_object_mask": False,
        "use_rgb": True,
        "attention_dropout": 0.1,
        "cross_attention_dropout": 0.1,
        "ff_mult": 4,
        "ff_dropout": 0.1,
        "gradient_checkpointing": False,
        "use_flash_attention": False,
        "backbone": {
            "name": "pointnet2",
            "layer1": {
                "npoint": 32,
                "radius_list": [0.1],
                "nsample_list": [16],
                "mlp_list": [4, 8],
            },
            "layer2": {
                "npoint": 16,
                "radius_list": [0.2],
                "nsample_list": [32],
                "mlp_list": [8, 16],
            },
            "layer3": {
                "npoint": 8,
                "radius_list": [0.4],
                "nsample_list": [64],
                "mlp_list": [16, 32],
            },
            "layer4": {
                "npoint": 4,
                "radius_list": [0.8],
                "nsample_list": [128],
                "mlp_list": [32, 64],
            },
        },
    })


def test_validate_dit_config_success():
    cfg = _build_minimal_cfg()
    assert validate_dit_config(cfg) is True


def test_validate_dit_config_missing_field_raises():
    cfg = _build_minimal_cfg()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("rot_type")
    broken_cfg = OmegaConf.create(cfg_dict)

    with pytest.raises(DiTConfigValidationError):
        validate_dit_config(broken_cfg)
