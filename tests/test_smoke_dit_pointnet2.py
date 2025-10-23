"""
Smoke test for DiT with PointNet2 backbone.

This test verifies that DiT can be instantiated with PointNet2 backbone
after PTv3 has been removed from the codebase.
"""

import os
import sys

from omegaconf import OmegaConf


def test_dit_pointnet2_instantiation():
    """Test DiT + PointNet2 instantiation."""
    repo_root = "/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra"
    sys.path.insert(0, repo_root)

    # Lazy import after path setup
    from models.decoder.dit import DiTModel

    # Load PointNet2 backbone config
    pointnet2_cfg_path = os.path.join(
        repo_root, "config/model/diffuser/decoder/backbone/pointnet2.yaml"
    )
    backbone_cfg = OmegaConf.load(pointnet2_cfg_path)

    # Minimal DiT config for instantiation
    dit_cfg = OmegaConf.create(
        {
            "name": "dit",
            "rot_type": "quat",
            "d_model": 512,
            "num_layers": 2,
            "num_heads": 8,
            "d_head": 64,
            "dropout": 0.1,
            "max_sequence_length": 100,
            "use_learnable_pos_embedding": False,
            "time_embed_dim": 1024,
            "time_embed_mult": 4,
            "use_adaptive_norm": True,
            "use_text_condition": False,
            "text_dropout_prob": 0.01,
            "use_negative_prompts": True,
            "use_object_mask": False,
            "use_rgb": True,
            "attention_dropout": 0.1,
            "cross_attention_dropout": 0.1,
            "ff_mult": 4,
            "ff_dropout": 0.1,
            "gradient_checkpointing": False,
            "use_flash_attention": False,
            "attention_chunk_size": 64,
            "backbone": backbone_cfg,
        }
    )

    try:
        model = DiTModel(dit_cfg)
        print("[SMOKE] DiT + PointNet2 instantiation OK")
        print(f"[SMOKE] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print("[SMOKE] DiT + PointNet2 instantiation FAILED:", repr(e))
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dit_pointnet2_instantiation()
    sys.exit(0 if success else 1)

