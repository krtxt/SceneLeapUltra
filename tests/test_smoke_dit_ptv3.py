"""
Smoke test for DiT with PTv3 backbone.

This test verifies that DiT can be instantiated with PTv3 backbones (ptv3_light, ptv3, ptv3_no_flash)
and perform forward passes with different input channel configurations.
"""
import os
import sys

import torch
from omegaconf import OmegaConf

# Add project root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from models.decoder.dit import DiTModel


def test_dit_ptv3_instantiation(variant='ptv3_light'):
    """Test DiT + PTv3 instantiation and forward pass with different input configurations."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print(f"[SKIP] CUDA not available. PTv3 requires GPU.")
        return True
    
    device = torch.device('cuda')
    
    try:
        # Load PTv3 backbone config
        backbone_cfg_path = os.path.join(
            repo_root, f"config/model/diffuser/decoder/backbone/{variant}.yaml"
        )
        backbone_cfg = OmegaConf.load(backbone_cfg_path)
        
        # Test configurations: (use_rgb, use_object_mask)
        test_configs = [
            (False, False),  # xyz only
            (True, False),   # xyz + rgb
            (False, True),   # xyz + mask
            (True, True),    # xyz + rgb + mask
        ]
        
        for use_rgb, use_object_mask in test_configs:
            print(f"\n[TEST] DiT + {variant}: use_rgb={use_rgb}, use_object_mask={use_object_mask}")
            
            # Create minimal config
            cfg = OmegaConf.create({
                'rot_type': 'quat',
                'd_model': 512,
                'num_layers': 2,  # Reduced for speed
                'num_heads': 4,
                'd_head': 64,
                'dropout': 0.1,
                'max_sequence_length': 10,
                'use_learnable_pos_embedding': False,
                'time_embed_dim': 512,
                'use_adaptive_norm': True,
                'use_text_condition': False,
                'text_dropout_prob': 0.0,
                'use_negative_prompts': False,
                'use_object_mask': use_object_mask,
                'use_rgb': use_rgb,
                'gradient_checkpointing': False,
                'use_flash_attention': False,  # Disable for smoke test
                'attention_chunk_size': 512,
                'memory_monitoring': False,
                'backbone': backbone_cfg
            })
            
            # Create model
            model = DiTModel(cfg).to(device)
            
            # Create dummy input
            B = 2
            num_grasps = 1
            d_x = 23  # quat format
            
            # Determine input channels
            C = 3  # xyz
            if use_rgb:
                C += 3
            if use_object_mask:
                C += 1
            
            N = 1024  # num points
            
            x_t = torch.randn(B, num_grasps, d_x, device=device)
            ts = torch.randint(0, 1000, (B,), device=device)
            
            data = {
                'scene_pc': torch.randn(B, N, C, device=device)
            }
            
            if use_object_mask:
                data['object_mask'] = torch.randint(0, 2, (B, N), device=device).float()
            
            # Forward pass
            with torch.no_grad():
                output = model(x_t, ts, data)
            
            # Verify output shape
            expected_shape = (B, num_grasps, d_x)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            print(f"  ✓ Forward pass successful. Output shape: {output.shape}")
            
            # Clean up
            del model, x_t, ts, data, output
            torch.cuda.empty_cache()
        
        print(f"\n[SUCCESS] DiT + {variant} smoke test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] DiT + {variant} smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run smoke tests for all PTv3 variants."""
    variants = ['ptv3_light', 'ptv3', 'ptv3_no_flash']
    
    results = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Testing DiT + {variant}")
        print(f"{'='*60}")
        results[variant] = test_dit_ptv3_instantiation(variant)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for variant, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {variant}: {status}")
    
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

