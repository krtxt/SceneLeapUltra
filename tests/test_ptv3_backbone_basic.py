"""
Basic test for PTv3 Backbone wrapper.

This test verifies that the PTv3Backbone wrapper can be instantiated
and perform basic forward passes.
"""
import os
import sys

import torch
from omegaconf import OmegaConf

# Add project root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)


def test_ptv3_backbone_basic():
    """Test basic PTv3Backbone instantiation and forward."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available. PTv3 requires GPU.")
        return True
    
    device = torch.device('cuda')
    
    try:
        from models.backbone.ptv3_backbone import PTV3Backbone

        # Load config
        cfg_path = os.path.join(
            repo_root, "config/model/diffuser/decoder/backbone/ptv3_light.yaml"
        )
        cfg = OmegaConf.load(cfg_path)
        
        print(f"[TEST] Creating PTv3Backbone with config: {cfg}")
        
        # Create backbone
        backbone = PTV3Backbone(cfg).to(device)
        
        print(f"  ✓ PTv3Backbone instantiated. Output dim: {backbone.output_dim}")
        
        # Test with different input configurations
        test_cases = [
            (3, "xyz only"),
            (6, "xyz + rgb"),
            (4, "xyz + mask"),
            (7, "xyz + rgb + mask"),
        ]
        
        for C, desc in test_cases:
            print(f"\n[TEST] Forward pass with {desc} (C={C})")
            
            B = 2
            N = 1024
            
            # Create dummy input
            pos = torch.randn(B, N, C, device=device)
            
            # Forward pass
            with torch.no_grad():
                xyz, feat = backbone(pos)
            
            print(f"  Input shape: {pos.shape}")
            print(f"  Output xyz shape: {xyz.shape}")
            print(f"  Output feat shape: {feat.shape}")
            print(f"  ✓ Forward pass successful")
            
            # Verify output shapes
            assert xyz.dim() == 3, f"xyz should be 3D, got {xyz.dim()}D"
            assert feat.dim() == 3, f"feat should be 3D, got {feat.dim()}D"
            assert xyz.shape[0] == B, f"Batch size mismatch"
            assert feat.shape[0] == B, f"Batch size mismatch"
            assert feat.shape[1] == backbone.output_dim, \
                f"Feature dim should be {backbone.output_dim}, got {feat.shape[1]}"
            
            del pos, xyz, feat
            torch.cuda.empty_cache()
        
        print(f"\n[SUCCESS] PTv3Backbone basic test passed!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] PTv3Backbone basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ptv3_backbone_basic()
    sys.exit(0 if success else 1)

