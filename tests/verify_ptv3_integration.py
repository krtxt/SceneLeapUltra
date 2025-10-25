#!/usr/bin/env python3
"""
Quick verification script for PTv3 integration.

This script checks if PTv3 backbone is properly integrated and all dependencies are available.
Run this before attempting to train with PTv3.
"""
import os
import sys

# Add project root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    deps = {
        'torch': 'PyTorch',
        'spconv': 'SparseConv (spconv)',
        'torch_scatter': 'PyTorch Scatter',
        'addict': 'Addict',
    }
    
    optional_deps = {
        'flash_attn': 'Flash Attention (optional, for ptv3/ptv3_light)',
    }
    
    all_required_ok = True
    
    # Check required dependencies
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_required_ok = False
    
    # Check optional dependencies
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Not installed (use ptv3_no_flash if needed)")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA Available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  ✗ CUDA Not Available - PTv3 requires GPU!")
            all_required_ok = False
    except:
        pass
    
    return all_required_ok


def check_config_files():
    """Check if all config files exist."""
    print("\n" + "=" * 60)
    print("Checking Configuration Files")
    print("=" * 60)
    
    config_files = [
        'config/model/diffuser/decoder/backbone/ptv3_light.yaml',
        'config/model/diffuser/decoder/backbone/ptv3.yaml',
        'config/model/diffuser/decoder/backbone/ptv3_no_flash.yaml',
    ]
    
    all_ok = True
    for cfg_path in config_files:
        full_path = os.path.join(repo_root, cfg_path)
        if os.path.exists(full_path):
            print(f"  ✓ {cfg_path}")
        else:
            print(f"  ✗ {cfg_path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_model_files():
    """Check if all model files exist."""
    print("\n" + "=" * 60)
    print("Checking Model Files")
    print("=" * 60)
    
    model_files = [
        'models/backbone/ptv3_backbone.py',
        'models/backbone/ptv3/ptv3.py',
        'models/backbone/ptv3/serialization/default.py',
    ]
    
    all_ok = True
    for model_path in model_files:
        full_path = os.path.join(repo_root, model_path)
        if os.path.exists(full_path):
            print(f"  ✓ {model_path}")
        else:
            print(f"  ✗ {model_path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_import():
    """Test if PTv3Backbone can be imported."""
    print("\n" + "=" * 60)
    print("Testing Imports")
    print("=" * 60)
    
    try:
        from models.backbone.ptv3_backbone import PTV3Backbone
        print(f"  ✓ PTV3Backbone imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import PTV3Backbone: {e}")
        return False


def test_basic_instantiation():
    """Test basic instantiation of PTv3Backbone."""
    print("\n" + "=" * 60)
    print("Testing Basic Instantiation")
    print("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠ Skipping instantiation test (no CUDA)")
            return True
        
        from omegaconf import OmegaConf

        from models.backbone.ptv3_backbone import PTV3Backbone
        
        cfg_path = os.path.join(repo_root, 'config/model/diffuser/decoder/backbone/ptv3_light.yaml')
        cfg = OmegaConf.load(cfg_path)
        
        model = PTV3Backbone(cfg)
        print(f"  ✓ PTv3Backbone instantiated (output_dim={model.output_dim})")
        
        del model
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"  ✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\nPTv3 Integration Verification")
    print("=" * 60)
    
    results = {
        'Dependencies': check_dependencies(),
        'Config Files': check_config_files(),
        'Model Files': check_model_files(),
        'Imports': test_import(),
        'Instantiation': test_basic_instantiation(),
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! PTv3 integration is ready to use.")
        print("\nNext steps:")
        print("  1. Run smoke tests: python tests/test_smoke_dit_ptv3.py")
        print("  2. Try training with PTv3:")
        print("     python train_lightning.py model/diffuser/decoder/backbone=ptv3_light")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Install missing dependencies: pip install spconv-cu118 torch-scatter addict")
        print("  - For Flash Attention: pip install flash-attn")
        print("  - Ensure CUDA is available and working")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

