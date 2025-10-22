#!/usr/bin/env python3
"""
Test script for Flow Matching implementation.

This script tests the basic functionality of the Flow Matching model
without requiring full training infrastructure.
"""

import torch
import logging
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)

def test_flow_matching_utils():
    """Test Flow Matching utilities."""
    print("\n" + "="*80)
    print("Testing Flow Matching Utilities")
    print("="*80)

    from models.utils.flow_matching_utils import OptimalTransportFlow, EulerSampler, HeunSampler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test parameters
    B, num_grasps, D = 4, 3, 25

    # Test OT Flow
    print("\n1. Testing Optimal Transport Flow...")
    ot_flow = OptimalTransportFlow(sigma_min=1e-4)

    # Sample data and time
    x1 = torch.randn(B, num_grasps, D, device=device)
    t = ot_flow.sample_time(B, device=device)
    print(f"   - Sampled timesteps: {t}")

    # Sample conditional flow
    x_t, u_t = ot_flow.sample_conditional_flow(x1, t)
    print(f"   - x_t shape: {x_t.shape}")
    print(f"   - u_t (target velocity) shape: {u_t.shape}")

    # Test loss
    pred_velocity = torch.randn_like(u_t)
    loss = ot_flow.compute_loss(pred_velocity, u_t)
    print(f"   - Flow matching loss: {loss.item():.6f}")

    # Test single grasp format
    x1_single = torch.randn(B, D, device=device)
    t_single = ot_flow.sample_time(B, device=device)
    x_t_single, u_t_single = ot_flow.sample_conditional_flow(x1_single, t_single)
    print(f"   - Single grasp x_t shape: {x_t_single.shape}")
    print(f"   - Single grasp u_t shape: {u_t_single.shape}")

    print("   ✓ Optimal Transport Flow tests passed!")

    return True


def test_flow_matching_model():
    """Test Flow Matching model basic instantiation and forward pass."""
    print("\n" + "="*80)
    print("Testing Flow Matching Core Components")
    print("="*80)

    print("\n1. Testing FlowMatchingCoreMixin initialization...")
    try:
        from models.utils.flow_matching_core import FlowMatchingCoreMixin

        # Create a mock class to test the mixin
        class MockFlowModel(FlowMatchingCoreMixin):
            def __init__(self):
                self.sigma_min = 1e-4
                self.sampler_type = 'heun'
                self.num_sampling_steps = 50
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self._init_flow_matching()

        model = MockFlowModel()
        print(f"   ✓ FlowMatchingCoreMixin initialized successfully!")
        print(f"   - Sampler type: {model.sampler_type}")
        print(f"   - Sampling steps: {model.num_sampling_steps}")
        print(f"   - Sigma min: {model.sigma_min}")
    except Exception as e:
        print(f"   ✗ FlowMatchingCoreMixin initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n2. Testing time sampling...")
    try:
        B = 8
        t = model.sample_time(B)
        print(f"   ✓ Time sampling successful!")
        print(f"   - Timesteps shape: {t.shape}")
        print(f"   - Timesteps range: [{t.min():.4f}, {t.max():.4f}]")
    except Exception as e:
        print(f"   ✗ Time sampling failed: {e}")
        return False

    print("\n3. Testing conditional flow sampling...")
    try:
        device = model.device
        B, num_grasps, D = 4, 3, 25
        x1 = torch.randn(B, num_grasps, D, device=device)
        t = model.sample_time(B)

        x_t, u_t = model.sample_conditional_flow(x1, t)
        print(f"   ✓ Conditional flow sampling successful!")
        print(f"   - x_t shape: {x_t.shape}")
        print(f"   - u_t shape: {u_t.shape}")
    except Exception as e:
        print(f"   ✗ Conditional flow sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n4. Testing velocity loss computation...")
    try:
        pred_velocity = torch.randn_like(u_t)
        loss = model.compute_velocity_loss(pred_velocity, u_t)
        print(f"   ✓ Velocity loss computation successful!")
        print(f"   - Loss value: {loss.item():.6f}")
    except Exception as e:
        print(f"   ✗ Velocity loss computation failed: {e}")
        return False

    print("\n   ✓ All Flow Matching core component tests passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Flow Matching Implementation Test Suite")
    print("="*80)

    tests_passed = 0
    total_tests = 2

    # Test 1: Utilities
    try:
        if test_flow_matching_utils():
            tests_passed += 1
    except Exception as e:
        print(f"\n✗ Utilities test failed with exception: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Model
    try:
        if test_flow_matching_model():
            tests_passed += 1
    except Exception as e:
        print(f"\n✗ Model test failed with exception: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print(f"Test Summary: {tests_passed}/{total_tests} test suites passed")
    print("="*80)

    if tests_passed == total_tests:
        print("\n✓ All tests passed! Flow Matching implementation is ready to use.")
        print("\nQuick start:")
        print("  1. Train with Flow Matching:")
        print("     ./train_distributed.sh model=flow_matching/flow_matching_dit")
        print("\n  2. Or use UNet decoder:")
        print("     ./train_distributed.sh model=flow_matching/flow_matching_unet")
        return 0
    else:
        print(f"\n✗ {total_tests - tests_passed} test suite(s) failed.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
