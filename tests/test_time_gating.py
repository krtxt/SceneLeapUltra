"""
测试时间门控模块的功能
"""

import torch
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.time_gating import (
    CosineSquaredGate, MLPGate, TimeGate, build_time_gate
)


def test_cosine_squared_gate():
    """测试余弦平方门控"""
    print("\n=== 测试 CosineSquaredGate ===")
    
    gate = CosineSquaredGate(scale=1.0)
    
    # 测试不同时间点的门控值
    test_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\n时间 t -> 门控因子 α(t):")
    for t_val in test_times:
        t = torch.tensor([t_val])
        alpha = gate(t)
        print(f"  t={t_val:.2f} -> α={alpha.item():.4f}")
    
    # 测试批次输入
    batch_size = 4
    t_batch = torch.linspace(0, 1, batch_size)
    alpha_batch = gate(t_batch)
    
    print(f"\n批次输入测试:")
    print(f"  输入形状: {t_batch.shape}")
    print(f"  输出形状: {alpha_batch.shape}")
    print(f"  输出范围: [{alpha_batch.min():.4f}, {alpha_batch.max():.4f}]")
    
    # 验证单调性（应该递减）
    assert alpha_batch[0] > alpha_batch[-1], "余弦平方门控应该单调递减"
    print("  ✓ 单调性检查通过")
    
    # 验证边界值
    assert torch.isclose(gate(torch.tensor([0.0])), torch.tensor([[[1.0]]])), "t=0 时应该输出 1.0"
    assert torch.isclose(gate(torch.tensor([1.0])), torch.tensor([[[0.0]]]), atol=1e-5), "t=1 时应该输出 0.0"
    print("  ✓ 边界值检查通过")
    
    print("\n✓ CosineSquaredGate 测试通过")


def test_mlp_gate():
    """测试可学习 MLP 门控"""
    print("\n=== 测试 MLPGate ===")
    
    input_dim = 512
    hidden_dims = [256, 128]
    init_value = 1.0
    warmup_steps = 100
    
    gate = MLPGate(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        init_value=init_value,
        warmup_steps=warmup_steps
    )
    
    # 测试训练模式下的 warmup
    gate.train()
    batch_size = 4
    time_emb = torch.randn(batch_size, input_dim)
    
    print(f"\n训练模式 (warmup 期间):")
    print(f"  global_step: {gate.global_step.item()}")
    alpha = gate(time_emb)
    print(f"  输出形状: {alpha.shape}")
    print(f"  输出值: {alpha.squeeze().tolist()}")
    
    # 在 warmup 期间，所有输出应该接近 init_value
    assert torch.allclose(alpha, torch.tensor([[[init_value]]]), atol=0.01), "Warmup 期间输出应该接近 init_value"
    print("  ✓ Warmup 检查通过")
    
    # 模拟训练若干步后
    for _ in range(warmup_steps + 10):
        gate.step()
    
    print(f"\n训练模式 (warmup 之后):")
    print(f"  global_step: {gate.global_step.item()}")
    alpha_after = gate(time_emb)
    print(f"  输出范围: [{alpha_after.min():.4f}, {alpha_after.max():.4f}]")
    
    # Warmup 后输出应该可以变化
    assert 0.0 <= alpha_after.min() <= 1.0, "输出应该在 [0, 1] 范围内"
    assert 0.0 <= alpha_after.max() <= 1.0, "输出应该在 [0, 1] 范围内"
    print("  ✓ 输出范围检查通过")
    
    # 测试评估模式
    gate.eval()
    alpha_eval = gate(time_emb)
    print(f"\n评估模式:")
    print(f"  输出范围: [{alpha_eval.min():.4f}, {alpha_eval.max():.4f}]")
    
    print("\n✓ MLPGate 测试通过")


def test_time_gate():
    """测试 TimeGate 统一接口"""
    print("\n=== 测试 TimeGate 统一接口 ===")
    
    # 测试 cos2 门控
    cos2_config = {
        'type': 'cos2',
        'apply_to': 'both',
        'scene_scale': 1.0,
        'text_scale': 0.8,
        'separate_text_gate': False
    }
    
    time_gate = TimeGate(cos2_config, time_embed_dim=512)
    
    print(f"\n配置: {cos2_config}")
    
    # 测试场景门控
    t = torch.tensor([0.5])
    alpha_scene = time_gate.get_scene_gate(t=t)
    print(f"  场景门控 (t=0.5): {alpha_scene.item():.4f}")
    assert alpha_scene is not None, "场景门控不应该为 None"
    
    # 测试文本门控
    alpha_text = time_gate.get_text_gate(t=t)
    print(f"  文本门控 (t=0.5): {alpha_text.item():.4f}")
    assert alpha_text is not None, "文本门控不应该为 None"
    
    # 验证缩放效果
    assert alpha_text < alpha_scene, "文本缩放因子较小，门控值应该更小"
    print("  ✓ 缩放效果检查通过")
    
    # 测试只应用于场景
    scene_only_config = {
        'type': 'cos2',
        'apply_to': 'scene',
        'scene_scale': 1.0,
        'text_scale': 1.0
    }
    
    time_gate_scene_only = TimeGate(scene_only_config)
    alpha_scene_only = time_gate_scene_only.get_scene_gate(t=t)
    alpha_text_none = time_gate_scene_only.get_text_gate(t=t)
    
    print(f"\n只应用于场景:")
    print(f"  场景门控: {alpha_scene_only.item():.4f}")
    print(f"  文本门控: {alpha_text_none}")
    assert alpha_scene_only is not None, "场景门控应该存在"
    assert alpha_text_none is None, "文本门控应该为 None"
    print("  ✓ 选择性应用检查通过")
    
    print("\n✓ TimeGate 测试通过")


def test_build_time_gate():
    """测试工厂函数"""
    print("\n=== 测试 build_time_gate 工厂函数 ===")
    
    # 测试关闭时间门控
    gate_disabled = build_time_gate(
        use_t_aware_conditioning=False,
        gate_config=None,
        time_embed_dim=512
    )
    print(f"\n关闭时间门控: {gate_disabled}")
    assert gate_disabled is None, "关闭时应该返回 None"
    print("  ✓ 关闭检查通过")
    
    # 测试启用默认配置
    gate_default = build_time_gate(
        use_t_aware_conditioning=True,
        gate_config=None,
        time_embed_dim=512
    )
    print(f"\n启用默认配置:")
    print(f"  门控类型: {gate_default.gate_type}")
    print(f"  应用范围: {gate_default.apply_to}")
    assert gate_default is not None, "启用时应该返回 TimeGate 实例"
    assert gate_default.gate_type == 'cos2', "默认应该使用 cos2 门控"
    print("  ✓ 默认配置检查通过")
    
    # 测试自定义配置
    custom_config = {
        'type': 'mlp',
        'apply_to': 'scene',
        'scene_scale': 0.5,
        'mlp_hidden_dims': [128, 64],
        'init_value': 0.8,
        'warmup_steps': 500
    }
    
    gate_custom = build_time_gate(
        use_t_aware_conditioning=True,
        gate_config=custom_config,
        time_embed_dim=512
    )
    print(f"\n自定义配置:")
    print(f"  门控类型: {gate_custom.gate_type}")
    print(f"  场景缩放: {gate_custom.scene_scale}")
    assert gate_custom.gate_type == 'mlp', "应该使用 MLP 门控"
    assert gate_custom.scene_scale == 0.5, "应该使用自定义缩放"
    print("  ✓ 自定义配置检查通过")
    
    print("\n✓ build_time_gate 测试通过")


def test_integration():
    """集成测试：模拟在 DiTBlock 中的使用"""
    print("\n=== 集成测试 ===")
    
    # 模拟参数
    batch_size = 2
    num_grasps = 10
    d_model = 512
    time_embed_dim = 1024
    
    # 创建时间门控
    gate_config = {
        'type': 'cos2',
        'apply_to': 'both',
        'scene_scale': 1.0,
        'text_scale': 1.0
    }
    
    time_gate = build_time_gate(
        use_t_aware_conditioning=True,
        gate_config=gate_config,
        time_embed_dim=time_embed_dim
    )
    
    # 模拟输入
    t_scalar = torch.rand(batch_size)  # [0, 1] 范围的时间
    time_emb = torch.randn(batch_size, time_embed_dim)
    scene_attn_out = torch.randn(batch_size, num_grasps, d_model)
    text_attn_out = torch.randn(batch_size, num_grasps, d_model)
    
    print(f"\n输入:")
    print(f"  t_scalar: {t_scalar}")
    print(f"  scene_attn_out 形状: {scene_attn_out.shape}")
    print(f"  text_attn_out 形状: {text_attn_out.shape}")
    
    # 应用场景门控
    alpha_scene = time_gate.get_scene_gate(t=t_scalar, time_emb=time_emb)
    scene_attn_gated = scene_attn_out * alpha_scene
    
    print(f"\n场景门控:")
    print(f"  alpha_scene 形状: {alpha_scene.shape}")
    print(f"  alpha_scene 值: {alpha_scene.squeeze().tolist()}")
    print(f"  门控后输出形状: {scene_attn_gated.shape}")
    
    # 应用文本门控
    alpha_text = time_gate.get_text_gate(t=t_scalar, time_emb=time_emb)
    text_attn_gated = text_attn_out * alpha_text
    
    print(f"\n文本门控:")
    print(f"  alpha_text 形状: {alpha_text.shape}")
    print(f"  alpha_text 值: {alpha_text.squeeze().tolist()}")
    print(f"  门控后输出形状: {text_attn_gated.shape}")
    
    # 验证形状
    assert scene_attn_gated.shape == scene_attn_out.shape, "门控后形状应该不变"
    assert text_attn_gated.shape == text_attn_out.shape, "门控后形状应该不变"
    print("\n  ✓ 形状检查通过")
    
    # 验证门控效果
    assert torch.all(torch.abs(scene_attn_gated) <= torch.abs(scene_attn_out) + 1e-5), "门控后值应该减小或不变"
    print("  ✓ 门控效果检查通过")
    
    print("\n✓ 集成测试通过")


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试时间门控模块")
    print("=" * 60)
    
    try:
        test_cosine_squared_gate()
        test_mlp_gate()
        test_time_gate()
        test_build_time_gate()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

