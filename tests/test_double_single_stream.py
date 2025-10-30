"""
测试双流+单流 DiT 架构的形状正确性和数值稳定性
"""
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import logging
from omegaconf import OmegaConf

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_config(
    use_double_stream=True,
    num_double_blocks=4,
    num_layers=12,
    single_block_variant="parallel",
):
    """创建测试配置"""
    config = {
        # 基础配置
        'name': 'dit',
        'rot_type': 'quat',
        'd_model': 256,
        'num_layers': num_layers,
        'num_heads': 8,
        'd_head': 32,
        'dropout': 0.1,
        'max_sequence_length': 100,
        'use_learnable_pos_embedding': False,
        
        # 时间嵌入
        'time_embed_dim': 512,
        'time_embed_mult': 4,
        'use_adaptive_norm': False,
        
        # AdaLN-Zero配置（双流+单流需要）
        'use_adaln_zero': use_double_stream,
        'use_scene_pooling': True,
        'adaln_mode': 'multi',
        
        # 双流+单流配置
        'use_double_stream': use_double_stream,
        'num_double_blocks': num_double_blocks,
        'single_block_variant': single_block_variant,
        'mlp_ratio': 4.0,
        
        # 条件配置
        'use_text_condition': True,
        'text_dropout_prob': 0.0,
        'use_negative_prompts': False,
        'use_object_mask': False,
        'use_rgb': False,
        'use_text_tokens': False,
        
        # 注意力配置
        'attention_dropout': 0.0,
        'cross_attention_dropout': 0.0,
        'use_flash_attention': False,
        'attention_chunk_size': 512,
        'gradient_checkpointing': False,
        
        # FFN 配置
        'ff_mult': 4,
        'ff_dropout': 0.1,
        
        # 其他特性（暂时禁用）
        'use_geometric_bias': False,
        'use_global_local_conditioning': False,
        'use_t_aware_conditioning': False,
        
        # Backbone配置
        'backbone': {
            'name': 'pointnet2',
            'output_dim': 512,
        }
    }
    return OmegaConf.create(config)


def test_double_single_stream_shapes():
    """测试双流+单流架构的形状正确性"""
    logger.info("=" * 60)
    logger.info("测试1: 双流+单流架构形状验证")
    logger.info("=" * 60)
    
    # 创建模型
    cfg = create_test_config(
        use_double_stream=True,
        num_double_blocks=4,
        num_layers=12,
        single_block_variant="parallel"
    )
    
    from models.decoder.dit import DiTModel
    model = DiTModel(cfg)
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    num_grasps = 10
    num_points = 1024
    d_x = 23  # quat
    
    x_t = torch.randn(batch_size, num_grasps, d_x)
    ts = torch.randint(0, 1000, (batch_size,))
    
    # 场景数据
    scene_pc = torch.randn(batch_size, num_points, 3)
    scene_mask = torch.ones(batch_size, num_points)
    
    # 文本数据
    positive_prompt = ["grasp the object"] * batch_size
    
    data = {
        'scene_pc': scene_pc,
        'scene_mask': scene_mask,
        'positive_prompt': positive_prompt,
    }
    
    # 条件预处理
    logger.info("条件预处理...")
    with torch.no_grad():
        cond_data = model.condition(data)
        data.update(cond_data)
    
    # 前向传播
    logger.info(f"前向传播...")
    logger.info(f"  输入: x_t shape={tuple(x_t.shape)}, ts shape={tuple(ts.shape)}")
    
    with torch.no_grad():
        output = model(x_t, ts, data)
    
    logger.info(f"  输出: output shape={tuple(output.shape)}")
    
    # 验证形状
    assert output.shape == x_t.shape, f"输出形状不匹配: {output.shape} != {x_t.shape}"
    assert torch.isfinite(output).all(), "输出包含NaN或Inf"
    
    logger.info("✓ 形状验证通过")
    logger.info(f"✓ 数值稳定性检查通过 (min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f})")
    logger.info("")
    
    return True


def test_single_grasp_mode():
    """测试单个抓取模式（batch flatten）"""
    logger.info("=" * 60)
    logger.info("测试2: 单个抓取模式")
    logger.info("=" * 60)
    
    cfg = create_test_config(
        use_double_stream=True,
        num_double_blocks=4,
        num_layers=12,
        single_block_variant="parallel"
    )
    
    from models.decoder.dit import DiTModel
    model = DiTModel(cfg)
    model.eval()
    
    # 单个抓取输入
    batch_size = 2
    d_x = 23
    x_t = torch.randn(batch_size, d_x)  # (B, d_x)
    ts = torch.randint(0, 1000, (batch_size,))
    
    scene_pc = torch.randn(batch_size, 1024, 3)
    data = {'scene_pc': scene_pc, 'positive_prompt': ["test"] * batch_size}
    
    with torch.no_grad():
        cond_data = model.condition(data)
        data.update(cond_data)
        output = model(x_t, ts, data)
    
    assert output.shape == x_t.shape, f"单抓取模式输出形状不匹配: {output.shape} != {x_t.shape}"
    logger.info(f"✓ 单抓取模式验证通过: {tuple(output.shape)}")
    logger.info("")
    
    return True


def test_legacy_mode():
    """测试传统模式（不使用双流）"""
    logger.info("=" * 60)
    logger.info("测试3: 传统模式（不使用双流）")
    logger.info("=" * 60)
    
    cfg = create_test_config(
        use_double_stream=False,
        num_double_blocks=0,
        num_layers=12,
        single_block_variant="legacy"
    )
    
    from models.decoder.dit import DiTModel
    model = DiTModel(cfg)
    model.eval()
    
    batch_size = 2
    num_grasps = 10
    d_x = 23
    
    x_t = torch.randn(batch_size, num_grasps, d_x)
    ts = torch.randint(0, 1000, (batch_size,))
    scene_pc = torch.randn(batch_size, 1024, 3)
    data = {'scene_pc': scene_pc, 'positive_prompt': ["test"] * batch_size}
    
    with torch.no_grad():
        cond_data = model.condition(data)
        data.update(cond_data)
        output = model(x_t, ts, data)
    
    assert output.shape == x_t.shape
    logger.info(f"✓ 传统模式验证通过: {tuple(output.shape)}")
    logger.info("")
    
    return True


def test_different_configurations():
    """测试不同的双流+单流配置"""
    logger.info("=" * 60)
    logger.info("测试4: 不同配置验证")
    logger.info("=" * 60)
    
    configs = [
        (2, 6, "parallel"),   # 2个双流 + 4个并行单流
        (4, 8, "parallel"),   # 4个双流 + 4个并行单流
        (6, 12, "parallel"),  # 6个双流 + 6个并行单流
    ]
    
    for double_blocks, total_layers, variant in configs:
        logger.info(f"配置: {double_blocks} double + {total_layers - double_blocks} {variant} single...")
        
        cfg = create_test_config(
            use_double_stream=True,
            num_double_blocks=double_blocks,
            num_layers=total_layers,
            single_block_variant=variant
        )
        
        from models.decoder.dit import DiTModel
        model = DiTModel(cfg)
        model.eval()
        
        x_t = torch.randn(2, 10, 23)
        ts = torch.randint(0, 1000, (2,))
        data = {
            'scene_pc': torch.randn(2, 1024, 3),
            'positive_prompt': ["test", "test"]
        }
        
        with torch.no_grad():
            cond_data = model.condition(data)
            data.update(cond_data)
            output = model(x_t, ts, data)
        
        assert output.shape == x_t.shape
        assert torch.isfinite(output).all()
        logger.info(f"  ✓ 通过")
    
    logger.info("✓ 所有配置验证通过")
    logger.info("")
    
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("双流+单流 DiT 架构测试套件")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("双流+单流形状验证", test_double_single_stream_shapes),
        ("单抓取模式", test_single_grasp_mode),
        ("传统模式", test_legacy_mode),
        ("不同配置", test_different_configurations),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

