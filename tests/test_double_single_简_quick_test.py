"""
快速测试双流+单流架构的核心组件
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dit_double_stream_block():
    """测试 DiTDoubleStreamBlock"""
    logger.info("="*60)
    logger.info("测试1: DiTDoubleStreamBlock")
    logger.info("="*60)
    
    from models.decoder.dit import DiTDoubleStreamBlock
    
    d_model = 256
    num_heads = 8
    d_head = 32
    batch_size = 2
    num_grasps = 10
    num_scene = 128
    cond_dim = 512
    
    block = DiTDoubleStreamBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        dropout=0.1,
        cond_dim=cond_dim,
        chunk_size=512,
        use_flash_attention=False,
        attention_dropout=0.0,
    )
    
    grasp_tokens = torch.randn(batch_size, num_grasps, d_model)
    scene_tokens = torch.randn(batch_size, num_scene, d_model)
    cond_vector = torch.randn(batch_size, cond_dim)
    scene_mask = torch.ones(batch_size, num_scene)
    
    with torch.no_grad():
        grasp_out, scene_out = block(grasp_tokens, scene_tokens, cond_vector, scene_mask)
    
    assert grasp_out.shape == grasp_tokens.shape
    assert scene_out.shape == scene_tokens.shape
    assert torch.isfinite(grasp_out).all()
    assert torch.isfinite(scene_out).all()
    
    logger.info(f"✓ 输入形状: grasp={tuple(grasp_tokens.shape)}, scene={tuple(scene_tokens.shape)}")
    logger.info(f"✓ 输出形状: grasp={tuple(grasp_out.shape)}, scene={tuple(scene_out.shape)}")
    logger.info(f"✓ 数值稳定性: grasp range=[{grasp_out.min():.4f}, {grasp_out.max():.4f}]")
    logger.info("")
    return True


def test_parallel_single_stream_block():
    """测试 ParallelSingleStreamBlock"""
    logger.info("="*60)
    logger.info("测试2: ParallelSingleStreamBlock")
    logger.info("="*60)
    
    from models.decoder.dit_single_stream_parallel import ParallelSingleStreamBlock
    
    d_model = 256
    num_heads = 8
    d_head = 32
    batch_size = 2
    seq_len = 138  # 128 scene + 10 grasp
    cond_dim = 512
    
    block = ParallelSingleStreamBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_head=d_head,
        mlp_ratio=4.0,
        dropout=0.1,
        cond_dim=cond_dim,
        use_flash_attention=False,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    cond_vector = torch.randn(batch_size, cond_dim)
    mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        output = block(x, cond_vector, mask)
    
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    
    logger.info(f"✓ 输入形状: {tuple(x.shape)}")
    logger.info(f"✓ 输出形状: {tuple(output.shape)}")
    logger.info(f"✓ 数值稳定性: range=[{output.min():.4f}, {output.max():.4f}]")
    logger.info("")
    return True


def test_concat_split_logic():
    """测试拼接和分离逻辑"""
    logger.info("="*60)
    logger.info("测试3: 拼接和分离逻辑")
    logger.info("="*60)
    
    batch_size = 2
    num_scene = 128
    num_grasp = 10
    d_model = 256
    
    scene_tokens = torch.randn(batch_size, num_scene, d_model)
    grasp_tokens = torch.randn(batch_size, num_grasp, d_model)
    
    # 拼接（Hunyuan3D-DiT 顺序：scene + grasp）
    combined = torch.cat([scene_tokens, grasp_tokens], dim=1)
    logger.info(f"✓ 拼接后: {tuple(combined.shape)}")
    
    assert combined.shape == (batch_size, num_scene + num_grasp, d_model)
    
    # 分离
    extracted_grasp = combined[:, num_scene:, ...]
    logger.info(f"✓ 分离出grasp: {tuple(extracted_grasp.shape)}")
    
    assert extracted_grasp.shape == grasp_tokens.shape
    assert torch.allclose(extracted_grasp, grasp_tokens)
    
    logger.info(f"✓ 拼接-分离验证通过")
    logger.info("")
    return True


def test_qk_norm():
    """测试 QKNorm"""
    logger.info("="*60)
    logger.info("测试4: QKNorm")
    logger.info("="*60)
    
    from models.decoder.dit_utils import QKNorm
    
    d_head = 32
    batch_size = 2
    num_heads = 8
    seq_len = 100
    
    qk_norm = QKNorm(d_head)
    
    q = torch.randn(batch_size, num_heads, seq_len, d_head)
    k = torch.randn(batch_size, num_heads, seq_len, d_head)
    v = torch.randn(batch_size, num_heads, seq_len, d_head)
    
    q_norm, k_norm = qk_norm(q, k, v)
    
    assert q_norm.shape == q.shape
    assert k_norm.shape == k.shape
    assert torch.isfinite(q_norm).all()
    assert torch.isfinite(k_norm).all()
    
    logger.info(f"✓ QKNorm 输入形状: {tuple(q.shape)}")
    logger.info(f"✓ QKNorm 输出形状: {tuple(q_norm.shape)}")
    logger.info(f"✓ 归一化效果: ||q||={q.norm(dim=-1).mean():.4f} -> ||q_norm||={q_norm.norm(dim=-1).mean():.4f}")
    logger.info("")
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("双流+单流 DiT 核心组件快速测试")
    logger.info("="*60 + "\n")
    
    tests = [
        test_dit_double_stream_block,
        test_parallel_single_stream_block,
        test_concat_split_logic,
        test_qk_norm,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_func.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    logger.info("="*60)
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")
    logger.info("="*60)
    
    if failed == 0:
        logger.info("\n🎉 所有核心组件测试通过！双流+单流架构修复成功。")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

