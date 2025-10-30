import pytest

torch = pytest.importorskip("torch")

from models.decoder.dit import DiTSingleParallelBlock, DiTConditioningError


def test_single_parallel_block_basic():
    block = DiTSingleParallelBlock(
        d_model=32,
        num_heads=4,
        d_head=8,
        dropout=0.0,
        use_adaln_zero=True,
        cond_dim=64,
        chunk_size=16,
        use_flash_attention=False,
        attention_dropout=0.0,
        cross_attention_dropout=0.0,
        use_geometric_bias=False,
        geometric_bias_module=None,
        time_gate=None,
    )

    x = torch.randn(2, 6, 32)
    scene = torch.randn(2, 5, 32)
    cond = torch.randn(2, 64)
    out = block(x, cond_vector=cond, scene_context=scene)
    assert out.shape == x.shape


def test_single_parallel_block_requires_cond():
    block = DiTSingleParallelBlock(
        d_model=16,
        num_heads=2,
        d_head=8,
        dropout=0.0,
        use_adaln_zero=True,
        cond_dim=24,
        chunk_size=8,
        use_flash_attention=False,
        attention_dropout=0.0,
        cross_attention_dropout=0.0,
        use_geometric_bias=False,
        geometric_bias_module=None,
        time_gate=None,
    )

    with pytest.raises(DiTConditioningError):
        block(torch.randn(1, 3, 16), cond_vector=None)  # type: ignore[arg-type]
