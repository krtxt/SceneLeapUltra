import pytest

torch = pytest.importorskip("torch")

from models.decoder.dit import DiTDoubleStreamBlock, DiTConditioningError


def test_double_stream_block_forward_shapes():
    block = DiTDoubleStreamBlock(
        d_model=32,
        num_heads=4,
        d_head=8,
        dropout=0.0,
        cond_dim=64,
        chunk_size=16,
        use_flash_attention=False,
        attention_dropout=0.0,
    )

    grasp_tokens = torch.randn(2, 5, 32)
    scene_tokens = torch.randn(2, 7, 32)
    cond_vector = torch.randn(2, 64)
    scene_mask = torch.ones(2, 7)

    grasp_out, scene_out = block(
        grasp_tokens=grasp_tokens,
        scene_tokens=scene_tokens,
        cond_vector=cond_vector,
        scene_mask=scene_mask,
    )

    assert grasp_out.shape == grasp_tokens.shape
    assert scene_out.shape == scene_tokens.shape


def test_double_stream_block_requires_cond_vector():
    block = DiTDoubleStreamBlock(
        d_model=16,
        num_heads=2,
        d_head=8,
        dropout=0.0,
        cond_dim=24,
        chunk_size=8,
        use_flash_attention=False,
        attention_dropout=0.0,
    )
    grasp_tokens = torch.randn(1, 3, 16)
    scene_tokens = torch.randn(1, 4, 16)

    with pytest.raises(DiTConditioningError):
        block(
            grasp_tokens=grasp_tokens,
            scene_tokens=scene_tokens,
            cond_vector=None,  # type: ignore[arg-type]
        )
