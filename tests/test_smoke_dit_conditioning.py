import torch
import torch.nn as nn

from models.decoder.dit_conditioning import prepare_scene_features


class DummyBackbone(nn.Module):
    def __init__(self, input_dim: int = 3, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, scene_points: torch.Tensor):
        # scene_points: [B, N, D]
        feat = self.proj(scene_points)  # [B, N, C]
        feat = feat.permute(0, 2, 1).contiguous()  # [B, C, N]
        return None, feat


def test_prepare_scene_features_smoke_pointnet2():
    # Minimal synthetic scene_pc (B, N, 3)
    B, N = 2, 8
    scene_pc = torch.randn(B, N, 3)
    data = {"scene_pc": scene_pc}

    scene_model = DummyBackbone(input_dim=3, output_dim=512)
    scene_projection = nn.Linear(getattr(scene_model, 'output_dim', 512), 256)

    out = prepare_scene_features(
        scene_model=scene_model,
        scene_projection=scene_projection,
        data=data,
        use_rgb=False,
        use_object_mask=False,
        device=scene_projection.weight.device,
        logger=None,
        strict=True,
    )

    assert isinstance(out, torch.Tensor)
    assert out.dim() == 3
    assert out.shape[0] == B

