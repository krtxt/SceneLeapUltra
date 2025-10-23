from .pointnet2 import Pointnet2Backbone
from .pointnet2_3sa import Pointnet2Backbone_3sa
from .resnet import build_resnet_backbone

def build_backbone(backbone_cfg):
    if backbone_cfg.name.lower() == "resnet":
        return build_resnet_backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet2":
        return Pointnet2Backbone(backbone_cfg)
    elif backbone_cfg.name.lower() == "pointnet2_3sa":
        return Pointnet2Backbone_3sa(backbone_cfg)
    else:
        raise NotImplementedError(f"No such backbone: {backbone_cfg.name}")
