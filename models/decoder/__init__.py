# from .detr3d_decoder import Detr3DDecoder
# from .grasp_embed_decoder import GraspKDecoder
# from .vanilla_decoder import VanillaDecoder
# from .task_decoder import GraspTaskDecoder
# from .unet import UNetModel
from .unet_new import UNetModel
from .dit import DiTModel
from .dit_config_validation import validate_dit_compatibility_with_diffuser

def build_decoder(decoder_cfg, diffuser_cfg=None):
    """
    Build decoder model based on configuration.
    
    Args:
        decoder_cfg: Decoder configuration
        diffuser_cfg: Optional diffuser configuration for compatibility validation
        
    Returns:
        Decoder model instance
    """
    if decoder_cfg.name.lower() == "task":
        # return GraspTaskDecoder(decoder_cfg)
        raise NotImplementedError(f"No such decoder: {decoder_cfg.name}")
    if decoder_cfg.name.lower() == "3detr":
        # return Detr3DDecoder(decoder_cfg)
        raise NotImplementedError(f"No such decoder: {decoder_cfg.name}")
    elif decoder_cfg.name.lower() == "vanilla":
        # return VanillaDecoder(decoder_cfg)
        raise NotImplementedError(f"No such decoder: {decoder_cfg.name}")
    elif decoder_cfg.name.lower() == "graspk":
        # return GraspKDecoder(decoder_cfg)
        raise NotImplementedError(f"No such decoder: {decoder_cfg.name}")
    elif decoder_cfg.name.lower() == "unet":
        return UNetModel(decoder_cfg)
    elif decoder_cfg.name.lower() == "dit":
        # Validate DiT compatibility with diffuser config if provided
        if diffuser_cfg is not None:
            validate_dit_compatibility_with_diffuser(decoder_cfg, diffuser_cfg)
        return DiTModel(decoder_cfg)
    else:
        raise NotImplementedError(f"No such decode: {decoder_cfg.name}")
