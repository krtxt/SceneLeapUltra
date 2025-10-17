from typing import Dict
import torch

# 手部姿态向量切片常量（与 DiffuserLightning 保持一致）
TRANSLATION_SLICE = slice(0, 3)
QPOS_SLICE = slice(3, 19)
ROTATION_SLICE = slice(19, None)

def build_pred_dict_adaptive(pred_x0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """根据 pred_x0 的维度自动构建预测字典，统一训练/推理接口使用。"""
    return {
        "pred_pose_norm": pred_x0,
        "qpos_norm": pred_x0[..., QPOS_SLICE],
        "translation_norm": pred_x0[..., TRANSLATION_SLICE],
        "rotation": pred_x0[..., ROTATION_SLICE],
        "pred_noise": torch.tensor([1.0], device=pred_x0.device),
        "noise": torch.tensor([1.0], device=pred_x0.device),
    } 