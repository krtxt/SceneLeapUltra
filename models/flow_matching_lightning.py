import logging
from typing import Any, Dict, Tuple

import torch

from models.base_lightning import BaseGraspGeneratorLightning
from models.decoder import build_decoder
from models.loss import GraspLossPose
from models.utils.flow_matching_core import FlowMatchingCoreMixin
from utils.hand_helper import process_hand_pose


class FlowMatchingLightning(FlowMatchingCoreMixin, BaseGraspGeneratorLightning):
    """Flow Matching 版抓取生成 LightningModule。"""

    def __init__(self, cfg):
        super().__init__()
        logging.info("Initializing FlowMatchingLightning model")
        self.save_hyperparameters()

        self.use_negative_prompts = cfg.get("use_negative_prompts", True)

        self.eps_model = build_decoder(cfg.decoder, cfg)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.rot_type = cfg.rot_type
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        self.mode = cfg.mode

        self.use_score = cfg.get("use_score", False)
        self.score_pretrain = cfg.get("score_pretrain", False)

        self.optimizer_cfg = cfg.optimizer
        self.scheduler = cfg.scheduler

        self.use_cfg = cfg.use_cfg
        self.guidance_scale = cfg.guidance_scale
        self.use_negative_guidance = cfg.use_negative_guidance and self.use_negative_prompts
        self.negative_guidance_scale = cfg.negative_guidance_scale

        self.fix_num_grasps = cfg.get("fix_num_grasps", False)
        self.target_num_grasps = cfg.get("target_num_grasps", None)

        wandb_opt = cfg.get("wandb_optimization", {})
        self._log_gradients = wandb_opt.get("log_gradients", False)
        self._gradient_freq = wandb_opt.get("gradient_freq", 1000)
        self._monitor_system = wandb_opt.get("monitor_system", False)
        self._system_freq = wandb_opt.get("system_freq", 500)

        self.flow_cfg = cfg.flow
        self.integrator_cfg = cfg.integrator

        self.base_std = self.flow_cfg.get("base_std", 1.0)
        self.velocity_scale = self.flow_cfg.get("velocity_scale", 1.0)
        self.time_embedding_scale = float(self.flow_cfg.get("time_embedding_scale", 1000.0))
        self.loss_weighting = (self.flow_cfg.get("loss_weighting", "none") or "none").lower()
        self.loss_reduction = (self.flow_cfg.get("loss_reduction", "mean") or "mean").lower()

        self.integration_steps = int(self.integrator_cfg.get("num_steps", 64))
        self.integration_method = self.integrator_cfg.get("method", "euler").lower()
        self.integration_t_start = float(self.integrator_cfg.get("t_start", 0.0))
        self.integration_t_end = float(self.integrator_cfg.get("t_end", 1.0))

    # =====================
    # Helper Methods
    # =====================

    def _compute_loss(
        self, batch: Dict, mode: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        norm_pose = processed_batch["norm_pose"]
        if norm_pose.dim() == 2:
            norm_pose = norm_pose.unsqueeze(1)
            processed_batch["norm_pose"] = norm_pose

        if torch.isnan(norm_pose).any():
            logging.error("NaN detected in normalized pose input")
            raise RuntimeError("NaN detected in normalized pose input")

        B = norm_pose.shape[0]
        num_grasps = norm_pose.shape[1]
        pose_dim = norm_pose.shape[2]

        base_noise = self._sample_base_noise(norm_pose)
        t = self._sample_times(B)
        x_t = self._interpolate(base_noise, norm_pose, t)

        condition_dict = self.eps_model.condition(processed_batch)
        processed_batch.update(condition_dict)

        ts = self._scale_timesteps(t)
        velocity_pred = self.eps_model(x_t, ts, processed_batch)
        velocity_target = self._velocity_target(base_noise, norm_pose, t)

        loss_tensor = (velocity_pred - velocity_target) ** 2
        weight = self._compute_time_weight(t, num_grasps, pose_dim)
        if weight is not None:
            loss_tensor = loss_tensor * weight

        if self.loss_reduction == "sum":
            denom = weight.sum() if weight is not None else torch.tensor(
                loss_tensor.numel(), device=self.device, dtype=loss_tensor.dtype
            )
            loss = loss_tensor.sum() / denom.clamp_min(1e-6)
        elif self.loss_reduction == "mean":
            loss = loss_tensor.mean()
        else:
            raise ValueError(f"Unsupported loss_reduction: {self.loss_reduction}")

        loss_dict = {"flow_matching": loss}
        processed_batch["flow_time"] = t
        processed_batch["base_noise"] = base_noise
        return loss, loss_dict, processed_batch

    def _compute_time_weight(self, t: torch.Tensor, num_grasps: int, pose_dim: int) -> torch.Tensor:
        weighting = self.loss_weighting if isinstance(self.loss_weighting, str) else "none"
        if weighting == "none":
            return None
        if weighting == "linear":
            weight = t
        elif weighting == "one_minus":
            weight = 1.0 - t
        elif weighting == "min":
            weight = torch.minimum(t, 1.0 - t)
        elif weighting == "max":
            weight = torch.maximum(t, 1.0 - t)
        else:
            raise ValueError(f"Unsupported loss_weighting: {self.loss_weighting}")

        weight = weight.view(-1, 1, 1).expand(-1, num_grasps, pose_dim)
        return weight

    # =====================
    # Sampling
    # =====================

    @torch.no_grad()
    def sample(
        self,
        data: Dict,
        k: int = 1,
        use_cfg: bool = None,
        guidance_scale: float = None,
        use_negative_guidance: bool = None,
        negative_guidance_scale: float = None,
    ):
        cfg_params = {
            "use_cfg": use_cfg if use_cfg is not None else self.use_cfg,
            "guidance_scale": guidance_scale if guidance_scale is not None else self.guidance_scale,
            "use_negative_guidance": use_negative_guidance if use_negative_guidance is not None else self.use_negative_guidance,
            "negative_guidance_scale": negative_guidance_scale if negative_guidance_scale is not None else self.negative_guidance_scale,
        }
        ksamples = [self._flow_sample_loop(data, **cfg_params) for _ in range(k)]
        return torch.stack(ksamples, dim=1)

    @torch.no_grad()
    def _flow_sample_loop(
        self,
        data: Dict,
        use_cfg: bool,
        guidance_scale: float,
        use_negative_guidance: bool,
        negative_guidance_scale: float,
    ) -> torch.Tensor:
        if "norm_pose" not in data:
            raise ValueError("norm_pose required for sampling")

        norm_pose = data["norm_pose"]
        if isinstance(norm_pose, torch.Tensor):
            B, orig_num_grasps, pose_dim = norm_pose.shape
            target_num_grasps = orig_num_grasps
            if self.fix_num_grasps and self.target_num_grasps is not None:
                target_num_grasps = self.target_num_grasps
                if target_num_grasps != orig_num_grasps:
                    logging.info(
                        f"FlowMatching: Adjusting grasp count from {orig_num_grasps} to {target_num_grasps}"
                    )
            if target_num_grasps == orig_num_grasps:
                x0 = self._sample_base_noise(norm_pose)
            else:
                ref = norm_pose.new_zeros((B, target_num_grasps, pose_dim))
                x0 = self._sample_base_noise(ref)
        else:
            pose_dim = norm_pose[0].shape[-1]
            ref = torch.zeros((len(norm_pose), pose_dim), device=self.device, dtype=torch.float32)
            x0 = self._sample_base_noise(ref)

        condition_dict = self.eps_model.condition(data)
        data.update(condition_dict)

        steps = self.integration_steps
        method = self.integration_method
        t_start = self.integration_t_start
        t_end = self.integration_t_end

        path = self._flow_integrate(
            x0,
            data,
            steps,
            method,
            use_cfg,
            guidance_scale,
            use_negative_guidance,
            negative_guidance_scale,
            t_start,
            t_end,
        )
        return path

    # =====================
    # Checkpoint Utilities
    # =====================

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            if (
                hasattr(self, "eps_model")
                and hasattr(self.eps_model, "_ensure_text_encoder")
                and hasattr(self.eps_model, "use_text_condition")
                and self.eps_model.use_text_condition
            ):
                self.eps_model._ensure_text_encoder()
                logging.info("Text encoder initialized for checkpoint loading")
            elif hasattr(self, "eps_model") and hasattr(self.eps_model, "use_text_condition"):
                logging.info(
                    f"Skipping text encoder initialization (use_text_condition={self.eps_model.use_text_condition})"
                )
        except Exception as e:
            logging.warning(f"Failed to ensure text encoder before loading checkpoint: {e}")

        if getattr(self, "use_score", False) and not getattr(self, "score_pretrain", False):
            model_state_dict = self.state_dict()
            checkpoint_state_dict = checkpoint["state_dict"]
            new_state_dict = {k: v for k, v in checkpoint_state_dict.items() if "score_heads" not in k}
            model_state_dict.update(new_state_dict)
            self.load_state_dict(model_state_dict)
            logging.info("Loaded checkpoint weights, excluding score_heads.")
        else:
            logging.info("Loading full checkpoint.")

    def _check_state_dict(self, dict1, dict2):
        keys1, keys2 = set(dict1.keys()), set(dict2.keys())
        if keys1 != keys2:
            logging.warning("State dict key mismatch!")
            logging.warning(f"Only in checkpoint: {keys1 - keys2}")
            logging.warning(f"Only in model: {keys2 - keys1}")
            return True

        for key in keys1:
            if dict1[key].shape != dict2[key].shape:
                logging.warning(
                    f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}"
                )
                return True
        return False
