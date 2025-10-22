import logging
from statistics import mean
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torchdiffeq import odeint

from models.decoder import build_decoder
from models.loss import GraspLossPose
from models.utils.diffusion_utils import make_schedule_ddpm
from models.utils.prediction import build_pred_dict_adaptive
from utils.hand_helper import (
    denorm_hand_pose_robust,
    process_hand_pose,
    process_hand_pose_test,
)


class FlowMatchingLightning(pl.LightningModule):
    """Flow Matching based grasp generation Lightning module."""

    def __init__(self, cfg):
        super().__init__()
        logging.info("Initialising FlowMatchingLightning model")
        self.save_hyperparameters()

        self.cfg = cfg
        self.rot_type = cfg.rot_type
        self.mode = cfg.mode
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        self.use_negative_prompts = cfg.get("use_negative_prompts", False)
        self.fix_num_grasps = cfg.get("fix_num_grasps", False)
        self.target_num_grasps = cfg.get("target_num_grasps", None)

        # Model components
        self.velocity_model = build_decoder(cfg.decoder, cfg)
        self.train_loss_fn = torch.nn.MSELoss(reduction="none")
        self.eval_criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights

        # Optimisation configuration
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        # Flow-specific configuration
        self.flow_cfg = cfg.get("flow_cfg", {})
        self.path_type = self.flow_cfg.get("path_type", "rectified").lower()
        self.diffusion_steps = None
        self._prepare_diffusion_schedule_if_needed()

        # ODE solver configuration
        self.ode_cfg = cfg.get("ode_solver_cfg", {})
        self.ode_method = self.ode_cfg.get("method", "dopri5")
        self.ode_n_steps = max(2, int(self.ode_cfg.get("n_steps", 20)))
        self.ode_rtol = float(self.ode_cfg.get("rtol", 1e-5))
        self.ode_atol = float(self.ode_cfg.get("atol", 1e-5))

        # Guidance configuration
        self.guidance_cfg = cfg.get("guidance_cfg", {})
        self.guidance_scale = float(self.guidance_cfg.get("w", 0.0))
        self.guidance_type = self.guidance_cfg.get("stable_guidance_type", "naive").lower()
        self.zero_init_threshold = float(self.guidance_cfg.get("zero_init_threshold", 0.0))
        self.use_cfg = self.guidance_scale > 0.0

        # Internal containers for logging
        self.validation_step_outputs = []
        self.metric_results = []

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int):
        processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        norm_pose = processed_batch["norm_pose"].to(self.device)
        hand_pose = processed_batch["hand_model_pose"].to(self.device)
        processed_batch["norm_pose"] = norm_pose
        processed_batch["hand_model_pose"] = hand_pose
        B, num_grasps, pose_dim = norm_pose.shape

        # Mask invalid grasps (all-zero placeholders)
        valid_mask = (hand_pose.abs().sum(dim=-1) > 0).float().unsqueeze(-1)

        # Sample initial noise and time
        x0 = torch.randn_like(norm_pose)
        ts = torch.rand(B, device=self.device)
        ts_broadcast = ts.view(B, 1, 1)

        xt, ut = self._compute_path_states(x0, norm_pose, ts_broadcast)

        # Prepare conditioning information
        condition_payload = self.velocity_model.condition(processed_batch)
        processed_batch.update(condition_payload)

        vt = self.velocity_model(xt, ts, processed_batch)

        loss_raw = self.train_loss_fn(vt, ut)
        loss_masked = loss_raw * valid_mask
        denom = valid_mask.sum() * pose_dim
        denom = torch.clamp(denom, min=1.0)
        loss = loss_masked.sum() / denom

        total_samples = B * num_grasps
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=total_samples, sync_dist=True)
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        self.log("train/lr", optimizer.param_groups[0]["lr"], prog_bar=False, on_step=True, batch_size=total_samples, sync_dist=True)

        if batch_idx % self.print_freq == 0 and self.trainer.is_global_zero:
            logging.info(
                f"[FlowMatching][Epoch {self.current_epoch}] Batch {batch_idx}/{len(self.trainer.train_dataloader)} - "
                f"loss={loss.item():.6f}"
            )

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_traj = self.sample(batch)
        pred_x1 = pred_traj[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x1)

        B, num_grasps, _ = pred_x1.shape
        batch_size = B * num_grasps

        loss_dict = self.eval_criterion(pred_dict, batch, mode="val")
        if hasattr(self.eval_criterion, "use_standardized_val_loss") and self.eval_criterion.use_standardized_val_loss:
            standard_losses = loss_dict.pop("_standard_losses", {})
            std_weights = getattr(self.eval_criterion, "std_val_weights", {})
            standardized_loss = sum(
                loss_dict[k] * std_weights.get(k, 0.0) for k in loss_dict.keys() if k in std_weights and std_weights[k] > 0
            )
            standard_loss = sum(
                standard_losses[k] * self.loss_weights.get(k, 0.0)
                for k in standard_losses.keys()
                if k in self.loss_weights
            )
            self.log("val/loss", standardized_loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("val/standardized_loss", standardized_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("val/standard_loss", standard_loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.validation_step_outputs.append(
                {
                    "loss": standardized_loss.item(),
                    "standardized_loss": standardized_loss.item(),
                    "standard_loss": standard_loss.item(),
                    "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                    "standard_loss_dict": {k: v.item() for k, v in standard_losses.items()},
                }
            )
        else:
            total_loss = sum(loss_dict[k] * self.loss_weights.get(k, 0.0) for k in loss_dict.keys())
            self.log("val/loss", total_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.validation_step_outputs.append(
                {
                    "loss": total_loss.item(),
                    "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                }
            )

        return {"loss_dict": loss_dict}

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        losses = [entry["loss"] for entry in self.validation_step_outputs]
        avg_loss = mean(losses)

        using_standardized = (
            hasattr(self.eval_criterion, "use_standardized_val_loss")
            and self.eval_criterion.use_standardized_val_loss
            and "standardized_loss" in self.validation_step_outputs[0]
        )

        if using_standardized:
            std_losses = [entry["standardized_loss"] for entry in self.validation_step_outputs]
            standard_losses = [entry["standard_loss"] for entry in self.validation_step_outputs]

            avg_std_loss = mean(std_losses)
            avg_standard_loss = mean(standard_losses)

            detailed_std = {
                k: mean(x["loss_dict"][k] for x in self.validation_step_outputs)
                for k in self.validation_step_outputs[0]["loss_dict"].keys()
            }
            detailed_standard = {
                k: mean(x["standard_loss_dict"][k] for x in self.validation_step_outputs)
                for k in self.validation_step_outputs[0]["standard_loss_dict"].keys()
            }

            self.log("val/standardized_total_loss", avg_std_loss, prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.log("val/original_total_loss", avg_standard_loss, prog_bar=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.log("val_loss", avg_std_loss, prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

            for k, v in detailed_std.items():
                self.log(f"val/std_{k}", v, prog_bar=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            for k, v in detailed_standard.items():
                self.log(f"val/orig_{k}", v, prog_bar=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        else:
            detailed_loss = {
                k: mean(x["loss_dict"][k] for x in self.validation_step_outputs)
                for k in self.validation_step_outputs[0]["loss_dict"].keys()
            }
            for k, v in detailed_loss.items():
                self.log(f"val/{k}", v, prog_bar=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.log("val/total_loss", avg_loss, prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
            self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.metric_results = []

    def test_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_traj = self.sample(batch)
        pred_x1 = pred_traj[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x1)
        metric_dict, metric_details = self.eval_criterion(pred_dict, batch, mode="test")
        self.metric_results.append(metric_details)
        return metric_dict

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_name = self.optimizer_cfg.name.lower()
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.optimizer_cfg.name}")

        sched_name = self.scheduler_cfg.name.lower()
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.t_max,
                eta_min=self.scheduler_cfg.min_lr,
            )
        elif sched_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_cfg.step_size,
                gamma=self.scheduler_cfg.step_gamma,
            )
        else:
            raise NotImplementedError(f"Unsupported scheduler: {self.scheduler_cfg.name}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, data: Dict, k: int = 1):
        if "norm_pose" not in data:
            raise ValueError("Batch must contain 'norm_pose' for sampling")

        # Prepare conditioning once
        conditioned_batch = self._clone_batch_for_inference(data)
        condition_payload = self.velocity_model.condition(conditioned_batch)
        conditioned_batch.update(condition_payload)

        uncond_batch = None
        if self.use_cfg:
            uncond_batch = self._build_unconditional_batch(data)
            uncond_condition = self.velocity_model.condition(uncond_batch)
            uncond_batch.update(uncond_condition)

        samples = [self._generate_single_sample(conditioned_batch, uncond_batch) for _ in range(k)]
        return torch.stack(samples, dim=1)

    def _generate_single_sample(self, cond_batch: Dict, uncond_batch: Optional[Dict]) -> torch.Tensor:
        norm_pose = cond_batch["norm_pose"]
        device = self.device

        if isinstance(norm_pose, torch.Tensor):
            norm_pose_tensor = norm_pose.to(device)
            cond_batch["norm_pose"] = norm_pose_tensor
            if uncond_batch is not None:
                uncond_batch["norm_pose"] = norm_pose_tensor
            B, num_grasps, pose_dim = norm_pose_tensor.shape
            target_num_grasps = num_grasps
            if self.fix_num_grasps and self.target_num_grasps is not None:
                target_num_grasps = self.target_num_grasps
            if target_num_grasps == num_grasps:
                x0 = torch.randn_like(norm_pose_tensor)
                valid_mask = (norm_pose_tensor.abs().sum(dim=-1) > 0).unsqueeze(-1).float()
            else:
                x0 = torch.randn(B, target_num_grasps, pose_dim, device=device)
                valid_mask = torch.ones(B, target_num_grasps, 1, device=device)
        else:
            raise NotImplementedError("List-based norm_pose sampling is not yet supported for Flow Matching")

        valid_mask = valid_mask.unsqueeze(1)
        times = torch.linspace(0.0, 1.0, steps=self.ode_n_steps, device=device, dtype=x0.dtype)

        def dynamics(t, state):
            return self._velocity_field(state, t, cond_batch, uncond_batch)

        solution = odeint(
            dynamics,
            x0,
            times,
            method=self.ode_method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )
        traj = solution.permute(1, 0, 2, 3)
        traj = traj * valid_mask  # enforce mask at all steps
        return traj

    # ------------------------------------------------------------------
    # Flow computation helpers
    # ------------------------------------------------------------------

    def _compute_path_states(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.path_type == "rectified":
            xt = (1.0 - t) * x0 + t * x1
            ut = x1 - x0
            return xt, ut
        if self.path_type == "diffusion":
            return self._diffusion_path(x0, x1, t)
        raise ValueError(f"Unsupported flow path type: {self.path_type}")

    def _diffusion_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "diffusion_alphas_cumprod"):
            raise RuntimeError("Diffusion schedule not initialised for diffusion path")

        steps_minus_one = float(self.diffusion_steps - 1)
        s = torch.clamp(t.squeeze(-1).squeeze(-1) * steps_minus_one, 0.0, steps_minus_one - 1e-6)
        lower_idx = torch.floor(s).long()
        upper_idx = torch.clamp(lower_idx + 1, max=self.diffusion_steps - 1)
        frac = (s - lower_idx.float()).unsqueeze(-1).unsqueeze(-1)

        alpha_cp_lower = self.diffusion_alphas_cumprod[lower_idx]
        alpha_cp_upper = self.diffusion_alphas_cumprod[upper_idx]
        alpha_cp_interp = alpha_cp_lower + (alpha_cp_upper - alpha_cp_lower) * frac.squeeze(-1).squeeze(-1)
        alpha_t = torch.sqrt(torch.clamp(alpha_cp_interp, min=1e-6)).view(-1, 1, 1)
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_cp_interp, min=1e-6)).view(-1, 1, 1)

        alpha_cp_deriv = (alpha_cp_upper - alpha_cp_lower) * steps_minus_one
        alpha_dot = (0.5 * alpha_cp_deriv / torch.clamp(alpha_t.view(-1), min=1e-6)).view(-1, 1, 1)
        sigma_dot = (-0.5 * alpha_cp_deriv / torch.clamp(sigma_t.view(-1), min=1e-6)).view(-1, 1, 1)

        xt = alpha_t * x1 + sigma_t * x0
        denom = torch.clamp(alpha_t.pow(2) + sigma_t.pow(2), min=1e-6)
        ut = ((alpha_dot * alpha_t + sigma_dot * sigma_t) * xt - alpha_dot * (alpha_t.pow(2)) * x1) / denom
        return xt, ut

    def _velocity_field(
        self,
        x_t: torch.Tensor,
        t_scalar: torch.Tensor,
        cond_batch: Dict,
        uncond_batch: Optional[Dict],
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        t_value = torch.as_tensor(t_scalar, dtype=torch.float32, device=x_t.device)
        t_values = t_value.repeat(batch_size)
        v_cond = self.velocity_model(x_t, t_values, cond_batch)

        if not self.use_cfg or uncond_batch is None:
            return v_cond

        if t_value.item() < self.zero_init_threshold:
            return self.velocity_model(x_t, t_values, uncond_batch)

        v_uncond = self.velocity_model(x_t, t_values, uncond_batch)
        if self.guidance_type == "cfg_zero_star":
            delta = v_cond - v_uncond
            numerator = (delta * v_uncond).sum(dim=-1, keepdim=True)
            denominator = torch.clamp((v_uncond.pow(2)).sum(dim=-1, keepdim=True), min=1e-6)
            s_star = torch.clamp(numerator / denominator, min=0.0, max=1.0)
            return v_uncond + self.guidance_scale * s_star * delta

        # Naive CFG as fallback
        return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _prepare_diffusion_schedule_if_needed(self):
        if self.path_type != "diffusion":
            return

        diffusion_cfg = self.flow_cfg.get("diffusion", {})
        self.diffusion_steps = int(diffusion_cfg.get("steps", 100))
        schedule_kwargs = diffusion_cfg.get(
            "schedule_cfg",
            {
                "beta": [0.0001, 0.02],
                "beta_schedule": "linear",
                "s": 0.008,
            },
        )
        schedule = make_schedule_ddpm(self.diffusion_steps, **schedule_kwargs)
        for key, value in schedule.items():
            self.register_buffer(f"diffusion_{key}", value.float(), persistent=False)

    def _clone_batch_for_inference(self, batch: Dict) -> Dict:
        cloned = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.to(self.device)
            elif isinstance(value, list):
                if value and all(isinstance(item, torch.Tensor) for item in value):
                    try:
                        stacked = torch.stack([item.to(self.device) for item in value])
                        cloned[key] = stacked
                    except Exception:
                        cloned[key] = [item.to(self.device) for item in value]
                else:
                    cloned[key] = value[:]  # shallow copy for list of non-tensors (e.g., prompts)
            else:
                cloned[key] = value
        return cloned

    def _build_unconditional_batch(self, batch: Dict) -> Dict:
        uncond = self._clone_batch_for_inference(batch)
        if "positive_prompt" in uncond and isinstance(uncond["positive_prompt"], list):
            uncond["positive_prompt"] = ["" for _ in uncond["positive_prompt"]]
        if "negative_prompts" in uncond:
            uncond["negative_prompts"] = None
        return uncond

    # ------------------------------------------------------------------
    # Public inference helpers
    # ------------------------------------------------------------------

    def forward_infer(self, data: Dict, k: int = 1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_norm = self.sample(data, k=k)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_norm)
        preds, targets = self.eval_criterion.infer_norm_process_dict(pred_dict, data)
        return preds, targets

    def forward_get_pose_raw(self, data: Dict, k: int = 1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_pose_norm = self.sample(data, k=k)[:, 0, -1]
        return denorm_hand_pose_robust(pred_pose_norm, self.rot_type, self.mode)
