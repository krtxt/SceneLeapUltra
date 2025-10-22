"""
Flow Matching Lightning Module

PyTorch Lightning module for training and inference with Flow Matching.
This module is designed to be a drop-in replacement for DDPMLightning,
sharing the same decoder architectures (DiT/UNet) and conditioning logic.

Key features:
- Full compatibility with existing DiT and UNet decoders
- Simpler training objective than DDPM (velocity regression)
- Faster sampling (typically 10-50 steps vs 100-1000 for DDPM)
- Supports classifier-free guidance and negative prompts
- Compatible with all existing data pipelines and loss functions
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from models.decoder import build_decoder
from models.loss import GraspLossPose
from utils.hand_helper import process_hand_pose, process_hand_pose_test, denorm_hand_pose_robust
from statistics import mean
import logging
from .utils.log_colors import HEADER, BLUE, GREEN, YELLOW, RED, ENDC, BOLD, UNDERLINE
from .utils.flow_matching_core import FlowMatchingCoreMixin
from .utils.prediction import build_pred_dict_adaptive
from .utils.logging_helpers import log_validation_summary


class FlowMatchingLightning(FlowMatchingCoreMixin, pl.LightningModule):
    """
    Flow Matching model using PyTorch Lightning.

    This model learns to predict velocity fields for conditional flow matching,
    providing a simpler and often faster alternative to DDPM while maintaining
    full compatibility with existing decoders and infrastructure.
    """

    def __init__(self, cfg):
        super().__init__()
        logging.info(f"{GREEN}Initializing FlowMatchingLightning model{ENDC}")
        self.save_hyperparameters()

        # Negative prompts support
        self.use_negative_prompts = cfg.get('use_negative_prompts', True)

        # Build velocity prediction model (reuse DDPM decoder architecture)
        # Note: The decoder (DiT/UNet) is the same, we just use it to predict velocity instead of noise
        self.velocity_model = build_decoder(cfg.decoder, cfg)

        # Loss and metrics
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.rot_type = cfg.rot_type
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        self.mode = cfg.mode

        # Flow matching specific parameters
        self.sigma_min = cfg.get('sigma_min', 1e-4)  # Minimum noise for numerical stability
        self.sampler_type = cfg.get('sampler_type', 'euler')  # 'euler' or 'heun'
        self.num_sampling_steps = cfg.get('num_sampling_steps', 50)  # Fewer steps than DDPM!
        self.loss_type = cfg.get('loss_type', 'l2')  # 'l2' or 'l1' for velocity matching

        # Initialize flow matching core
        self._init_flow_matching()

        # Optimizer and scheduler config
        self.optimizer_cfg = cfg.optimizer
        self.scheduler = cfg.scheduler

        # Classifier-free guidance
        self.use_cfg = cfg.get('use_cfg', False)
        self.guidance_scale = cfg.get('guidance_scale', 1.0)
        self.use_negative_guidance = cfg.get('use_negative_guidance', False) and self.use_negative_prompts
        self.negative_guidance_scale = cfg.get('negative_guidance_scale', 0.0)

        # Grasp count control
        self.fix_num_grasps = cfg.get('fix_num_grasps', False)
        self.target_num_grasps = cfg.get('target_num_grasps', None)

        # WandB optimization parameters
        wandb_opt = cfg.get('wandb_optimization', {})
        self._log_gradients = wandb_opt.get('log_gradients', False)
        self._gradient_freq = wandb_opt.get('gradient_freq', 1000)
        self._monitor_system = wandb_opt.get('monitor_system', False)
        self._system_freq = wandb_opt.get('system_freq', 500)

        logging.info(f"{BLUE}Flow Matching Config:{ENDC}")
        logging.info(f"  - Sampler: {self.sampler_type}")
        logging.info(f"  - Sampling steps: {self.num_sampling_steps}")
        logging.info(f"  - Sigma min: {self.sigma_min}")
        logging.info(f"  - Loss type: {self.loss_type}")

    # =====================
    # PyTorch Lightning Hooks
    # =====================

    def training_step(self, batch, batch_idx):
        """Training step for flow matching."""
        # Compute flow matching loss
        loss, loss_dict, processed_batch = self._compute_flow_matching_loss(batch, mode='train')

        # Calculate total samples
        norm_pose = processed_batch['norm_pose']
        B, num_grasps, _ = norm_pose.shape
        total_samples = B * num_grasps

        # Log training losses
        train_log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        train_log_dict["train/total_loss"] = loss
        self.log_dict(train_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True,
                     batch_size=total_samples, sync_dist=True)

        # Log learning rate
        optimizer = self.optimizers()
        self.log("train/lr", optimizer.param_groups[0]['lr'], prog_bar=False, logger=True,
                on_step=True, batch_size=total_samples, sync_dist=True)

        # Log gradient norm periodically
        if self._log_gradients and batch_idx % self._gradient_freq == 0:
            total_norm = 0
            param_count = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_norm = total_norm ** 0.5
                self.log("train/grad_norm", total_norm, prog_bar=False, logger=True,
                        on_step=True, batch_size=total_samples, sync_dist=True)

        # Log system resource usage periodically
        if self._monitor_system and batch_idx % self._system_freq == 0:
            if torch.cuda.is_available():
                self.log("system/gpu_memory_allocated_gb", torch.cuda.memory_allocated() / 1024**3,
                        prog_bar=False, logger=True, on_step=True, batch_size=total_samples)
                self.log("system/gpu_memory_reserved_gb", torch.cuda.memory_reserved() / 1024**3,
                        prog_bar=False, logger=True, on_step=True, batch_size=total_samples)

        # Log detailed loss info
        if batch_idx % self.print_freq == 0 and self.trainer.is_global_zero:
            logging.info("")
            logging.info(f'{HEADER}Epoch {self.current_epoch} - Batch [{batch_idx}/{len(self.trainer.train_dataloader)}]{ENDC}')
            logging.info(f'{GREEN}{"Loss:":<21s} {loss.item():.4f}{ENDC}')
            for k, v in loss_dict.items():
                logging.info(f'{BLUE}{k.title() + ":":<21s} {v.item():.4f}{ENDC}')

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """Validation step using flow matching sampling."""
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)

        # Sample using flow matching (returns trajectory)
        pred_trajectory = self.sample(batch)  # (B, k=1, num_steps+1, num_grasps, D)
        pred_x0 = pred_trajectory[:, 0, -1]  # Take last timestep of first sample: (B, num_grasps, D)

        pred_dict = build_pred_dict_adaptive(pred_x0)

        B, num_grasps, _ = pred_x0.shape
        batch_size = B * num_grasps

        loss_dict = self.criterion(pred_dict, batch, mode='val')

        # Handle standardized validation loss (same as DDPM)
        if hasattr(self.criterion, 'use_standardized_val_loss') and self.criterion.use_standardized_val_loss:
            standard_loss_dict = loss_dict.pop('_standard_losses', {})
            std_weights = getattr(self.criterion, 'std_val_weights', {})
            loss = sum(v * std_weights.get(k, 0) for k, v in loss_dict.items()
                      if k in std_weights and std_weights[k] > 0)
            standard_loss = sum(v * self.loss_weights[k] for k, v in standard_loss_dict.items()
                               if k in self.loss_weights)

            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("val/standardized_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("val/standard_loss", standard_loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            self.validation_step_outputs.append({
                "loss": loss.item(),
                "standardized_loss": loss.item(),
                "standard_loss": standard_loss.item(),
                "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                "standard_loss_dict": {k: v.item() for k, v in standard_loss_dict.items()}
            })
        else:
            loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)
            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            self.validation_step_outputs.append({
                "loss": loss.item(),
                "loss_dict": {k: v.item() for k, v in loss_dict.items()}
            })

        return {"loss": loss, "loss_dict": loss_dict}

    def on_validation_epoch_end(self):
        """Validation epoch end (same as DDPM)."""
        val_loss = [x["loss"] for x in self.validation_step_outputs]
        avg_loss = mean(val_loss)

        using_standardized = (hasattr(self.criterion, 'use_standardized_val_loss') and
                             self.criterion.use_standardized_val_loss and
                             len(self.validation_step_outputs) > 0 and
                             "standardized_loss" in self.validation_step_outputs[0])

        if using_standardized:
            std_loss = [x["standardized_loss"] for x in self.validation_step_outputs]
            standard_loss = [x["standard_loss"] for x in self.validation_step_outputs]
            avg_std_loss = mean(std_loss)
            avg_standard_loss = mean(standard_loss)

            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean([x["loss_dict"][k] for x in self.validation_step_outputs])

            standard_detailed_loss = {}
            if self.validation_step_outputs and "standard_loss_dict" in self.validation_step_outputs[0]:
                for k in self.validation_step_outputs[0]["standard_loss_dict"].keys():
                    standard_detailed_loss[k] = mean([x["standard_loss_dict"][k] for x in self.validation_step_outputs])

            std_loss_std = torch.std(torch.tensor(std_loss)).item() if len(std_loss) > 1 else 0.0
            std_loss_min = min(std_loss) if std_loss else 0.0
            std_loss_max = max(std_loss) if std_loss else 0.0

            log_validation_summary(
                epoch=self.current_epoch, num_batches=len(self.validation_step_outputs),
                avg_loss=avg_std_loss, loss_std=std_loss_std,
                loss_min=std_loss_min, loss_max=std_loss_max,
                val_detailed_loss=val_detailed_loss
            )

            val_log_dict = {f"val/std_{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update({
                "val/standardized_total_loss": avg_std_loss,
                "val/std_loss_std": std_loss_std,
                "val/std_loss_min": std_loss_min,
                "val/std_loss_max": std_loss_max
            })

            standard_log_dict = {f"val/orig_{k}": v for k, v in standard_detailed_loss.items()}
            standard_log_dict.update({"val/original_total_loss": avg_standard_loss})
            val_log_dict.update(standard_log_dict)

            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True,
                         batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss', avg_std_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss_standardized', avg_std_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss_original', avg_standard_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)
        else:
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean([x["loss_dict"][k] for x in self.validation_step_outputs])

            loss_std = torch.std(torch.tensor(val_loss)).item() if len(val_loss) > 1 else 0.0
            loss_min = min(val_loss) if val_loss else 0.0
            loss_max = max(val_loss) if val_loss else 0.0

            log_validation_summary(
                epoch=self.current_epoch, num_batches=len(self.validation_step_outputs),
                avg_loss=avg_loss, loss_std=loss_std, loss_min=loss_min, loss_max=loss_max,
                val_detailed_loss=val_detailed_loss
            )

            val_log_dict = {f"val/{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update({
                "val/total_loss": avg_loss, "val/loss_std": loss_std,
                "val/loss_min": loss_min, "val/loss_max": loss_max
            })
            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True,
                         batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss', avg_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        self.log('val/epoch', float(self.current_epoch), prog_bar=False, logger=True,
                on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('val/num_batches', float(len(self.validation_step_outputs)), prog_bar=False,
                logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        if hasattr(self, 'lr_schedulers') and self.lr_schedulers():
            current_lr = (self.lr_schedulers().get_last_lr()[0] if hasattr(self.lr_schedulers(), 'get_last_lr')
                         else self.optimizers().param_groups[0]['lr'])
            self.log('val/lr', current_lr, prog_bar=False, logger=True, on_epoch=True,
                    batch_size=self.batch_size, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.metric_results = []

    def test_step(self, batch, batch_idx):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_trajectory = self.sample(batch)
        pred_x0 = pred_trajectory[:, 0, -1]  # (B, num_grasps, D)

        pred_dict = build_pred_dict_adaptive(pred_x0)
        B, num_grasps, _ = pred_x0.shape

        metric_dict, metric_details = self.criterion(pred_dict, batch, mode='test')
        self.metric_results.append(metric_details)
        return metric_dict

    # ========================
    # Optimizer Configuration
    # ========================

    def configure_optimizers(self):
        """Configure optimizer and scheduler (same as DDPM)."""
        optimizer_name = self.optimizer_cfg.name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr,
                                        weight_decay=self.optimizer_cfg.weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr,
                                         weight_decay=self.optimizer_cfg.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer not supported: {self.optimizer_cfg.name}")

        self.last_epoch = self.current_epoch - 1 if self.current_epoch else -1
        logging.info(f"Setting scheduler last_epoch to: {self.last_epoch}")

        scheduler_name = self.scheduler.name.lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.scheduler.t_max, eta_min=self.scheduler.min_lr, last_epoch=self.last_epoch)
        elif scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.scheduler.step_size, gamma=self.scheduler.step_gamma, last_epoch=self.last_epoch)
        else:
            raise NotImplementedError(f"Scheduler not supported: {self.scheduler.name}")

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # =====================
    # Inference Methods
    # =====================

    def forward_infer(self, data: Dict, k=1, timestep=-1):
        """Inference with k samples at a specified timestep."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_trajectory = self.sample(data, k=k)
        pred_x0 = pred_trajectory[:, :, timestep]  # (B, k, num_grasps, D)
        pred_dict = build_pred_dict_adaptive(pred_x0)
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(pred_dict, data)
        return preds_hand, targets_hand

    def forward_get_pose(self, data: Dict, k=1):
        """Get pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_trajectory = self.sample(data, k=k)
        pred_x0 = pred_trajectory[:, :, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        outputs, targets = self.criterion.infer_norm_process_dict_get_pose(pred_dict, data)
        return outputs, targets

    def forward_get_pose_raw(self, data: Dict, k=1):
        """Get raw, denormalized pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_trajectory = self.sample(data, k=k)
        pred_pose_norm = pred_trajectory[:, :, -1]
        hand_model_pose = denorm_hand_pose_robust(pred_pose_norm, self.rot_type, self.mode)
        return hand_model_pose

    # =====================
    # Helper Methods
    # =====================

    def _compute_flow_matching_loss(
        self,
        batch: Dict,
        mode: str = 'train'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """
        Compute flow matching loss.

        Flow matching objective:
            L = E_{t, x_0, x_1} [||v_theta(x_t, t) - (x_1 - x_0)||^2]

        Args:
            batch: Input batch
            mode: 'train' or 'val'

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
            processed_batch: Processed batch data
        """
        # Process hand pose
        processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        norm_pose = processed_batch['norm_pose']  # x_1 (data)
        B = norm_pose.shape[0]

        # Sample timesteps
        t = self.sample_time(B)  # Continuous time in [0, 1]

        # Sample conditional flow: x_t and target velocity u_t
        x_t, u_t = self.sample_conditional_flow(norm_pose, t)

        # Compute conditioning features
        condition_dict = self.velocity_model.condition(processed_batch)
        processed_batch.update(condition_dict)

        # Predict velocity
        pred_velocity = self.velocity_model(x_t, t, processed_batch)

        # Flow matching loss (velocity matching)
        velocity_loss = self.compute_velocity_loss(pred_velocity, u_t)

        # Build prediction dict for compatibility with existing loss functions
        # We need to convert velocity prediction to pose prediction for the criterion
        pred_x0 = self._velocity_to_pose(x_t, pred_velocity, t)
        pred_dict = {"pred_pose_norm": pred_x0}

        if self.use_negative_prompts and 'neg_pred' in condition_dict and condition_dict['neg_pred'] is not None:
            pred_dict['neg_pred'] = condition_dict['neg_pred']
            pred_dict['neg_text_features'] = condition_dict['neg_text_features']

        # Compute additional losses using the criterion
        loss_dict = self.criterion(pred_dict, processed_batch, mode=mode)

        # Add velocity matching loss
        loss_dict['velocity_loss'] = velocity_loss

        # Total loss
        loss = sum(v * self.loss_weights.get(k, 0.0) for k, v in loss_dict.items() if k in self.loss_weights)
        # Add velocity loss with weight 1.0 if not in loss_weights
        if 'velocity_loss' not in self.loss_weights:
            loss = loss + velocity_loss

        return loss, loss_dict, processed_batch

    def _velocity_to_pose(self, x_t: torch.Tensor, v_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert velocity prediction to pose prediction for loss computation.

        For OT flow: x_t = (1-t)*x_0 + t*x_1
        Given: v_t ≈ x_1 - x_0
        We can estimate: x_1 ≈ x_t + (1-t)*v_t

        Args:
            x_t: (B, num_grasps, D) - Current state
            v_t: (B, num_grasps, D) - Predicted velocity
            t: (B,) - Timesteps

        Returns:
            pred_x1: (B, num_grasps, D) - Predicted final pose
        """
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
            v_t = v_t.unsqueeze(1)

        B, num_grasps, D = x_t.shape
        t_expanded = t.view(B, 1, 1).expand(B, num_grasps, 1)

        # Estimate x_1 from x_t and velocity
        pred_x1 = x_t + (1 - t_expanded) * v_t

        if input_dim == 2:
            pred_x1 = pred_x1.squeeze(1)

        return pred_x1

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Ensure text encoder is initialized before loading checkpoint."""
        try:
            if (hasattr(self, 'velocity_model') and
                hasattr(self.velocity_model, '_ensure_text_encoder') and
                hasattr(self.velocity_model, 'use_text_condition') and
                self.velocity_model.use_text_condition):
                self.velocity_model._ensure_text_encoder()
                logging.info("Text encoder initialized for checkpoint loading")
        except Exception as e:
            logging.warning(f"Failed to ensure text encoder before loading checkpoint: {e}")
