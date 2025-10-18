import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from models.decoder import build_decoder
from models.loss import GraspLossPose
from models.utils.diffusion_utils import make_schedule_ddpm
from utils.hand_helper import process_hand_pose, process_hand_pose_test, denorm_hand_pose_robust
from statistics import mean
import logging
from .utils.log_colors import HEADER, BLUE, GREEN, YELLOW, RED, ENDC, BOLD, UNDERLINE
from .utils.diffusion_core import DiffusionCoreMixin
from .utils.prediction import build_pred_dict_adaptive
from .utils.logging_helpers import log_validation_summary, convert_number_to_emoji

class DDPMLightning(DiffusionCoreMixin, pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        logging.info(f"{GREEN}Initializing DDPMLightning model{ENDC}")
        self.save_hyperparameters()
        
        self.use_negative_prompts = cfg.get('use_negative_prompts', True)
        
        self.eps_model = build_decoder(cfg.decoder, cfg)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.rot_type = cfg.rot_type
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        self.use_score = cfg.use_score
        self.score_pretrain = cfg.score_pretrain
        
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.pred_x0 = cfg.pred_x0
        self.mode = cfg.mode
        
        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
            
        self.optimizer_cfg = cfg.optimizer
        self.scheduler = cfg.scheduler

        self.use_cfg = cfg.use_cfg
        self.guidance_scale = cfg.guidance_scale
        self.use_negative_guidance = cfg.use_negative_guidance and self.use_negative_prompts
        self.negative_guidance_scale = cfg.negative_guidance_scale
        
        # Grasp count control (diffusion-level configuration)
        self.fix_num_grasps = cfg.get('fix_num_grasps', False)
        self.target_num_grasps = cfg.get('target_num_grasps', None)

        # WandB optimization parameters
        wandb_opt = cfg.get('wandb_optimization', {})
        self._log_gradients = wandb_opt.get('log_gradients', False)
        self._gradient_freq = wandb_opt.get('gradient_freq', 1000)
        self._monitor_system = wandb_opt.get('monitor_system', False)
        self._system_freq = wandb_opt.get('system_freq', 500)

    # =====================
    # PyTorch Lightning Hooks
    # =====================

    def training_step(self, batch, batch_idx):
        """Training step: supports single and multi-grasp parallel training."""
        loss, loss_dict, processed_batch = self._compute_loss(batch, mode='train')

        # Calculate total samples based on processed batch shape
        norm_pose = processed_batch['norm_pose']
        B, num_grasps, _ = norm_pose.shape
        total_samples = B * num_grasps

        # Log training losses with "train/" prefix for WandB organization
        train_log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        train_log_dict["train/total_loss"] = loss
        self.log_dict(train_log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=total_samples, sync_dist=True)

        # Log learning rate
        optimizer = self.optimizers()
        self.log("train/lr", optimizer.param_groups[0]['lr'], prog_bar=False, logger=True, on_step=True, batch_size=total_samples, sync_dist=True)

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
                self.log("train/grad_norm", total_norm, prog_bar=False, logger=True, on_step=True, batch_size=total_samples, sync_dist=True)

        # Log system resource usage periodically
        if self._monitor_system and batch_idx % self._system_freq == 0:
            if torch.cuda.is_available():
                self.log("system/gpu_memory_allocated_gb", torch.cuda.memory_allocated() / 1024**3, prog_bar=False, logger=True, on_step=True, batch_size=total_samples)
                self.log("system/gpu_memory_reserved_gb", torch.cuda.memory_reserved() / 1024**3, prog_bar=False, logger=True, on_step=True, batch_size=total_samples)

            # Log training speed (samples/sec)
            if hasattr(self, '_last_log_time'):
                import time
                current_time = time.time()
                time_diff = current_time - self._last_log_time
                if time_diff > 0:
                    self.log("system/samples_per_sec", total_samples / time_diff, prog_bar=False, logger=True, on_step=True, batch_size=total_samples)
            import time
            self._last_log_time = time.time()

        # Log detailed loss info on the main process
        if batch_idx % self.print_freq == 0 and self.trainer.is_global_zero:
            empty_formatter = logging.Formatter('')
            root_logger = logging.getLogger()
            original_formatters = [handler.formatter for handler in root_logger.handlers]

            for handler in root_logger.handlers:
                handler.setFormatter(empty_formatter)
            logging.info("")
            for handler, formatter in zip(root_logger.handlers, original_formatters):
                handler.setFormatter(formatter)

            logging.info(f'{HEADER}Epoch {self.current_epoch} - Batch [{batch_idx}/{len(self.trainer.train_dataloader)}]{ENDC}')
            logging.info(f'{GREEN}{"Loss:":<21s} {loss.item():.4f}{ENDC}')
            for k, v in loss_dict.items():
                logging.info(f'{BLUE}{k.title() + ":":<21s} {v.item():.4f}{ENDC}')

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        """Validation step: supports multi-grasp parallel inference."""
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)

        B, num_grasps, _ = pred_x0.shape
        batch_size = B * num_grasps

        loss_dict = self.criterion(pred_dict, batch, mode='val')

        # Handle standardized validation loss
        if hasattr(self.criterion, 'use_standardized_val_loss') and self.criterion.use_standardized_val_loss:
            # Extract standard losses for logging
            standard_loss_dict = loss_dict.pop('_standard_losses', {})

            # Calculate standardized total loss
            std_weights = getattr(self.criterion, 'std_val_weights', {})
            loss = sum(v * std_weights.get(k, 0) for k, v in loss_dict.items()
                      if k in std_weights and std_weights[k] > 0)

            # Calculate standard total loss for comparison
            standard_loss = sum(v * self.loss_weights[k] for k, v in standard_loss_dict.items()
                               if k in self.loss_weights)

            # Log both losses
            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("val/standardized_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("val/standard_loss", standard_loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            # Store both for epoch-end processing
            self.validation_step_outputs.append({
                "loss": loss.item(),
                "standardized_loss": loss.item(),
                "standard_loss": standard_loss.item(),
                "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                "standard_loss_dict": {k: v.item() for k, v in standard_loss_dict.items()}
            })
        else:
            # Standard validation loss calculation
            loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)
            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            self.validation_step_outputs.append({
                "loss": loss.item(),
                "loss_dict": {k: v.item() for k, v in loss_dict.items()}
            })

        return {"loss": loss, "loss_dict": loss_dict}
        
    def on_validation_epoch_end(self):
        val_loss = [x["loss"] for x in self.validation_step_outputs]
        avg_loss = mean(val_loss)

        # Check if we're using standardized validation loss
        using_standardized = (hasattr(self.criterion, 'use_standardized_val_loss') and
                             self.criterion.use_standardized_val_loss and
                             len(self.validation_step_outputs) > 0 and
                             "standardized_loss" in self.validation_step_outputs[0])

        if using_standardized:
            # Process both standardized and standard losses
            std_loss = [x["standardized_loss"] for x in self.validation_step_outputs]
            standard_loss = [x["standard_loss"] for x in self.validation_step_outputs]

            avg_std_loss = mean(std_loss)
            avg_standard_loss = mean(standard_loss)

            # Detailed losses for standardized
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean([x["loss_dict"][k] for x in self.validation_step_outputs])

            # Detailed losses for standard (for comparison)
            standard_detailed_loss = {}
            if self.validation_step_outputs and "standard_loss_dict" in self.validation_step_outputs[0]:
                for k in self.validation_step_outputs[0]["standard_loss_dict"].keys():
                    standard_detailed_loss[k] = mean([x["standard_loss_dict"][k] for x in self.validation_step_outputs])

            # Statistics for standardized loss
            std_loss_std = torch.std(torch.tensor(std_loss)).item() if len(std_loss) > 1 else 0.0
            std_loss_min = min(std_loss) if std_loss else 0.0
            std_loss_max = max(std_loss) if std_loss else 0.0

            # Log validation summary with standardized loss
            log_validation_summary(
                epoch=self.current_epoch, num_batches=len(self.validation_step_outputs),
                avg_loss=avg_std_loss, loss_std=std_loss_std,
                loss_min=std_loss_min, loss_max=std_loss_max,
                val_detailed_loss=val_detailed_loss
            )

            # Log standardized validation losses to WandB
            val_log_dict = {f"val/std_{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update({
                "val/standardized_total_loss": avg_std_loss,
                "val/std_loss_std": std_loss_std,
                "val/std_loss_min": std_loss_min,
                "val/std_loss_max": std_loss_max
            })

            # Log standard losses for comparison
            standard_log_dict = {f"val/orig_{k}": v for k, v in standard_detailed_loss.items()}
            standard_log_dict.update({
                "val/original_total_loss": avg_standard_loss
            })
            val_log_dict.update(standard_log_dict)

            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

            # Use standardized loss for ModelCheckpoint (this is the key change!)
            self.log('val_loss', avg_std_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

            # Also log both for manual comparison
            self.log('val_loss_standardized', avg_std_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss_original', avg_standard_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)

        else:
            # Standard validation loss processing
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean([x["loss_dict"][k] for x in self.validation_step_outputs])

            num_batches = len(self.validation_step_outputs)
            loss_std = torch.std(torch.tensor(val_loss)).item() if len(val_loss) > 1 else 0.0
            loss_min = min(val_loss) if val_loss else 0.0
            loss_max = max(val_loss) if val_loss else 0.0

            log_validation_summary(
                epoch=self.current_epoch, num_batches=num_batches, avg_loss=avg_loss,
                loss_std=loss_std, loss_min=loss_min, loss_max=loss_max,
                val_detailed_loss=val_detailed_loss
            )

            # Log validation losses to WandB with "val/" prefix
            val_log_dict = {f"val/{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update({
                "val/total_loss": avg_loss, "val/loss_std": loss_std,
                "val/loss_min": loss_min, "val/loss_max": loss_max
            })
            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

            # Log for ModelCheckpoint
            self.log('val_loss', avg_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        # Log additional validation info (common for both modes)
        self.log('val/epoch', float(self.current_epoch), prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)
        self.log('val/num_batches', float(len(self.validation_step_outputs)), prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        # Log current learning rate if scheduler exists
        if hasattr(self, 'lr_schedulers') and self.lr_schedulers():
            current_lr = self.lr_schedulers().get_last_lr()[0] if hasattr(self.lr_schedulers(), 'get_last_lr') else self.optimizers().param_groups[0]['lr']
            self.log('val/lr', current_lr, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.metric_results = []

    def test_step(self, batch, batch_idx):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:, 0, -1]  # Take last timestep of first sample
        
        pred_dict = build_pred_dict_adaptive(pred_x0)
        B, num_grasps, _ = pred_x0.shape
        
        metric_dict, metric_details = self.criterion(pred_dict, batch, mode='test')
        self.metric_results.append(metric_details)
        return metric_dict

    def test_step_teaser(self, batch):
        B = self.batch_size
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:,0,-1]
        
        pred_dict = build_pred_dict_adaptive(pred_x0)
        metric_dict, metric_details = self.criterion.forward_metric(pred_dict, batch)
        return metric_dict, metric_details

    # ========================
    # Optimizer Configuration
    # ========================

    def configure_optimizers(self):
        optimizer_name = self.optimizer_cfg.name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay)
        else:
            raise NotImplementedError(f"Optimizer not supported: {self.optimizer_cfg.name}")
        
        self.last_epoch = self.current_epoch - 1 if self.current_epoch else -1
        if hasattr(self, 'use_score') and self.use_score and not self.score_pretrain:
            logging.info("Using score model without pretraining; optimizer state will not be loaded.")
            self.trainer.fit_loop.epoch_progress.current.completed = 0
            self.last_epoch = -1
        
        logging.info(f"Setting scheduler last_epoch to: {self.last_epoch}")
        
        scheduler_name = self.scheduler.name.lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.scheduler.t_max, eta_min=self.scheduler.min_lr, last_epoch=self.last_epoch)
        elif scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.scheduler.step_size, gamma=self.scheduler.step_gamma, last_epoch=self.last_epoch)
        else:
            raise NotImplementedError(f"Scheduler not supported: {self.scheduler.name}")
            
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # =====================
    # Inference Methods
    # =====================

    def forward_infer(self, data: Dict, k=1, timestep=-1):
        """Inference with k samples at a specified timestep."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, :, timestep]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(pred_dict, data)
        return preds_hand, targets_hand
    
    def forward_infer_step(self, data: Dict, k=1, timesteps=[1, 3, 5, 7, 9, 11, 91, 93, 95, 97, -1]):
        """Inference with k samples at multiple timesteps."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)
        results = []
        for timestep in timesteps:
            pred_x0_t = pred_x0[:, :, timestep]
            pred_dict = build_pred_dict_adaptive(pred_x0_t)
            preds_hand, _ = self.criterion.infer_norm_process_dict(pred_dict, data)
            results.append(preds_hand)
        return results

    def forward_get_pose(self, data: Dict, k=1):
        """Get pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, :, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        outputs, targets = self.criterion.infer_norm_process_dict_get_pose(pred_dict, data)
        return outputs, targets

    def forward_get_pose_matched(self, data: Dict, k=1):
        """Get matched pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        matched_preds, matched_targets, outputs, targets = self.criterion.forward_infer(pred_dict, data)
        return matched_preds, matched_targets, outputs, targets

    def forward_get_pose_raw(self, data: Dict, k=1):
        """Get raw, denormalized pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_pose_norm = self.sample(data, k=k)[:, :, -1]
        hand_model_pose = denorm_hand_pose_robust(pred_pose_norm, self.rot_type, self.mode)
        return hand_model_pose

    def forward_train_instance(self, data: Dict):
        """Forward pass for training a single instance."""
        data = process_hand_pose(data, rot_type=self.rot_type, mode=self.mode)
        B = data['norm_pose'].shape[0]
        
        ts = self._sample_timesteps(B)
        noise = torch.randn_like(data['norm_pose'], device=self.device)
        x_t = self.q_sample(x0=data['norm_pose'], t=ts, noise=noise)

        condition = self.eps_model.condition(data)
        data["cond"] = condition
        output = self.eps_model(x_t, ts, data)

        if self.pred_x0:
            pred_dict = {"pred_pose_norm": output, "noise": noise}
        else:
            pred_x0 = self._compute_pred_x0_from_noise(x_t, ts, output)
            pred_dict = {"pred_noise": output, "pred_pose_norm": pred_x0, "noise": noise}

        outputs, targets = self.criterion.get_hand_model_pose(pred_dict, data)
        outputs, targets = self.criterion.get_hand(outputs, targets)
        return outputs, targets

    # =====================
    # Helper Methods
    # =====================

    def _compute_loss(self, batch: Dict, mode: str = 'train') -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Unified loss computation logic for training and validation."""
        processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        norm_pose = processed_batch['norm_pose']
        B = norm_pose.shape[0]

        ts = self._sample_timesteps(B)
        noise = torch.randn_like(norm_pose, device=self.device)
        x_t = self.q_sample(x0=norm_pose, t=ts, noise=noise)

        condition_dict = self.eps_model.condition(processed_batch)
        processed_batch.update(condition_dict)

        output = self.eps_model(x_t, ts, processed_batch)

        if self.pred_x0:
            pred_dict = {"pred_pose_norm": output, "noise": noise}
        else:
            pred_x0 = self._compute_pred_x0_from_noise(x_t, ts, output)
            pred_dict = {"pred_noise": output, "pred_pose_norm": pred_x0, "noise": noise}

        if self.use_negative_prompts and 'neg_pred' in condition_dict and condition_dict['neg_pred'] is not None:
            pred_dict['neg_pred'] = condition_dict['neg_pred']
            pred_dict['neg_text_features'] = condition_dict['neg_text_features']

        loss_dict = self.criterion(pred_dict, processed_batch, mode=mode)
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)

        return loss, loss_dict, processed_batch

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Ensure lazily-initialized submodules (e.g., text encoder) exist BEFORE Lightning
        restores the state_dict, otherwise keys under `eps_model.text_encoder.*` will be
        reported as Unexpected.
        Also keep the original special handling for `score_heads`.
        """
        # 1) Make sure the text encoder is constructed so its parameters are present
        # Only initialize if text conditioning is enabled
        try:
            if (hasattr(self, 'eps_model') and 
                hasattr(self.eps_model, '_ensure_text_encoder') and
                hasattr(self.eps_model, 'use_text_condition') and
                self.eps_model.use_text_condition):
                self.eps_model._ensure_text_encoder()
                logging.info("Text encoder initialized for checkpoint loading")
            elif hasattr(self, 'eps_model') and hasattr(self.eps_model, 'use_text_condition'):
                logging.info(f"Skipping text encoder initialization (use_text_condition={self.eps_model.use_text_condition})")
        except Exception as e:
            logging.warning(f"Failed to ensure text encoder before loading checkpoint: {e}")

        # 2) Handle optional score model loading behavior
        if hasattr(self, 'use_score') and self.use_score and not self.score_pretrain:
            model_state_dict = self.state_dict()
            checkpoint_state_dict = checkpoint['state_dict']

            # Load all weights except for score_heads
            new_state_dict = {k: v for k, v in checkpoint_state_dict.items() if 'score_heads' not in k}
            model_state_dict.update(new_state_dict)
            self.load_state_dict(model_state_dict)
            logging.info("Loaded checkpoint weights, excluding score_heads.")
        else:
            # Standard checkpoint loading (Lightning will proceed with strict=True)
            logging.info("Loading full checkpoint.")

    def _check_state_dict(self, dict1, dict2):
        """Helper to check for inconsistencies between two state dicts."""
        keys1, keys2 = set(dict1.keys()), set(dict2.keys())
        if keys1 != keys2:
            logging.warning("State dict key mismatch!")
            logging.warning(f"Only in checkpoint: {keys1 - keys2}")
            logging.warning(f"Only in model: {keys2 - keys1}")
            return True
        
        for key in keys1:
            if dict1[key].shape != dict2[key].shape:
                logging.warning(f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}")
                return True
        return False

    def _build_pred_dict_adaptive(self, pred_x0):
        """(Deprecated) Backward compatibility wrapper for build_pred_dict_adaptive."""
        logging.warning("_build_pred_dict_adaptive is deprecated; use build_pred_dict_adaptive from models.utils.prediction instead.")
        return build_pred_dict_adaptive(pred_x0)