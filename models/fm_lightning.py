"""
Flow Matching Lightning Module

This module implements the Flow Matching training and sampling procedures
using PyTorch Lightning, following the paradigm from "Flow Matching for 
Generative Modeling" (Lipman et al., 2023).
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import logging
import math
from statistics import mean

# Import model building
from models.decoder import build_decoder

# Import loss and utilities
from models.loss import GraspLossPose
from utils.hand_helper import process_hand_pose, process_hand_pose_test, denorm_hand_pose_robust
from models.utils.prediction import build_pred_dict_adaptive
from models.utils.logging_helpers import log_validation_summary
from models.utils.log_colors import HEADER, BLUE, GREEN, YELLOW, RED, ENDC, BOLD


class FlowMatchingLightning(pl.LightningModule):
    """
    Flow Matching training module for grasp synthesis.
    
    Key differences from DDPM:
    - Continuous time t ∈ [0, 1] instead of discrete timesteps
    - Predict velocity field v(x, t) instead of noise ε
    - Use ODE solvers instead of SDE sampling
    - Linear optimal transport path by default
    """
    
    def __init__(self, cfg):
        super().__init__()
        logging.info(f"{GREEN}Initializing FlowMatchingLightning model{ENDC}")
        self.save_hyperparameters()
        
        # Build model (DiTFM)
        self.model = build_decoder(cfg.decoder, cfg)
        
        # Loss function
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        
        # Basic configs
        self.rot_type = cfg.rot_type
        self.mode = cfg.mode
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        
        # Flow Matching specific configs
        self.fm_cfg = cfg.fm
        self.path_type = self.fm_cfg.get('path', 'linear_ot')
        self.t_sampler = self.fm_cfg.get('t_sampler', 'uniform')
        self.t_weight = self.fm_cfg.get('t_weight', None)
        self.continuous_time = self.fm_cfg.get('continuous_time', True)
        
        # Solver configs
        self.solver_cfg = cfg.solver
        self.solver_type = self.solver_cfg.get('type', 'rk4')
        self.nfe = self.solver_cfg.get('nfe', 32)
        
        # Guidance configs
        self.guidance_cfg = cfg.guidance
        self.use_cfg = self.guidance_cfg.get('enable_cfg', False)
        self.cond_drop_prob = self.guidance_cfg.get('cond_drop_prob', 0.1)
        self.guidance_scale = self.guidance_cfg.get('scale', 3.0)
        self.diff_clip = self.guidance_cfg.get('diff_clip', 5.0)
        self.use_pc_correction = self.guidance_cfg.get('pc_correction', False)
        
        # Optimizer and scheduler configs
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        
        # Debug configs
        self.debug_cfg = cfg.get('debug', {})
        self.check_nan = self.debug_cfg.get('check_nan', True)
        self.log_tensor_stats = self.debug_cfg.get('log_tensor_stats', False)
        
        # Compatibility configs
        self.compat_cfg = cfg.get('compat', {})
        self.keep_ddpm_interface = self.compat_cfg.get('keep_ddpm_interface', True)
        
        # Grasp count control
        self.fix_num_grasps = cfg.get('fix_num_grasps', False)
        self.target_num_grasps = cfg.get('target_num_grasps', None)
        
        # Training statistics
        self._training_step_outputs = []
        
        logging.info(f"Flow Matching initialized with path: {self.path_type}, "
                    f"solver: {self.solver_type}, NFE: {self.nfe}")
    
    # =====================
    # Training
    # =====================
    
    def training_step(self, batch, batch_idx):
        """
        Training step for Flow Matching.
        
        Key differences from DDPM:
        1. Sample continuous time t ∈ [0, 1]
        2. Use linear interpolation x_t = (1-t)·x0 + t·x1
        3. Target velocity v* = x1 - x0 (constant for linear OT)
        4. Predict velocity field v_θ(x_t, t)
        """
        # Process hand pose to normalized space
        processed_batch = process_hand_pose(batch, self.rot_type, self.mode)
        x0 = processed_batch['norm_pose']  # [B, num_grasps, D]
        
        B, num_grasps, D = x0.shape
        
        # Sample continuous time
        t = self._sample_time(B)  # [B], values in [0, 1]
        
        # Sample noise endpoint x1 ~ N(0, I)
        x1 = torch.randn_like(x0, device=self.device)
        
        # Compute interpolation and target velocity
        if self.path_type == 'linear_ot':
            # Linear optimal transport path
            x_t, v_star = self._linear_ot_path(x0, x1, t)
        else:
            raise NotImplementedError(f"Path type {self.path_type} not implemented")
        
        # Check for NaN
        if self.check_nan:
            if torch.isnan(x_t).any():
                logging.error("NaN detected in x_t")
                raise RuntimeError("NaN in interpolated state")
            if torch.isnan(v_star).any():
                logging.error("NaN detected in v_star")
                raise RuntimeError("NaN in target velocity")
        
        # Compute conditioning features
        condition_dict = self.model.condition(processed_batch)
        processed_batch.update(condition_dict)
        
        # Apply conditioning dropout for CFG training
        if self.use_cfg and self.training:
            # Randomly drop conditioning with probability cond_drop_prob
            drop_mask = torch.rand(B, device=self.device) < self.cond_drop_prob
            if drop_mask.any():
                # Create unconditioned batch entries
                for key in ['scene_cond', 'text_cond']:
                    if key in processed_batch and processed_batch[key] is not None:
                        # Zero out conditioning for dropped samples
                        mask_expanded = drop_mask.view(B, *([1] * (processed_batch[key].dim() - 1)))
                        processed_batch[key] = processed_batch[key] * (~mask_expanded).float()
        
        # Forward pass: predict velocity
        v_pred = self.model(x_t, t, processed_batch)  # [B, num_grasps, D]
        
        # Compute loss
        if self.t_weight is not None:
            # Apply time weighting
            weight = self._compute_time_weight(t)
            loss = (F.mse_loss(v_pred, v_star, reduction='none') * weight.view(-1, 1, 1)).mean()
        else:
            loss = F.mse_loss(v_pred, v_star)
        
        # For compatibility with existing loss computation, create pred_dict
        # We need to compute pred_x0 from velocity for the criterion
        # From linear OT: x_t = (1-t)*x0 + t*x1, v = x1 - x0
        # Therefore: x0 = x_t - t*v
        t_expanded = t.view(-1, 1, 1)
        pred_x0 = x_t - t_expanded * v_pred
        
        if self.keep_ddpm_interface:
            pred_dict = {
                "pred_pose_norm": pred_x0,  # Predicted x0 for criterion
                "pred_noise": v_pred,  # Velocity prediction  
                "noise": v_star  # Target velocity
            }
            
            # Compute additional losses (optional)
            loss_dict = self.criterion(pred_dict, processed_batch, mode='train')
            
            # Use only MSE loss for now, can integrate other losses later
            total_loss = loss  # Could add: + sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k != 'noise_loss')
        else:
            total_loss = loss
            loss_dict = {'velocity_loss': loss}
        
        # Logging
        total_samples = B * num_grasps
        train_log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        train_log_dict["train/total_loss"] = total_loss
        train_log_dict["train/mean_t"] = t.mean()
        train_log_dict["train/velocity_norm"] = torch.norm(v_pred, dim=-1).mean()
        
        self.log_dict(train_log_dict, prog_bar=False, logger=True, on_step=True, 
                     on_epoch=True, batch_size=total_samples, sync_dist=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        self.log("train/lr", optimizer.param_groups[0]['lr'], prog_bar=False, 
                logger=True, on_step=True, batch_size=total_samples, sync_dist=True)
        
        # Detailed logging
        if batch_idx % self.print_freq == 0 and self.trainer.is_global_zero:
            logging.info(f'{HEADER}Epoch {self.current_epoch} - Batch [{batch_idx}/{len(self.trainer.train_dataloader)}]{ENDC}')
            logging.info(f'{GREEN}{"Loss:":<21s} {total_loss.item():.4f}{ENDC}')
            logging.info(f'{BLUE}{"Mean t:":<21s} {t.mean().item():.4f}{ENDC}')
            logging.info(f'{BLUE}{"||v_pred||:":<21s} {torch.norm(v_pred, dim=-1).mean().item():.4f}{ENDC}')
            logging.info(f'{BLUE}{"||v_star||:":<21s} {torch.norm(v_star, dim=-1).mean().item():.4f}{ENDC}')
        
        return total_loss
    
    # =====================
    # Validation
    # =====================
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
    
    def validation_step(self, batch, batch_idx):
        """Validation using ODE sampling."""
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        
        # Sample using ODE solver
        pred_x0 = self.sample(batch)  # [B, num_grasps, D]
        
        # Build prediction dictionary
        pred_dict = build_pred_dict_adaptive(pred_x0)
        
        B, num_grasps, _ = pred_x0.shape
        batch_size = B * num_grasps
        
        # Compute validation losses
        loss_dict = self.criterion(pred_dict, batch, mode='val')
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() 
                  if k in self.loss_weights and torch.is_tensor(v))
        
        self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
        
        self.validation_step_outputs.append({
            "loss": loss.item(),
            "loss_dict": {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        })
        
        return {"loss": loss, "loss_dict": loss_dict}
    
    def on_validation_epoch_end(self):
        val_loss = [x["loss"] for x in self.validation_step_outputs]
        avg_loss = mean(val_loss) if val_loss else 0.0
        
        # Compute detailed losses
        val_detailed_loss = {}
        if self.validation_step_outputs:
            for k in self.validation_step_outputs[0]["loss_dict"].keys():
                # Handle both scalar and tensor values, skip dicts
                values = [x["loss_dict"][k] for x in self.validation_step_outputs]
                # Only compute mean for numeric values (skip nested dicts)
                if values and not isinstance(values[0], dict):
                    val_detailed_loss[k] = mean(values)
        
        # Log validation summary
        log_validation_summary(
            epoch=self.current_epoch, 
            num_batches=len(self.validation_step_outputs),
            avg_loss=avg_loss,
            loss_std=0.0,
            loss_min=min(val_loss) if val_loss else 0.0,
            loss_max=max(val_loss) if val_loss else 0.0,
            val_detailed_loss=val_detailed_loss
        )
        
        # Log to wandb
        val_log_dict = {f"val/{k}": v for k, v in val_detailed_loss.items()}
        val_log_dict["val/total_loss"] = avg_loss
        self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True, 
                     batch_size=self.batch_size, sync_dist=True)
        
        # Log for checkpoint
        self.log('val_loss', avg_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
        
        self.validation_step_outputs.clear()
    
    # =====================
    # Sampling
    # =====================
    
    @torch.no_grad()
    def sample(self, data: Dict, k: int = 1) -> torch.Tensor:
        """
        Sample from the flow model using ODE integration.
        
        Args:
            data: Batch data with conditioning information
            k: Number of samples to generate (default: 1)
            
        Returns:
            Sampled poses in normalized space [B, num_grasps, D] if k=1
            or [B, k, num_grasps, D] if k>1
        """
        # Get shape from data
        if 'norm_pose' in data:
            norm_pose = data['norm_pose']
            B, num_grasps, D = norm_pose.shape
        else:
            # Fallback shape
            B = len(data['positive_prompt']) if 'positive_prompt' in data else 1
            num_grasps = self.target_num_grasps if self.target_num_grasps else 64
            D = 25 if self.rot_type == 'r6d' else 23
        
        # Pre-compute conditioning
        condition_dict = self.model.condition(data)
        data.update(condition_dict)
        
        # Generate k samples
        samples = []
        for _ in range(k):
            # Initialize from noise: x1 ~ N(0, I)
            x1 = torch.randn(B, num_grasps, D, device=self.device)
            
            # Integrate ODE from t=1 to t=0
            x0 = self._integrate_ode(x1, data)
            samples.append(x0)
        
        if k == 1:
            return samples[0]
        else:
            return torch.stack(samples, dim=1)
    
    def _integrate_ode(self, x1: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        Integrate the ODE dx/dt = v_θ(x, t) from t=1 to t=0.
        
        Args:
            x1: Initial state (noise) [B, num_grasps, D]
            data: Conditioning data
            
        Returns:
            x0: Final state (denoised) [B, num_grasps, D]
        """
        if self.solver_type == 'heun':
            return self._heun_solver(x1, data)
        elif self.solver_type == 'rk4':
            return self._rk4_solver(x1, data)
        else:
            raise NotImplementedError(f"Solver {self.solver_type} not implemented")
    
    def _heun_solver(self, x1: torch.Tensor, data: Dict) -> torch.Tensor:
        """Heun's method (2nd order Runge-Kutta)."""
        dt = 1.0 / self.nfe
        x = x1.clone()
        
        for i in range(self.nfe):
            t = 1.0 - i * dt  # Time from 1 to dt
            t_tensor = torch.full((x.shape[0],), t, device=self.device)
            
            # Predictor step
            v = self._velocity_fn(x, t_tensor, data)
            x_mid = x - 0.5 * dt * v
            
            # Corrector step
            t_mid_tensor = torch.full((x.shape[0],), t - 0.5 * dt, device=self.device)
            v_mid = self._velocity_fn(x_mid, t_mid_tensor, data)
            x = x - dt * v_mid
        
        return x
    
    def _rk4_solver(self, x1: torch.Tensor, data: Dict) -> torch.Tensor:
        """4th order Runge-Kutta method."""
        dt = 1.0 / self.nfe
        x = x1.clone()
        
        for i in range(self.nfe):
            t = 1.0 - i * dt
            t_tensor = torch.full((x.shape[0],), t, device=self.device)
            
            # RK4 steps
            k1 = self._velocity_fn(x, t_tensor, data)
            
            t2_tensor = torch.full((x.shape[0],), t - 0.5 * dt, device=self.device)
            k2 = self._velocity_fn(x - 0.5 * dt * k1, t2_tensor, data)
            k3 = self._velocity_fn(x - 0.5 * dt * k2, t2_tensor, data)
            
            t4_tensor = torch.full((x.shape[0],), t - dt, device=self.device)
            k4 = self._velocity_fn(x - dt * k3, t4_tensor, data)
            
            # Update
            x = x - dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        return x
    
    def _velocity_fn(self, x: torch.Tensor, t: torch.Tensor, data: Dict) -> torch.Tensor:
        """
        Compute velocity field, optionally with CFG.
        
        Args:
            x: Current state [B, num_grasps, D]
            t: Current time [B]
            data: Conditioning data
            
        Returns:
            v: Velocity field [B, num_grasps, D]
        """
        if not self.use_cfg or self.training:
            # No CFG during training or if disabled
            return self.model(x, t, data)
        
        # Classifier-free guidance
        # Compute conditional and unconditional velocities
        v_cond = self.model(x, t, data)
        
        # Create unconditional data
        data_uncond = data.copy()
        for key in ['scene_cond', 'text_cond']:
            if key in data_uncond and data_uncond[key] is not None:
                data_uncond[key] = torch.zeros_like(data_uncond[key])
        
        v_uncond = self.model(x, t, data_uncond)
        
        # Apply CFG with optional clipping
        v = self._apply_cfg(v_cond, v_uncond)
        
        # Optional predictor-corrector
        if self.use_pc_correction:
            v = self._pc_correction(x, t, v, data)
        
        return v
    
    def _apply_cfg(self, v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier-free guidance with stability improvements.
        
        Args:
            v_cond: Conditional velocity [B, num_grasps, D]
            v_uncond: Unconditional velocity [B, num_grasps, D]
            
        Returns:
            v_cfg: Guided velocity [B, num_grasps, D]
        """
        # Compute difference
        diff = v_cond - v_uncond
        
        # Apply norm clipping/rescaling if specified
        if self.diff_clip > 0:
            diff_norm = torch.norm(diff, dim=-1, keepdim=True)
            scale = torch.minimum(
                torch.ones_like(diff_norm),
                self.diff_clip / (diff_norm + 1e-8)
            )
            diff = diff * scale
        
        # Apply guidance
        v_cfg = v_cond + self.guidance_scale * diff
        
        return v_cfg
    
    def _pc_correction(self, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor, 
                      data: Dict) -> torch.Tensor:
        """
        Optional predictor-corrector step for improved accuracy.
        
        Args:
            x: Current state [B, num_grasps, D]
            t: Current time [B]
            v: Predicted velocity [B, num_grasps, D]
            data: Conditioning data
            
        Returns:
            v_corrected: Corrected velocity [B, num_grasps, D]
        """
        # Simple PC: take a small step and re-evaluate
        dt_pc = 0.01  # Small step for correction
        x_pred = x - dt_pc * v
        v_corrected = self.model(x_pred, t - dt_pc, data)
        
        # Average predictor and corrector
        return 0.5 * (v + v_corrected)
    
    # =====================
    # Path and Time Utilities
    # =====================
    
    def _sample_time(self, batch_size: int) -> torch.Tensor:
        """
        Sample time values according to specified distribution.
        
        Args:
            batch_size: Number of samples
            
        Returns:
            t: Time values in [0, 1], shape [batch_size]
        """
        if self.t_sampler == 'uniform':
            t = torch.rand(batch_size, device=self.device)
        elif self.t_sampler == 'cosine':
            # Cosine schedule emphasizes middle timesteps
            u = torch.rand(batch_size, device=self.device)
            t = (torch.acos(1 - 2*u) / math.pi)
        elif self.t_sampler == 'beta':
            # Beta distribution, can be tuned
            alpha, beta = 2.0, 2.0  # Symmetric, emphasizes middle
            dist = torch.distributions.Beta(alpha, beta)
            t = dist.sample((batch_size,)).to(self.device)
        else:
            raise ValueError(f"Unknown time sampler: {self.t_sampler}")
        
        return t
    
    def _compute_time_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-dependent loss weight.
        
        Args:
            t: Time values [B]
            
        Returns:
            weight: Loss weights [B]
        """
        if self.t_weight == 'cosine':
            # Higher weight for middle timesteps
            weight = torch.sin(math.pi * t)
        elif self.t_weight == 'beta':
            # Beta-like weighting
            weight = 4 * t * (1 - t)
        else:
            weight = torch.ones_like(t)
        
        return weight
    
    def _linear_ot_path(self, x0: torch.Tensor, x1: torch.Tensor, 
                        t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear optimal transport path.
        
        Args:
            x0: Start point (data) [B, num_grasps, D]
            x1: End point (noise) [B, num_grasps, D]
            t: Time values [B]
            
        Returns:
            x_t: Interpolated state [B, num_grasps, D]
            v_star: Target velocity [B, num_grasps, D]
        """
        # Expand t for broadcasting
        t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
        
        # Linear interpolation
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Constant velocity for linear path
        v_star = x1 - x0
        
        return x_t, v_star
    
    # =====================
    # Optimizer Configuration
    # =====================
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer_name = self.optimizer_cfg.name.lower()
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.optimizer_cfg.lr, 
                weight_decay=self.optimizer_cfg.weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.optimizer_cfg.lr, 
                weight_decay=self.optimizer_cfg.weight_decay
            )
        else:
            raise NotImplementedError(f"Optimizer not supported: {self.optimizer_cfg.name}")
        
        # Scheduler
        scheduler_name = self.scheduler_cfg.name.lower()
        
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.scheduler_cfg.t_max, 
                eta_min=self.scheduler_cfg.min_lr
            )
        elif scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                self.scheduler_cfg.step_size, 
                gamma=self.scheduler_cfg.step_gamma
            )
        else:
            raise NotImplementedError(f"Scheduler not supported: {self.scheduler_cfg.name}")
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss"
            }
        }
    
    # =====================
    # Inference Methods (for compatibility)
    # =====================
    
    def forward_infer(self, data: Dict, k: int = 1, timestep: int = -1):
        """Inference compatible with existing interface."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)
        
        if k > 1:
            # Take first sample for compatibility
            pred_x0 = pred_x0[:, 0]
        
        pred_dict = build_pred_dict_adaptive(pred_x0)
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(pred_dict, data)
        return preds_hand, targets_hand
    
    def forward_get_pose(self, data: Dict, k: int = 1):
        """Get pose predictions."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)
        
        if k > 1:
            pred_x0 = pred_x0[:, 0]
        
        pred_dict = build_pred_dict_adaptive(pred_x0)
        outputs, targets = self.criterion.infer_norm_process_dict_get_pose(pred_dict, data)
        return outputs, targets
