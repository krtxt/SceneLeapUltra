"""
Flow Matching Lightning Module

This module implements the Flow Matching training and sampling procedures
using PyTorch Lightning, following the paradigm from "Flow Matching for 
Generative Modeling" (Lipman et al., 2023).
"""

import logging
import math
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model building
from models.decoder import build_decoder
from models.fm.guidance import apply_cfg as fm_apply_cfg
from models.fm.guidance import \
    predictor_corrector_step as fm_predictor_corrector_step
from models.fm.paths import add_stochasticity, get_path_fn
# Flow Matching components
from models.fm.solvers import integrate_ode as fm_integrate_ode
# Optimal Transport
from models.fm.optimal_transport import (SinkhornOT, apply_optimal_matching,
                                         compute_matching_quality)
# Import loss and utilities
from models.loss import GraspLossPose
from models.utils.log_colors import (BLUE, BOLD, ENDC, GREEN, HEADER, RED,
                                     YELLOW)
from models.utils.logging_helpers import log_validation_summary
from models.utils.prediction import build_pred_dict_adaptive
from utils.hand_helper import (denorm_hand_pose_robust, process_hand_pose,
                               process_hand_pose_test)


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
        self.noise_dist = self.fm_cfg.get('noise_dist', 'normal')
        self.sfm_sigma = self.fm_cfg.get('sfm', {}).get('sigma', 0.0)

        # Path function (fallback to linear_ot when unknown)
        try:
            self.path_fn = get_path_fn(self.path_type)
        except Exception as e:
            logging.warning(f"Unknown path type '{self.path_type}', falling back to 'linear_ot': {e}")
            self.path_type = 'linear_ot'
            self.path_fn = get_path_fn(self.path_type)
        
        # Solver configs
        self.solver_cfg = cfg.solver
        self.solver_type = self.solver_cfg.get('type', 'rk4')
        self.nfe = self.solver_cfg.get('nfe', 32)
        self.rtol = self.solver_cfg.get('rtol', 1e-3)
        self.atol = self.solver_cfg.get('atol', 1e-5)
        self.max_step = self.solver_cfg.get('max_step', 0.03125)
        self.min_step = self.solver_cfg.get('min_step', 1e-4)
        self.reverse_time = self.solver_cfg.get('reverse_time', True)
        self.save_trajectories = self.solver_cfg.get('save_trajectories', False)
        
        # Guidance configs
        self.guidance_cfg = cfg.guidance
        self.use_cfg = self.guidance_cfg.get('enable_cfg', False)
        self.cond_drop_prob = self.guidance_cfg.get('cond_drop_prob', 0.1)
        self.guidance_scale = self.guidance_cfg.get('scale', 3.0)
        self.diff_clip = self.guidance_cfg.get('diff_clip', 5.0)
        self.guidance_method = self.guidance_cfg.get('method', 'clipped')
        self.early_steps_scale = self.guidance_cfg.get('early_steps_scale', 0.0)
        self.late_steps_scale = self.guidance_cfg.get('late_steps_scale', 1.0)
        self.transition_point = self.guidance_cfg.get('transition_point', 0.5)
        self.use_pc_correction = self.guidance_cfg.get('pc_correction', False)
        self.dt_correction = self.guidance_cfg.get('dt_correction', 0.01)
        self.num_corrections = self.guidance_cfg.get('num_corrections', 1)
        
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
        
        # Optimal Transport configs
        self.ot_cfg = self.fm_cfg.get('optimal_transport', {})
        self.use_optimal_matching = self.ot_cfg.get('enable', False)
        self.ot_reg = self.ot_cfg.get('reg', 0.1)
        self.ot_iters = self.ot_cfg.get('num_iters', 50)
        self.ot_distance_metric = self.ot_cfg.get('distance_metric', 'euclidean')
        self.ot_matching_strategy = self.ot_cfg.get('matching_strategy', 'greedy')
        self.ot_normalize_cost = self.ot_cfg.get('normalize_cost', True)
        self.ot_start_epoch = self.ot_cfg.get('start_epoch', 0)
        self.ot_log_freq = self.ot_cfg.get('log_freq', 100)

        # Hybrid coupling (mix OT noise with independent Gaussian noise)
        hybrid_cfg = self.ot_cfg.get('hybrid_coupling', {})
        self.use_hybrid_coupling = hybrid_cfg.get('enable', False)
        self.hybrid_beta = float(hybrid_cfg.get('beta', 0.2))

        if not 0.0 <= self.hybrid_beta <= 1.0:
            raise ValueError(f"hybrid_coupling.beta must be in [0,1], got {self.hybrid_beta}")

        # Parse component weights for weighted OT（切片使用全局默认）
        component_weights = None

        if self.ot_distance_metric == 'weighted':
            # 仅读取权重；切片由 OT 实现内部按全局常量推断
            component_weights = self.ot_cfg.get('component_weights', {})
            if not component_weights:
                raise ValueError(
                    "distance_metric='weighted' requires 'component_weights' in config"
                )
            logging.info(f"{GREEN}Using weighted OT with component weights: {component_weights}{ENDC}")

        # Initialize OT solver if enabled
        if self.use_optimal_matching:
            self.ot_solver = SinkhornOT(
                reg=self.ot_reg,
                num_iters=self.ot_iters,
                distance_metric=self.ot_distance_metric,
                matching_strategy=self.ot_matching_strategy,
                normalize_cost=self.ot_normalize_cost,
                component_weights=component_weights,
            )
            logging.info(f"{GREEN}Optimal Transport enabled: reg={self.ot_reg}, "
                        f"iters={self.ot_iters}, metric={self.ot_distance_metric}{ENDC}")

            if self.use_hybrid_coupling:
                logging.info(f"{GREEN}Hybrid coupling enabled with beta={self.hybrid_beta}{ENDC}")
        else:
            self.ot_solver = None
            logging.info(f"{YELLOW}Optimal Transport disabled (using random index pairing){ENDC}")
            if self.use_hybrid_coupling:
                logging.warning(f"{YELLOW}Hybrid coupling requested but OT disabled; "
                                f"mixing defaults to independent noise.{ENDC}")
        
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
        
        # ========== Optimal Transport Matching ==========
        # 如果启用OT且达到起始epoch，执行Sinkhorn配对
        if (self.use_optimal_matching and 
            self.ot_solver is not None and 
            self.current_epoch >= self.ot_start_epoch):
            
            # 计算最优配对
            if batch_idx % self.ot_log_freq == 0:
                # 详细模式：返回配对信息用于日志
                matchings, ot_info = self.ot_solver(x0, x1, return_info=True)
                
                # 记录配对质量
                batch_total = B * num_grasps
                self.log("train/ot_matched_dist", ot_info['matched_distance'], 
                        prog_bar=False, logger=True, on_step=True, batch_size=batch_total, sync_dist=True)
                self.log("train/ot_random_dist", ot_info['random_distance'], 
                        prog_bar=False, logger=True, on_step=True, batch_size=batch_total, sync_dist=True)
                self.log("train/ot_improvement", ot_info['improvement'], 
                        prog_bar=False, logger=True, on_step=True, batch_size=batch_total, sync_dist=True)
                
                if self.trainer.is_global_zero:
                    logging.info(f"{BLUE}[OT] Epoch {self.current_epoch}, Batch {batch_idx}: "
                               f"matched_dist={ot_info['matched_distance']:.4f}, "
                               f"random_dist={ot_info['random_distance']:.4f}, "
                               f"improvement={ot_info['improvement']:.1f}%{ENDC}")
            else:
                # 快速模式：仅返回配对索引
                matchings = self.ot_solver(x0, x1, return_info=False)
            
            # 应用配对：重排序噪声
            x1 = apply_optimal_matching(x0, x1, matchings)

        # ========== Hybrid Coupling: mix OT noise with independent Gaussian ==========
        if (self.use_hybrid_coupling and self.use_optimal_matching and
                self.ot_solver is not None and self.current_epoch >= self.ot_start_epoch):
            if self.hybrid_beta > 0.0:
                mix_coeff = math.sqrt(max(1.0 - self.hybrid_beta, 0.0))
                noise_coeff = math.sqrt(self.hybrid_beta)
                epsilon = torch.randn_like(x1)
                x1 = mix_coeff * x1 + noise_coeff * epsilon
            self.log("train/hybrid_beta", self.hybrid_beta, prog_bar=False, logger=True,
                     on_step=True, batch_size=B * num_grasps, sync_dist=True)
        
        # Compute interpolation and target velocity
        # Use FM paths module
        x_t, v_star = self.path_fn(x0, x1, t)

        # Stochastic Flow Matching (training ablation)
        if self.sfm_sigma and self.sfm_sigma > 0:
            v_star = add_stochasticity(v_star, sigma=float(self.sfm_sigma))
        
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
        self._ensure_scene_mask(processed_batch)
        
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
        if self.path_type == 'linear_ot':
            pred_x0 = x_t - t_expanded * v_pred
        else:
            # Only linear_ot supports this simple inversion reliably
            if self.keep_ddpm_interface:
                raise NotImplementedError("keep_ddpm_interface only supports x0 reconstruction when path=linear_ot")
            pred_x0 = None
        
        if self.keep_ddpm_interface:
            pred_dict = {
                "pred_pose_norm": pred_x0,  # Predicted x0 for criterion
                "pred_noise": v_pred,  # Velocity prediction  
                "noise": v_star  # Target velocity
            }
            
            # Compute additional losses (optional)
            loss_dict = self.criterion(pred_dict, processed_batch, mode='train')
            
            # Combine primary FM MSE loss with set-level loss (conservative integration)
            # Only add set loss contribution to keep velocity MSE as the main objective
            set_total = self.criterion.get_weighted_set_loss(loss_dict)
            total_loss = loss + (set_total if torch.is_tensor(set_total) else 0.0)
        else:
            total_loss = loss
            loss_dict = {'velocity_loss': loss}
        
        # --- Gradient & numerical stability logging (lightweight) ---
        try:
            if self.trainer.is_global_zero and (batch_idx % max(1, self.print_freq) == 0):
                # 1) Parameter grad norms (after backward below we can log again if needed)
                def _grad_norm(module: nn.Module) -> float:
                    total = 0.0
                    for p in module.parameters():
                        if p.grad is not None:
                            g = p.grad.data
                            total += float(g.norm(2)**2)
                    return total ** 0.5
                global_gn = _grad_norm(self)
                scene_gn = _grad_norm(self.model.scene_model) if hasattr(self.model, 'scene_model') else 0.0
                adaln_gn = 0.0
                for m in self.model.modules():
                    if hasattr(m, 'modulation'):
                        for p in m.modulation.parameters():
                            if p.grad is not None:
                                adaln_gn += float(p.grad.data.norm(2)**2)
                adaln_gn = adaln_gn ** 0.5

                # 2) AdaLNZero forward stats aggregation
                from models.decoder.dit import AdaLNZero
                cond_std_mean = None
                gate_mean = None
                xmod_std_mean = None
                valid_layers = 0
                for m in self.model.modules():
                    if isinstance(m, AdaLNZero) and hasattr(m, '_last_stats') and isinstance(m._last_stats, dict):
                        s = m._last_stats
                        # 仅统计有限值样本
                        if s.get('cond_isfinite', True):
                            valid_layers += 1
                            cond_std_mean = (0.0 if cond_std_mean is None else cond_std_mean) + float(s.get('cond_std', 0.0))
                            gate_mean = (0.0 if gate_mean is None else gate_mean) + float(s.get('gate_mean', 0.0))
                            xmod_std_mean = (0.0 if xmod_std_mean is None else xmod_std_mean) + float(s.get('xmod_std', 0.0))
                if valid_layers > 0:
                    cond_std_mean = cond_std_mean / valid_layers
                    gate_mean = gate_mean / valid_layers
                    xmod_std_mean = xmod_std_mean / valid_layers

                logging.info(
                    f"[fm_train] ep={self.current_epoch} it={batch_idx} "
                    f"loss={float(loss.item()):.4f} grad(global/scene/adaln)=({global_gn:.2e}/{scene_gn:.2e}/{adaln_gn:.2e}) "
                    f"adaln(cond_std_mean={cond_std_mean if cond_std_mean is not None else 'n/a'}, "
                    f"gate_mean={gate_mean if gate_mean is not None else 'n/a'}, "
                    f"xmod_std_mean={xmod_std_mean if xmod_std_mean is not None else 'n/a'})"
                )
        except Exception:
            pass

        # Prepare set loss values for logging
        set_loss_values = {}
        if isinstance(loss_dict, dict) and getattr(self.criterion, "set_loss_enabled", False):
            for key in ("set_total", "set_total_raw", "set_cd", "set_ot", "set_repulsion"):
                value = loss_dict.get(key)
                if value is not None:
                    set_loss_values[key] = value

        # Logging
        total_samples = B * num_grasps
        train_log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        train_log_dict["train/total_loss"] = total_loss
        train_log_dict["train/mean_t"] = t.mean()
        train_log_dict["train/velocity_norm"] = torch.norm(v_pred, dim=-1).mean()
        
        self.log_dict(train_log_dict, prog_bar=False, logger=True, on_step=True, 
                     on_epoch=True, batch_size=total_samples, sync_dist=True)
        
        # Step time-aware gating (MLP gate warmup/learning)
        try:
            gate = getattr(self.model, 'time_gate', None)
            if gate is not None:
                gate.step()
        except Exception:
            # 不阻塞训练，安静失败
            pass
        
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
            if getattr(self.criterion, "set_loss_enabled", False) and set_loss_values:
                logging.info(f'{BLUE}{"--- Set Loss Components ---":<21s}{ENDC}')
                for key in ("set_total", "set_total_raw", "set_cd", "set_ot", "set_repulsion"):
                    value = set_loss_values.get(key)
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor):
                        value = float(value.detach().float().mean().item())
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    logging.info(f'{BLUE}{key + ":":<21s} {value:.4f}{ENDC}')
        
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

        # Compute set/quality metrics once per batch
        set_metrics = {}
        quality_metrics = {}
        try:
            tmp_metrics = self.criterion.compute_set_metrics(pred_dict, batch)
            if isinstance(tmp_metrics, dict):
                set_metrics = {k: float(v) for k, v in tmp_metrics.items()}
                if set_metrics:
                    val_metrics_log = {f"val/{k}": v for k, v in set_metrics.items()}
                    self.log_dict(val_metrics_log, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                                  batch_size=batch_size, sync_dist=True)
        except Exception:
            pass

        try:
            tmp_quality = self.criterion.compute_quality_metrics(pred_dict, batch)
            if isinstance(tmp_quality, dict):
                quality_metrics = {k: float(v) for k, v in tmp_quality.items()}
                if quality_metrics:
                    quality_log = {f"val/quality/{k}": v for k, v in quality_metrics.items()}
                    self.log_dict(quality_log, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                                  batch_size=batch_size, sync_dist=True)
        except Exception:
            pass

        # Handle standardized validation loss (align with DDPMLightning)
        if hasattr(self.criterion, 'use_standardized_val_loss') and self.criterion.use_standardized_val_loss:
            # Extract standard losses for logging
            standard_loss_dict = loss_dict.pop('_standard_losses', {})

            # Calculate standardized total loss
            std_weights = getattr(self.criterion, 'std_val_weights', {})
            loss = sum(
                v * std_weights.get(k, 0)
                for k, v in loss_dict.items()
                if k in std_weights and std_weights[k] > 0
            )

            # Calculate standard total loss for comparison
            standard_loss = sum(
                v * self.loss_weights[k]
                for k, v in standard_loss_dict.items()
                if k in self.loss_weights
            )

            # Log both losses
            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)
            self.log("val/standardized_loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
            self.log("val/standard_loss", standard_loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            # Store both for epoch-end processing
            self.validation_step_outputs.append({
                "loss": loss.item(),
                "standardized_loss": loss.item(),
                "standard_loss": standard_loss.item(),
                "loss_dict": {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()},
                "standard_loss_dict": {k: (v.item() if torch.is_tensor(v) else v) for k, v in standard_loss_dict.items()},
                "set_metrics": set_metrics,
                "quality_metrics": quality_metrics,
            })
        else:
            # Standard validation loss calculation
            loss = sum(
                v * self.loss_weights[k]
                for k, v in loss_dict.items()
                if k in self.loss_weights and torch.is_tensor(v)
            )
            self.log("val/loss", loss, prog_bar=False, batch_size=batch_size, sync_dist=True)

            self.validation_step_outputs.append({
                "loss": loss.item(),
                "loss_dict": {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()},
                "set_metrics": set_metrics,
                "quality_metrics": quality_metrics,
            })

        return {"loss": loss, "loss_dict": loss_dict}
    
    def on_validation_epoch_end(self):
        val_loss = [x["loss"] for x in self.validation_step_outputs]
        avg_loss = mean(val_loss) if val_loss else 0.0
        loss_std = stdev(val_loss) if len(val_loss) > 1 else 0.0

        # Detect if standardized validation loss is used
        using_standardized = (
            hasattr(self.criterion, 'use_standardized_val_loss') and
            self.criterion.use_standardized_val_loss and
            len(self.validation_step_outputs) > 0 and
            ("standardized_loss" in self.validation_step_outputs[0])
        )

        if using_standardized:
            # Aggregate standardized and standard losses
            std_loss = [x["standardized_loss"] for x in self.validation_step_outputs]
            standard_loss = [x["standard_loss"] for x in self.validation_step_outputs]

            avg_std_loss = mean(std_loss) if std_loss else 0.0
            avg_standard_loss = mean(standard_loss) if standard_loss else 0.0

            # Detailed losses for standardized
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean([x["loss_dict"][k] for x in self.validation_step_outputs])

            # Detailed losses for standard (for comparison)
            standard_detailed_loss = {}
            if self.validation_step_outputs and ("standard_loss_dict" in self.validation_step_outputs[0]):
                for k in self.validation_step_outputs[0]["standard_loss_dict"].keys():
                    standard_detailed_loss[k] = mean([x["standard_loss_dict"][k] for x in self.validation_step_outputs])

            # Statistics for standardized loss
            std_loss_std = torch.std(torch.tensor(std_loss)).item() if len(std_loss) > 1 else 0.0
            std_loss_min = min(std_loss) if std_loss else 0.0
            std_loss_max = max(std_loss) if std_loss else 0.0

            set_metrics = self._aggregate_metrics(self.validation_step_outputs, "set_metrics")
            quality_metrics = self._aggregate_metrics(self.validation_step_outputs, "quality_metrics")
            log_validation_summary(
                epoch=self.current_epoch,
                num_batches=len(self.validation_step_outputs),
                avg_loss=avg_std_loss,
                loss_std=std_loss_std,
                loss_min=std_loss_min,
                loss_max=std_loss_max,
                val_detailed_loss=val_detailed_loss,
                val_set_metrics=set_metrics,
                val_quality_metrics=quality_metrics,
            )

            # Log standardized validation losses
            val_log_dict = {f"val/std_{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update({
                "val/standardized_total_loss": avg_std_loss,
                "val/std_loss_std": std_loss_std,
                "val/std_loss_min": std_loss_min,
                "val/std_loss_max": std_loss_max,
            })

            # Log standard (original) losses for comparison
            standard_log_dict = {f"val/orig_{k}": v for k, v in standard_detailed_loss.items()}
            standard_log_dict.update({
                "val/original_total_loss": avg_standard_loss,
            })
            val_log_dict.update(standard_log_dict)

            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

            # Use standardized loss for ModelCheckpoint
            self.log('val_loss', avg_std_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
            # Also log both for manual comparison
            self.log('val_loss_standardized', avg_std_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)
            self.log('val_loss_original', avg_standard_loss, prog_bar=False, batch_size=self.batch_size, sync_dist=True)
        else:
            # Compute detailed losses (standard)
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    values = [x["loss_dict"][k] for x in self.validation_step_outputs]
                    if values and not isinstance(values[0], dict):
                        val_detailed_loss[k] = mean(values)

            set_metrics = self._aggregate_metrics(self.validation_step_outputs, "set_metrics")
            quality_metrics = self._aggregate_metrics(self.validation_step_outputs, "quality_metrics")
            log_validation_summary(
                epoch=self.current_epoch,
                num_batches=len(self.validation_step_outputs),
                avg_loss=avg_loss,
                loss_std=loss_std,
                loss_min=min(val_loss) if val_loss else 0.0,
                loss_max=max(val_loss) if val_loss else 0.0,
                val_detailed_loss=val_detailed_loss,
                val_set_metrics=set_metrics,
                val_quality_metrics=quality_metrics,
            )

            # Log to wandb
            val_log_dict = {f"val/{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict["val/total_loss"] = avg_loss
            val_log_dict["val/loss_std"] = loss_std
            self.log_dict(val_log_dict, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

            # Log for checkpoint
            self.log('val_loss', avg_loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)

        self.validation_step_outputs.clear()

    @staticmethod
    def _aggregate_metrics(outputs, key: str) -> Dict[str, float]:
        aggregated: Dict[str, list] = {}
        for item in outputs:
            if not isinstance(item, dict):
                continue
            metric_dict = item.get(key)
            if not isinstance(metric_dict, dict):
                continue
            for name, value in metric_dict.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                aggregated.setdefault(name, []).append(scalar)
        return {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in aggregated.items()}

    @staticmethod
    def _ensure_scene_mask(batch: Dict) -> None:
        """Ensure scene_mask exists when scene_cond is present."""
        scene_context = batch.get('scene_cond')
        if scene_context is None or not isinstance(scene_context, torch.Tensor):
            return

        mask = batch.get('scene_mask')
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3:
                mask = mask.squeeze(1)
            if mask.shape == (scene_context.shape[0], scene_context.shape[1]):
                mask = torch.clamp(
                    mask.to(scene_context.device, dtype=torch.float32), 0.0, 1.0
                )
                batch['scene_mask'] = mask
                return
            else:
                logging.warning(
                    f"[Scene Mask] Provided scene_mask shape {tuple(mask.shape)} "
                    f"mismatches scene_cond {scene_context.shape[:2]}, rebuilding full mask."
                )

        B, N, _ = scene_context.shape
        batch['scene_mask'] = torch.ones(B, N, device=scene_context.device, dtype=torch.float32)
    
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
        self._ensure_scene_mask(data)
        
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
        Integrate ODE via models.fm.solvers unified interface.
        """
        def velocity_fn(x: torch.Tensor, t_tensor: torch.Tensor, d: Dict) -> torch.Tensor:
            v = self._velocity_fn(x, t_tensor, d)
            # Optional SFM during sampling
            if self.sfm_sigma and self.sfm_sigma > 0 and not self.training:
                v = add_stochasticity(v, sigma=float(self.sfm_sigma))
            return v

        x0, info = fm_integrate_ode(
            velocity_fn=velocity_fn,
            x1=x1,
            data=data,
            solver_type=self.solver_type,
            nfe=int(self.nfe),
            rtol=float(self.rtol),
            atol=float(self.atol),
            max_step=float(self.max_step) if self.max_step is not None else None,
            min_step=float(self.min_step),
            reverse_time=bool(self.reverse_time),
            save_trajectory=bool(self.save_trajectories),
        )

        # Optional: log solver stats
        if self.trainer is not None and self.trainer.is_global_zero and isinstance(info, dict) and 'stats' in info:
            stats = info['stats']
            try:
                nfe = float(stats.get('nfe', 0))
                effective_nfe = float(stats.get('effective_nfe', 0))
                batch_sz = int(x1.shape[0] * (x1.shape[1] if x1.dim() > 2 else 1))
                self.log(
                    "sample/nfe",
                    nfe,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_sz,
                )
                self.log(
                    "sample/effective_nfe",
                    effective_nfe,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_sz,
                )
            except Exception:
                pass

        return x0
    
    
    
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
            # Return conditional velocity during training or when CFG is disabled
            return self.model(x, t, data)

        # Classifier-Free Guidance in inference
        v_cond = self.model(x, t, data)

        # Build unconditional branch
        data_uncond = data.copy()
        for key in ['scene_cond', 'text_cond']:
            if key in data_uncond and data_uncond[key] is not None:
                data_uncond[key] = torch.zeros_like(data_uncond[key])

        v_uncond = self.model(x, t, data_uncond)

        # Apply CFG with models.fm.guidance (supports multiple strategies)
        v = fm_apply_cfg(
            v_cond=v_cond,
            v_uncond=v_uncond,
            scale=float(self.guidance_scale),
            method=str(self.guidance_method),
            clip_norm=float(self.diff_clip),
            t=t,
            early_steps_scale=float(self.early_steps_scale),
            late_steps_scale=float(self.late_steps_scale),
            transition_point=float(self.transition_point),
        )

        # Optional predictor-corrector refinement
        if self.use_pc_correction:
            v = fm_predictor_corrector_step(
                x=x,
                t=t,
                v=v,
                velocity_fn=lambda xp, tp, d: self.model(xp, tp, d),
                data=data,
                dt_correction=float(self.dt_correction),
                num_corrections=int(self.num_corrections),
            )

        return v
    
    
    
    
    
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
    
    def forward_get_pose_matched(self, data: Dict, k: int = 1):
        """Get matched pose predictions (for visualization)."""
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)
        
        if k > 1:
            pred_x0 = pred_x0[:, 0]
        
        pred_dict = build_pred_dict_adaptive(pred_x0)
        matched_preds, matched_targets, outputs, targets = self.criterion.forward_infer(pred_dict, data)
        return matched_preds, matched_targets, outputs, targets
