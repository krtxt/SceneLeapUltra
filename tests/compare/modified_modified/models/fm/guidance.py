"""
Classifier-Free Guidance for Flow Matching

This module implements stabilized CFG specifically designed for
Flow Matching models, addressing issues like off-manifold drift.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
import logging


def apply_cfg_basic(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    scale: float = 3.0
) -> torch.Tensor:
    """
    Basic classifier-free guidance.
    
    v_cfg = v_cond + scale * (v_cond - v_uncond)
    
    Args:
        v_cond: Conditional velocity [B, num_grasps, D]
        v_uncond: Unconditional velocity [B, num_grasps, D]
        scale: Guidance scale (higher = stronger conditioning)
        
    Returns:
        v_cfg: Guided velocity [B, num_grasps, D]
    """
    return v_cond + scale * (v_cond - v_uncond)


def apply_cfg_clipped(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    scale: float = 3.0,
    clip_norm: float = 5.0
) -> torch.Tensor:
    """
    CFG with norm clipping to prevent off-manifold drift.
    
    The difference (v_cond - v_uncond) is clipped to have maximum norm
    of clip_norm. This prevents extreme velocities that can lead to
    unstable trajectories.
    
    Args:
        v_cond: Conditional velocity [B, num_grasps, D]
        v_uncond: Unconditional velocity [B, num_grasps, D]
        scale: Guidance scale
        clip_norm: Maximum norm for the difference vector
        
    Returns:
        v_cfg: Guided velocity with clipped difference
    """
    diff = v_cond - v_uncond
    
    # Compute norms
    diff_norm = torch.norm(diff, dim=-1, keepdim=True)  # [B, num_grasps, 1]
    
    # Clip if exceeds threshold
    if clip_norm > 0:
        scale_factor = torch.minimum(
            torch.ones_like(diff_norm),
            clip_norm / (diff_norm + 1e-8)
        )
        diff = diff * scale_factor
    
    v_cfg = v_cond + scale * diff
    
    return v_cfg


def apply_cfg_rescaled(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    scale: float = 3.0,
    target_norm: Optional[float] = None
) -> torch.Tensor:
    """
    CFG with norm rescaling instead of clipping.
    
    Instead of clipping, this rescales the difference to have a target norm,
    which can provide smoother guidance.
    
    Args:
        v_cond: Conditional velocity [B, num_grasps, D]
        v_uncond: Unconditional velocity [B, num_grasps, D]
        scale: Guidance scale
        target_norm: Target norm for rescaling (if None, use mean norm)
        
    Returns:
        v_cfg: Guided velocity with rescaled difference
    """
    diff = v_cond - v_uncond
    
    # Compute norms
    diff_norm = torch.norm(diff, dim=-1, keepdim=True)  # [B, num_grasps, 1]
    
    # Determine target norm
    if target_norm is None:
        # Use mean norm as target
        target_norm = diff_norm.mean().item()
    
    # Rescale
    scale_factor = target_norm / (diff_norm + 1e-8)
    diff_rescaled = diff * scale_factor
    
    v_cfg = v_cond + scale * diff_rescaled
    
    return v_cfg


def apply_cfg_adaptive(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    scale: float = 3.0,
    t: Optional[torch.Tensor] = None,
    early_steps_scale: float = 0.0,
    late_steps_scale: float = 1.0,
    transition_point: float = 0.5
) -> torch.Tensor:
    """
    CFG with time-dependent scaling.
    
    Applies different guidance scales at different time points:
    - Early steps (t near 1): Weaker guidance to allow broad exploration
    - Late steps (t near 0): Full guidance for refinement
    
    This follows insights from "Rectified-CFG++" and similar works.
    
    Args:
        v_cond: Conditional velocity [B, num_grasps, D]
        v_uncond: Unconditional velocity [B, num_grasps, D]
        scale: Base guidance scale
        t: Current time values [B] (0 to 1)
        early_steps_scale: Scale multiplier for early steps
        late_steps_scale: Scale multiplier for late steps
        transition_point: Time point for transition (0 to 1)
        
    Returns:
        v_cfg: Guided velocity with adaptive scaling
    """
    diff = v_cond - v_uncond
    
    if t is not None:
        # Compute time-dependent scale
        # Linear interpolation between early and late scales
        t_normalized = torch.clamp(t, 0.0, 1.0)
        
        # For reverse time (1 to 0), early is high t, late is low t
        # Transition from early_steps_scale to late_steps_scale
        time_scale = torch.where(
            t_normalized > transition_point,
            early_steps_scale + (late_steps_scale - early_steps_scale) * 
                (1.0 - (t_normalized - transition_point) / (1.0 - transition_point)),
            torch.ones_like(t_normalized) * late_steps_scale
        )
        
        # Expand for broadcasting
        time_scale = time_scale.view(-1, 1, 1)  # [B, 1, 1]
        
        effective_scale = scale * time_scale
    else:
        effective_scale = scale
    
    v_cfg = v_cond + effective_scale * diff
    
    return v_cfg


def predictor_corrector_step(
    x: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    velocity_fn: Callable,
    data: Dict,
    dt_correction: float = 0.01,
    num_corrections: int = 1
) -> torch.Tensor:
    """
    Predictor-corrector step for improved accuracy.
    
    After predicting the velocity, take a small correction step to
    refine the estimate. This can help keep trajectories on-manifold.
    
    Args:
        x: Current state [B, num_grasps, D]
        t: Current time [B]
        v: Predicted velocity [B, num_grasps, D]
        velocity_fn: Function to compute velocity
        data: Conditioning data
        dt_correction: Step size for correction
        num_corrections: Number of correction iterations
        
    Returns:
        v_corrected: Corrected velocity [B, num_grasps, D]
    """
    v_corrected = v
    
    for _ in range(num_corrections):
        # Take a small step with current velocity
        x_pred = x - dt_correction * v_corrected
        t_pred = t - dt_correction
        
        # Re-evaluate velocity at predicted point
        v_new = velocity_fn(x_pred, t_pred, data)
        
        # Average predictor and corrector
        v_corrected = 0.5 * (v_corrected + v_new)
    
    return v_corrected


class CFGScheduler:
    """
    Schedule CFG scale based on timestep for improved stability.
    
    This class implements various scheduling strategies for CFG scale:
    - constant: Fixed scale throughout
    - linear: Linearly increase/decrease
    - cosine: Smooth cosine schedule
    - zero_early: Zero guidance in early steps, full later
    """
    
    def __init__(
        self,
        base_scale: float = 3.0,
        schedule_type: str = 'constant',
        start_scale: float = 0.0,
        end_scale: float = 1.0,
        transition_steps: int = 5
    ):
        self.base_scale = base_scale
        self.schedule_type = schedule_type
        self.start_scale = start_scale
        self.end_scale = end_scale
        self.transition_steps = transition_steps
    
    def get_scale(self, t: torch.Tensor, total_steps: int, current_step: int) -> float:
        """
        Get CFG scale for current step.
        
        Args:
            t: Current time value [B] or scalar
            total_steps: Total number of integration steps
            current_step: Current step index (0 to total_steps-1)
            
        Returns:
            scale: CFG scale to use
        """
        if self.schedule_type == 'constant':
            return self.base_scale
        
        elif self.schedule_type == 'linear':
            # Linear interpolation from start to end
            progress = current_step / max(total_steps - 1, 1)
            scale_mult = self.start_scale + (self.end_scale - self.start_scale) * progress
            return self.base_scale * scale_mult
        
        elif self.schedule_type == 'cosine':
            # Cosine schedule (smooth transition)
            progress = current_step / max(total_steps - 1, 1)
            scale_mult = self.start_scale + 0.5 * (self.end_scale - self.start_scale) * \
                        (1 - torch.cos(torch.tensor(progress * 3.14159)))
            return self.base_scale * scale_mult.item()
        
        elif self.schedule_type == 'zero_early':
            # Zero guidance in early steps, full in later steps
            if current_step < self.transition_steps:
                return 0.0
            else:
                return self.base_scale
        
        else:
            return self.base_scale
    
    def __call__(self, t: torch.Tensor, total_steps: int, current_step: int) -> float:
        return self.get_scale(t, total_steps, current_step)


def apply_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    scale: float = 3.0,
    method: str = 'clipped',
    clip_norm: float = 5.0,
    t: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Unified CFG application interface.
    
    Args:
        v_cond: Conditional velocity
        v_uncond: Unconditional velocity
        scale: Guidance scale
        method: 'basic', 'clipped', 'rescaled', or 'adaptive'
        clip_norm: Clipping threshold (for 'clipped' method)
        t: Time values (for 'adaptive' method)
        **kwargs: Additional method-specific arguments
        
    Returns:
        v_cfg: Guided velocity
    """
    if method == 'basic':
        return apply_cfg_basic(v_cond, v_uncond, scale)
    elif method == 'clipped':
        return apply_cfg_clipped(v_cond, v_uncond, scale, clip_norm)
    elif method == 'rescaled':
        target_norm = kwargs.get('target_norm', None)
        return apply_cfg_rescaled(v_cond, v_uncond, scale, target_norm)
    elif method == 'adaptive':
        if t is None:
            logging.warning("Adaptive CFG requires time values, falling back to clipped")
            return apply_cfg_clipped(v_cond, v_uncond, scale, clip_norm)
        return apply_cfg_adaptive(v_cond, v_uncond, scale, t, **kwargs)
    else:
        raise ValueError(f"Unknown CFG method: {method}")

