"""
Flow Matching Paths

This module implements different probability paths for Flow Matching:
1. Linear OT path (default): straight path from data to noise
2. Diffusion path (for ablation): VP/VE schedules with analytical velocity
"""

import torch
import math
from typing import Tuple, Optional


def linear_ot_path(
    x0: torch.Tensor, 
    x1: torch.Tensor, 
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear Optimal Transport path.
    
    This is the default path for Flow Matching, providing the simplest
    and most direct interpolation between data and noise distributions.
    
    Path: x_t = (1-t)·x0 + t·x1
    Velocity: v* = x1 - x0 (constant)
    
    Args:
        x0: Data samples [B, num_grasps, D]
        x1: Noise samples [B, num_grasps, D]
        t: Time values in [0, 1], shape [B]
        
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


def diffusion_path_vp(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    beta_min: float = 0.1,
    beta_max: float = 20.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variance Preserving (VP) diffusion path with analytical velocity.
    
    This path follows the VP-SDE from DDPM but provides analytical velocity
    for Flow Matching training. Used only for ablation studies.
    
    Path: x_t = α(t)·x0 + σ(t)·ε
    Velocity: v* = α'(t)·x0 + σ'(t)·ε
    
    WARNING: This is for ablation only. Linear OT is recommended for FM.
    
    Args:
        x0: Data samples [B, num_grasps, D]
        x1: Noise samples (ε) [B, num_grasps, D]
        t: Time values in [0, 1], shape [B]
        beta_min: Minimum noise schedule value
        beta_max: Maximum noise schedule value
        
    Returns:
        x_t: Noisy state [B, num_grasps, D]
        v_star: Analytical velocity [B, num_grasps, D]
    """
    # Expand t for broadcasting
    t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
    
    # VP schedule: β(t) = β_min + t(β_max - β_min)
    # Integral: ∫β(s)ds = β_min·t + 0.5·t²(β_max - β_min)
    beta_integral = beta_min * t_expanded + 0.5 * t_expanded**2 * (beta_max - beta_min)
    
    # α(t) and σ(t)
    alpha_t = torch.exp(-0.5 * beta_integral)
    sigma_t = torch.sqrt(1 - alpha_t**2)
    
    # Derivatives for velocity
    # β(t) = β_min + t(β_max - β_min)
    beta_t = beta_min + t_expanded * (beta_max - beta_min)
    
    # α'(t) = -0.5·β(t)·α(t)
    alpha_prime_t = -0.5 * beta_t * alpha_t
    
    # σ'(t) = β(t)·α(t)²/σ(t)
    sigma_prime_t = beta_t * alpha_t**2 / (sigma_t + 1e-8)
    
    # Path and velocity
    x_t = alpha_t * x0 + sigma_t * x1
    v_star = alpha_prime_t * x0 + sigma_prime_t * x1
    
    return x_t, v_star


def diffusion_path_ve(
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variance Exploding (VE) diffusion path with analytical velocity.
    
    Path: x_t = x0 + σ(t)·ε
    Velocity: v* = σ'(t)·ε
    
    WARNING: This is for ablation only. Linear OT is recommended for FM.
    
    Args:
        x0: Data samples [B, num_grasps, D]
        x1: Noise samples (ε) [B, num_grasps, D]
        t: Time values in [0, 1], shape [B]
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        x_t: Noisy state [B, num_grasps, D]
        v_star: Analytical velocity [B, num_grasps, D]
    """
    # Expand t for broadcasting
    t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
    
    # VE schedule: σ(t) = σ_min·(σ_max/σ_min)^t
    log_ratio = math.log(sigma_max / sigma_min)
    sigma_t = sigma_min * torch.exp(t_expanded * log_ratio)
    
    # Derivative: σ'(t) = log(σ_max/σ_min)·σ(t)
    sigma_prime_t = log_ratio * sigma_t
    
    # Path and velocity
    x_t = x0 + sigma_t * x1
    v_star = sigma_prime_t * x1
    
    return x_t, v_star


def get_path_fn(path_type: str, **kwargs):
    """
    Factory function to get the path function.
    
    Args:
        path_type: Type of path ('linear_ot', 'vp', 've')
        **kwargs: Additional arguments for specific paths
        
    Returns:
        Path function that takes (x0, x1, t) and returns (x_t, v_star)
    """
    if path_type == 'linear_ot':
        return linear_ot_path
    elif path_type == 'vp' or path_type == 'diffusion_vp':
        def vp_path(x0, x1, t):
            return diffusion_path_vp(x0, x1, t, 
                                    beta_min=kwargs.get('beta_min', 0.1),
                                    beta_max=kwargs.get('beta_max', 20.0))
        return vp_path
    elif path_type == 've' or path_type == 'diffusion_ve':
        def ve_path(x0, x1, t):
            return diffusion_path_ve(x0, x1, t,
                                    sigma_min=kwargs.get('sigma_min', 0.01),
                                    sigma_max=kwargs.get('sigma_max', 50.0))
        return ve_path
    else:
        raise ValueError(f"Unknown path type: {path_type}. "
                       f"Supported: 'linear_ot', 'vp', 've'")


# Stochastic Flow Matching (optional)
def add_stochasticity(
    v: torch.Tensor,
    sigma: float = 0.0
) -> torch.Tensor:
    """
    Add stochasticity to the flow for Stochastic Flow Matching.
    
    This converts the deterministic ODE into an SDE:
    dx = v(x,t)dt + σ·dW
    
    Args:
        v: Deterministic velocity [B, num_grasps, D]
        sigma: Noise level (0 for deterministic)
        
    Returns:
        v_stochastic: Velocity with added noise [B, num_grasps, D]
    """
    if sigma <= 0:
        return v
    
    noise = torch.randn_like(v) * sigma
    return v + noise
