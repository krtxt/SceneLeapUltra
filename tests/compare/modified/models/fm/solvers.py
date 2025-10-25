"""
ODE Solvers for Flow Matching

This module provides various ODE integration methods for sampling from
Flow Matching models, including fixed-step and adaptive methods.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ODESolverStats:
    """Statistics tracker for ODE integration."""
    
    def __init__(self):
        self.nfe = 0  # Number of function evaluations
        self.failed_steps = 0
        self.accepted_steps = 0
        self.total_time = 0.0
        self.step_sizes = []
    
    def reset(self):
        self.nfe = 0
        self.failed_steps = 0
        self.accepted_steps = 0
        self.total_time = 0.0
        self.step_sizes = []
    
    def get_effective_nfe(self, use_cfg: bool = False) -> int:
        """Get effective NFE considering CFG doubles the forward calls."""
        multiplier = 2 if use_cfg else 1
        return self.nfe * multiplier
    
    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            'nfe': self.nfe,
            'effective_nfe': self.get_effective_nfe(),
            'failed_steps': self.failed_steps,
            'accepted_steps': self.accepted_steps,
            'total_time': self.total_time,
            'avg_step_size': sum(self.step_sizes) / len(self.step_sizes) if self.step_sizes else 0.0,
            'min_step_size': min(self.step_sizes) if self.step_sizes else 0.0,
            'max_step_size': max(self.step_sizes) if self.step_sizes else 0.0
        }


def heun_solver(
    velocity_fn: Callable,
    x1: torch.Tensor,
    data: Dict,
    nfe: int = 32,
    reverse_time: bool = True,
    save_trajectory: bool = False,
    stats: Optional[ODESolverStats] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Heun's method (2nd order Runge-Kutta) for ODE integration.
    
    This is a predictor-corrector method that provides better accuracy
    than Euler method with moderate computational cost.
    
    Args:
        velocity_fn: Function that computes v(x, t) given x and t
        x1: Initial state at t=1 [B, num_grasps, D]
        data: Conditioning data dictionary
        nfe: Number of function evaluations
        reverse_time: If True, integrate from t=1 to t=0
        save_trajectory: If True, save intermediate states
        stats: Statistics tracker
        
    Returns:
        x0: Final state [B, num_grasps, D]
        info: Dictionary with solver information
    """
    if stats is None:
        stats = ODESolverStats()
    
    start_time = time.time()
    
    dt = 1.0 / nfe
    x = x1.clone()
    
    trajectory = [x.clone()] if save_trajectory else None
    
    for i in range(nfe):
        if reverse_time:
            t = 1.0 - i * dt  # From 1 to dt
            t_next = t - dt
        else:
            t = i * dt  # From 0 to 1-dt
            t_next = t + dt
        
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        
        # Predictor: Euler step
        v1 = velocity_fn(x, t_tensor, data)
        stats.nfe += 1
        
        if reverse_time:
            x_pred = x - dt * v1
        else:
            x_pred = x + dt * v1
        
        # Corrector: evaluate at predicted point
        t_next_tensor = torch.full((x.shape[0],), t_next, device=x.device, dtype=torch.float32)
        v2 = velocity_fn(x_pred, t_next_tensor, data)
        stats.nfe += 1
        
        # Average the two velocities
        v_avg = 0.5 * (v1 + v2)
        
        # Update
        if reverse_time:
            x = x - dt * v_avg
        else:
            x = x + dt * v_avg
        
        stats.accepted_steps += 1
        stats.step_sizes.append(dt)
        
        if save_trajectory:
            trajectory.append(x.clone())
    
    stats.total_time = time.time() - start_time
    
    info = {
        'solver': 'heun',
        'nfe': stats.nfe,
        'trajectory': trajectory if save_trajectory else None,
        'stats': stats.summary()
    }
    
    return x, info


def rk4_solver(
    velocity_fn: Callable,
    x1: torch.Tensor,
    data: Dict,
    nfe: int = 32,
    reverse_time: bool = True,
    save_trajectory: bool = False,
    stats: Optional[ODESolverStats] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    4th order Runge-Kutta method for ODE integration.
    
    RK4 provides higher accuracy than Heun with 4 function evaluations per step.
    This is the recommended default solver for Flow Matching.
    
    Args:
        velocity_fn: Function that computes v(x, t) given x and t
        x1: Initial state at t=1 [B, num_grasps, D]
        data: Conditioning data dictionary
        nfe: Number of function evaluations (must be multiple of 4)
        reverse_time: If True, integrate from t=1 to t=0
        save_trajectory: If True, save intermediate states
        stats: Statistics tracker
        
    Returns:
        x0: Final state [B, num_grasps, D]
        info: Dictionary with solver information
    """
    if stats is None:
        stats = ODESolverStats()
    
    # Ensure NFE is multiple of 4 for RK4
    num_steps = nfe // 4
    if nfe % 4 != 0:
        logging.warning(f"NFE {nfe} not multiple of 4, using {num_steps*4} instead")
        nfe = num_steps * 4
    
    start_time = time.time()
    
    dt = 1.0 / num_steps
    x = x1.clone()
    
    trajectory = [x.clone()] if save_trajectory else None
    
    for i in range(num_steps):
        if reverse_time:
            t = 1.0 - i * dt
        else:
            t = i * dt
        
        # RK4 stages
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        k1 = velocity_fn(x, t_tensor, data)
        stats.nfe += 1
        
        t_half = t + 0.5 * dt if not reverse_time else t - 0.5 * dt
        t_half_tensor = torch.full((x.shape[0],), t_half, device=x.device, dtype=torch.float32)
        
        if reverse_time:
            k2 = velocity_fn(x - 0.5 * dt * k1, t_half_tensor, data)
            stats.nfe += 1
            k3 = velocity_fn(x - 0.5 * dt * k2, t_half_tensor, data)
            stats.nfe += 1
        else:
            k2 = velocity_fn(x + 0.5 * dt * k1, t_half_tensor, data)
            stats.nfe += 1
            k3 = velocity_fn(x + 0.5 * dt * k2, t_half_tensor, data)
            stats.nfe += 1
        
        t_next = t + dt if not reverse_time else t - dt
        t_next_tensor = torch.full((x.shape[0],), t_next, device=x.device, dtype=torch.float32)
        
        if reverse_time:
            k4 = velocity_fn(x - dt * k3, t_next_tensor, data)
            stats.nfe += 1
            x = x - dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        else:
            k4 = velocity_fn(x + dt * k3, t_next_tensor, data)
            stats.nfe += 1
            x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        stats.accepted_steps += 1
        stats.step_sizes.append(dt)
        
        if save_trajectory:
            trajectory.append(x.clone())
    
    stats.total_time = time.time() - start_time
    
    info = {
        'solver': 'rk4',
        'nfe': stats.nfe,
        'trajectory': trajectory if save_trajectory else None,
        'stats': stats.summary()
    }
    
    return x, info


def rk45_adaptive_solver(
    velocity_fn: Callable,
    x1: torch.Tensor,
    data: Dict,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    max_step: Optional[float] = None,
    min_step: float = 1e-4,
    reverse_time: bool = True,
    save_trajectory: bool = False,
    stats: Optional[ODESolverStats] = None,
    max_nfe: int = 1000
) -> Tuple[torch.Tensor, Dict]:
    """
    Adaptive RK45 (Dormand-Prince) method for ODE integration.
    
    This method automatically adjusts step sizes based on error estimates,
    providing a good balance between accuracy and efficiency.
    
    Args:
        velocity_fn: Function that computes v(x, t) given x and t
        x1: Initial state [B, num_grasps, D]
        data: Conditioning data dictionary
        rtol: Relative tolerance for error control
        atol: Absolute tolerance for error control
        max_step: Maximum step size (if None, set to 1/32)
        min_step: Minimum step size
        reverse_time: If True, integrate from t=1 to t=0
        save_trajectory: If True, save intermediate states
        stats: Statistics tracker
        max_nfe: Maximum number of function evaluations
        
    Returns:
        x0: Final state [B, num_grasps, D]
        info: Dictionary with solver information
    """
    if stats is None:
        stats = ODESolverStats()
    
    if max_step is None:
        max_step = 0.03125  # 1/32
    
    start_time = time.time()
    
    # RK45 coefficients (Dormand-Prince)
    c = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=torch.float32)
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=torch.float32)
    b_star = torch.tensor([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=torch.float32)
    
    x = x1.clone()
    t = 1.0 if reverse_time else 0.0
    dt = max_step if reverse_time else -max_step  # Initial step size
    
    trajectory = [x.clone()] if save_trajectory else None
    
    target_t = 0.0 if reverse_time else 1.0
    
    while (reverse_time and t > target_t + min_step) or (not reverse_time and t < target_t - min_step):
        if stats.nfe >= max_nfe:
            logging.warning(f"RK45: Maximum NFE {max_nfe} reached at t={t:.4f}")
            break
        
        # Adjust step size to not overshoot
        if reverse_time:
            dt = max(-max_step, min(-min_step, min(dt, target_t - t)))
        else:
            dt = min(max_step, max(min_step, max(dt, target_t - t)))
        
        # Compute RK45 stages
        k = []
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        k.append(velocity_fn(x, t_tensor, data))
        stats.nfe += 1
        
        for i in range(1, 7):
            x_stage = x.clone()
            for j, a_ij in enumerate(a[i]):
                x_stage = x_stage + dt * a_ij * k[j]
            
            t_stage = t + c[i] * dt
            t_stage_tensor = torch.full((x.shape[0],), t_stage, device=x.device, dtype=torch.float32)
            k.append(velocity_fn(x_stage, t_stage_tensor, data))
            stats.nfe += 1
        
        # 5th order solution
        x_new = x.clone()
        for i, b_i in enumerate(b):
            x_new = x_new + dt * b_i * k[i]
        
        # 4th order solution (for error estimate)
        x_star = x.clone()
        for i, b_star_i in enumerate(b_star):
            x_star = x_star + dt * b_star_i * k[i]
        
        # Error estimate
        error = torch.abs(x_new - x_star)
        scale = atol + rtol * torch.maximum(torch.abs(x), torch.abs(x_new))
        error_norm = torch.sqrt(torch.mean((error / scale) ** 2))
        
        # Accept or reject step
        if error_norm <= 1.0:
            # Accept step
            x = x_new
            t = t + dt
            stats.accepted_steps += 1
            stats.step_sizes.append(abs(dt))
            
            if save_trajectory:
                trajectory.append(x.clone())
        else:
            # Reject step
            stats.failed_steps += 1
        
        # Adjust step size for next step
        if error_norm > 0:
            dt_new = dt * min(5.0, max(0.2, 0.9 * (1.0 / error_norm) ** 0.2))
            dt = dt_new
        
        # Clamp step size
        if reverse_time:
            dt = max(-max_step, min(-min_step, dt))
        else:
            dt = min(max_step, max(min_step, dt))
    
    stats.total_time = time.time() - start_time
    
    info = {
        'solver': 'rk45',
        'nfe': stats.nfe,
        'trajectory': trajectory if save_trajectory else None,
        'stats': stats.summary()
    }
    
    return x, info


def integrate_ode(
    velocity_fn: Callable,
    x1: torch.Tensor,
    data: Dict,
    solver_type: str = 'rk4',
    nfe: int = 32,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    max_step: Optional[float] = None,
    min_step: float = 1e-4,
    reverse_time: bool = True,
    save_trajectory: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Unified interface for ODE integration.
    
    Args:
        velocity_fn: Function that computes v(x, t)
        x1: Initial state
        data: Conditioning data
        solver_type: 'heun', 'rk4', or 'rk45'
        nfe: Number of function evaluations (for fixed-step solvers)
        rtol, atol: Tolerances for adaptive solver
        max_step, min_step: Step size limits for adaptive solver
        reverse_time: Integration direction
        save_trajectory: Whether to save intermediate states
        
    Returns:
        x0: Final state
        info: Solver information dictionary
    """
    stats = ODESolverStats()
    
    if solver_type == 'heun':
        return heun_solver(velocity_fn, x1, data, nfe, reverse_time, save_trajectory, stats)
    elif solver_type == 'rk4':
        return rk4_solver(velocity_fn, x1, data, nfe, reverse_time, save_trajectory, stats)
    elif solver_type == 'rk45':
        return rk45_adaptive_solver(velocity_fn, x1, data, rtol, atol, max_step, 
                                    min_step, reverse_time, save_trajectory, stats)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

