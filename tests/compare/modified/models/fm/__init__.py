"""
Flow Matching Module

This module provides core functionality for Flow Matching based generative modeling:
- paths: Probability paths (linear OT, diffusion paths)
- solvers: ODE integration methods (Heun, RK4, RK45)
- guidance: Classifier-free guidance for conditional generation
"""

from .paths import (
    linear_ot_path,
    diffusion_path_vp,
    diffusion_path_ve,
    get_path_fn,
    add_stochasticity
)

from .solvers import (
    heun_solver,
    rk4_solver,
    rk45_adaptive_solver,
    integrate_ode,
    ODESolverStats
)

from .guidance import (
    apply_cfg,
    apply_cfg_basic,
    apply_cfg_clipped,
    apply_cfg_rescaled,
    apply_cfg_adaptive,
    predictor_corrector_step,
    CFGScheduler
)

__all__ = [
    # Paths
    'linear_ot_path',
    'diffusion_path_vp',
    'diffusion_path_ve',
    'get_path_fn',
    'add_stochasticity',
    
    # Solvers
    'heun_solver',
    'rk4_solver',
    'rk45_adaptive_solver',
    'integrate_ode',
    'ODESolverStats',
    
    # Guidance
    'apply_cfg',
    'apply_cfg_basic',
    'apply_cfg_clipped',
    'apply_cfg_rescaled',
    'apply_cfg_adaptive',
    'predictor_corrector_step',
    'CFGScheduler',
]

