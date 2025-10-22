"""
Flow Matching Utilities for SceneLeapUltra

This module implements Optimal Transport (OT) Flow Matching for grasp synthesis.
Flow Matching learns a velocity field that transports samples from a prior distribution
to the data distribution, providing a simpler and often faster alternative to DDPM.

Key differences from DDPM:
- Direct regression of velocity field instead of noise prediction
- Simpler training objective without complex noise schedules
- Often requires fewer sampling steps for inference
- Uses straight-line conditional flows (Optimal Transport paths)

References:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport (Tong et al., 2023)
"""

from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings for flow matching.

    Args:
        timesteps: (B,) - Continuous time values in [0, 1]
        embedding_dim: Dimension of the output embedding
        max_period: Controls the minimum frequency of the embeddings

    Returns:
        embeddings: (B, embedding_dim) - Sinusoidal position embeddings
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    )

    # timesteps are in [0, 1] for flow matching
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


class OptimalTransportFlow:
    """
    Optimal Transport (OT) Flow Matching implementation.

    Uses conditional straight-line flows from prior to data:
        x_t = (1 - t) * x_0_prior + t * x_1_data + sigma_t * noise

    The model learns to predict the velocity field:
        v_theta(x_t, t) ≈ dx/dt = x_1 - x_0 + sigma'(t) / sigma(t) * noise

    For simplicity, we use sigma(t) = sigma_min (constant small noise).
    """

    def __init__(self, sigma_min: float = 1e-4):
        """
        Args:
            sigma_min: Minimum noise level for numerical stability
        """
        self.sigma_min = sigma_min

    def sample_time(self, batch_size: int, device: torch.device, eps: float = 1e-3) -> torch.Tensor:
        """
        Sample random timesteps uniformly from [eps, 1].

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensors on
            eps: Small epsilon to avoid t=0 (numerical stability)

        Returns:
            t: (B,) - Sampled timesteps in [eps, 1]
        """
        return torch.rand(batch_size, device=device) * (1 - eps) + eps

    def sample_conditional_flow(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the conditional probability path q(x_t | x_0, x_1).

        For OT flow matching with constant noise:
            x_t = (1-t)*x_0 + t*x_1 + sigma_min*epsilon

        Target velocity:
            u_t = x_1 - x_0

        Args:
            x1: (B, num_grasps, D) or (B, D) - Data samples (target grasp poses)
            t: (B,) - Timesteps in [0, 1]
            x0: (B, num_grasps, D) or (B, D) - Prior samples (default: standard Gaussian)

        Returns:
            x_t: Interpolated samples at time t
            u_t: Target velocity field at time t
        """
        # Handle dimensions
        if x1.dim() == 2:
            x1 = x1.unsqueeze(1)
            squeeze_output = True
        elif x1.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f"Unsupported input dimension: {x1.dim()}")

        B, num_grasps, D = x1.shape

        # Sample from prior if not provided (standard Gaussian)
        if x0 is None:
            x0 = torch.randn_like(x1)
        elif x0.dim() == 2:
            x0 = x0.unsqueeze(1)

        # Expand time dimension for broadcasting
        t_expanded = t.view(B, 1, 1).expand(B, num_grasps, 1)

        # Sample conditional flow with constant noise
        noise = torch.randn_like(x1) * self.sigma_min
        x_t = (1 - t_expanded) * x0 + t_expanded * x1 + noise

        # Compute target velocity (straight-line OT path)
        # u_t = dx/dt = x1 - x0 (for straight-line paths)
        u_t = x1 - x0

        if squeeze_output:
            x_t = x_t.squeeze(1)
            u_t = u_t.squeeze(1)

        return x_t, u_t

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target_velocity: torch.Tensor,
        loss_type: str = 'l2'
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Simple MSE between predicted and target velocity:
            L = E_{t, x_0, x_1} [||v_theta(x_t, t) - u_t||^2]

        Args:
            model_output: (B, num_grasps, D) - Predicted velocity from model
            target_velocity: (B, num_grasps, D) - Target velocity u_t
            loss_type: 'l2' or 'l1'

        Returns:
            loss: Scalar loss value
        """
        if loss_type == 'l2':
            loss = F.mse_loss(model_output, target_velocity)
        elif loss_type == 'l1':
            loss = F.l1_loss(model_output, target_velocity)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss


class EulerSampler:
    """
    Euler method sampler for flow matching ODE integration.

    Given a learned velocity field v_theta(x, t), we solve the ODE:
        dx/dt = v_theta(x, t), x(0) = x_0 ~ N(0, I)

    Using Euler's method:
        x_{t+dt} = x_t + dt * v_theta(x_t, t)
    """

    def __init__(self, num_steps: int = 100, sigma_min: float = 1e-4):
        """
        Args:
            num_steps: Number of integration steps
            sigma_min: Minimum noise level (for consistency with training)
        """
        self.num_steps = num_steps
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        condition_dict: Dict,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample from the flow by integrating the learned velocity field.

        Args:
            model: Model that predicts v_theta(x_t, t, condition)
            x0: (B, num_grasps, D) or (B, D) - Initial noise sample
            condition_dict: Dictionary with conditioning information
            num_steps: Number of integration steps (overrides self.num_steps if provided)

        Returns:
            x_1: (B, num_grasps, D) or (B, D) - Generated samples at t=1
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        # Start from prior
        x_t = x0

        # Integrate from t=0 to t=1
        for step in range(steps):
            t = step / steps
            # Create batch of timesteps
            t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)

            # Predict velocity
            v_t = model(x_t, t_batch, condition_dict)

            # Euler step
            x_t = x_t + dt * v_t

        return x_t

    @torch.no_grad()
    def sample_trajectory(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        condition_dict: Dict,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample and return the full trajectory from t=0 to t=1.

        Args:
            model: Model that predicts v_theta(x_t, t, condition)
            x0: (B, num_grasps, D) or (B, D) - Initial noise sample
            condition_dict: Dictionary with conditioning information
            num_steps: Number of integration steps

        Returns:
            trajectory: (B, num_steps+1, num_grasps, D) - Full trajectory including x_0
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        # Start from prior
        x_t = x0
        trajectory = [x_t]

        # Integrate from t=0 to t=1
        for step in range(steps):
            t = step / steps
            t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)

            # Predict velocity
            v_t = model(x_t, t_batch, condition_dict)

            # Euler step
            x_t = x_t + dt * v_t
            trajectory.append(x_t)

        # Stack trajectory: (num_steps+1, B, ...) -> (B, num_steps+1, ...)
        trajectory = torch.stack(trajectory, dim=1)

        return trajectory


class HeunSampler:
    """
    Heun's method (2nd order) sampler for more accurate ODE integration.

    Heun's method (also called improved Euler):
        k1 = v_theta(x_t, t)
        k2 = v_theta(x_t + dt*k1, t+dt)
        x_{t+dt} = x_t + dt/2 * (k1 + k2)
    """

    def __init__(self, num_steps: int = 50, sigma_min: float = 1e-4):
        """
        Args:
            num_steps: Number of integration steps (typically fewer than Euler for same quality)
            sigma_min: Minimum noise level
        """
        self.num_steps = num_steps
        self.sigma_min = sigma_min

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        condition_dict: Dict,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample using Heun's method (2nd order Runge-Kutta).

        Args:
            model: Model that predicts v_theta(x_t, t, condition)
            x0: (B, num_grasps, D) or (B, D) - Initial noise sample
            condition_dict: Dictionary with conditioning information
            num_steps: Number of integration steps

        Returns:
            x_1: Generated samples at t=1
        """
        steps = num_steps if num_steps is not None else self.num_steps
        dt = 1.0 / steps

        x_t = x0

        for step in range(steps):
            t = step / steps
            t_batch = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.float32)

            # First velocity evaluation
            v1 = model(x_t, t_batch, condition_dict)

            # Predictor step
            x_pred = x_t + dt * v1

            # Second velocity evaluation at predicted point
            t_next = min((step + 1) / steps, 1.0)
            t_next_batch = torch.full((x_t.shape[0],), t_next, device=x_t.device, dtype=torch.float32)
            v2 = model(x_pred, t_next_batch, condition_dict)

            # Corrector step (average of two velocities)
            x_t = x_t + dt * 0.5 * (v1 + v2)

        return x_t


def make_flow_matching_sampler(sampler_type: str = 'euler', num_steps: int = 100, **kwargs):
    """
    Factory function to create flow matching samplers.

    Args:
        sampler_type: 'euler' or 'heun'
        num_steps: Number of integration steps
        **kwargs: Additional arguments for the sampler

    Returns:
        Sampler instance
    """
    if sampler_type.lower() == 'euler':
        return EulerSampler(num_steps=num_steps, **kwargs)
    elif sampler_type.lower() == 'heun':
        return HeunSampler(num_steps=num_steps, **kwargs)
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}. Choose 'euler' or 'heun'.")


if __name__ == '__main__':
    # Test Flow Matching utilities
    print("Testing Flow Matching utilities...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, num_grasps, D = 4, 3, 25

    # Test OT Flow
    ot_flow = OptimalTransportFlow(sigma_min=1e-4)

    # Sample data
    x1 = torch.randn(B, num_grasps, D, device=device)
    t = ot_flow.sample_time(B, device=device)

    print(f"\nSampled timesteps: {t}")

    # Sample conditional flow
    x_t, u_t = ot_flow.sample_conditional_flow(x1, t)
    print(f"x_t shape: {x_t.shape}")
    print(f"u_t shape: {u_t.shape}")

    # Test loss computation
    pred_velocity = torch.randn_like(u_t)
    loss = ot_flow.compute_loss(pred_velocity, u_t)
    print(f"Flow matching loss: {loss.item():.6f}")

    # Test single grasp format
    x1_single = torch.randn(B, D, device=device)
    t_single = ot_flow.sample_time(B, device=device)
    x_t_single, u_t_single = ot_flow.sample_conditional_flow(x1_single, t_single)
    print(f"\nSingle grasp - x_t shape: {x_t_single.shape}")
    print(f"Single grasp - u_t shape: {u_t_single.shape}")

    print("\nFlow Matching utilities test completed!")
