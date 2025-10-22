"""
Flow Matching Core Mixin

This mixin provides core flow matching functionality similar to DiffusionCoreMixin.
It implements the training and sampling logic for conditional flow matching.

Key differences from DDPM:
- Uses continuous time t ∈ [0,1] instead of discrete timesteps
- Predicts velocity field v(x,t) instead of noise
- Simpler training objective: regress to (x1 - x0)
- Faster sampling: typically needs 10-50 steps vs 100-1000 for DDPM
"""

import torch
from typing import Dict, Tuple, Optional
import logging
from .flow_matching_utils import OptimalTransportFlow, make_flow_matching_sampler


class FlowMatchingCoreMixin:
    """Core flow matching logic mixin.

    This class is framework-independent and only requires subclasses to implement:
    - velocity_model: network with .condition and __call__(x_t, t, data) methods
    - sampler_type: str ('euler' or 'heun')
    - num_sampling_steps: int
    - sigma_min: float
    - device: torch.device
    """

    # Pose vector slice constants (same as DDPM for compatibility)
    TRANSLATION_SLICE = slice(0, 3)
    QPOS_SLICE = slice(3, 19)
    ROTATION_SLICE = slice(19, None)

    def _init_flow_matching(self):
        """Initialize flow matching components. Call this in __init__."""
        # Create OT flow matching instance
        self.ot_flow = OptimalTransportFlow(sigma_min=self.sigma_min)

        # Create sampler for inference
        self.sampler = make_flow_matching_sampler(
            sampler_type=self.sampler_type,
            num_steps=self.num_sampling_steps,
            sigma_min=self.sigma_min
        )

    # --------------------
    # Flow Matching Core Logic
    # --------------------

    def sample_time(self, batch_size: int) -> torch.Tensor:
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample

        Returns:
            t: (B,) - Continuous time values in [0, 1]
        """
        return self.ot_flow.sample_time(batch_size, self.device)

    def sample_conditional_flow(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from conditional flow q(x_t | x_0, x_1) and get target velocity.

        Args:
            x1: (B, num_grasps, D) or (B, D) - Data samples (normalized grasp poses)
            t: (B,) - Timesteps in [0, 1]
            x0: (B, num_grasps, D) or (B, D) - Prior samples (optional, defaults to Gaussian)

        Returns:
            x_t: (B, num_grasps, D) or (B, D) - Interpolated samples
            u_t: (B, num_grasps, D) or (B, D) - Target velocity field
        """
        return self.ot_flow.sample_conditional_flow(x1, t, x0)

    def compute_velocity_loss(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            pred_velocity: (B, num_grasps, D) - Model prediction
            target_velocity: (B, num_grasps, D) - Target velocity

        Returns:
            loss: Scalar velocity matching loss
        """
        return self.ot_flow.compute_loss(
            pred_velocity,
            target_velocity,
            loss_type=getattr(self, 'loss_type', 'l2')
        )

    def model_predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        data: Dict
    ) -> torch.Tensor:
        """
        Predict velocity from the model.

        Args:
            x_t: (B, num_grasps, D) or (B, D) - Noisy input
            t: (B,) - Continuous timesteps in [0, 1]
            data: Conditioning data dictionary

        Returns:
            velocity: (B, num_grasps, D) or (B, D) - Predicted velocity field
        """
        # Note: velocity_model is expected to be the decoder (DiT or UNet)
        # It's the same model as eps_model in DDPM, just used differently
        return self.velocity_model(x_t, t, data)

    # --------------------
    # Sampling (Inference)
    # --------------------

    @torch.no_grad()
    def sample(
        self,
        data: Dict,
        k: int = 1,
        num_steps: Optional[int] = None,
        use_cfg: bool = None,
        guidance_scale: float = None,
        use_negative_guidance: bool = None,
        negative_guidance_scale: float = None
    ) -> torch.Tensor:
        """
        Sample from the flow model using ODE integration.

        Args:
            data: Dictionary containing conditioning information
            k: Number of independent samples per batch item
            num_steps: Number of integration steps (overrides default)
            use_cfg: Enable classifier-free guidance (optional)
            guidance_scale: CFG scale (optional)
            use_negative_guidance: Enable negative guidance (optional)
            negative_guidance_scale: Negative guidance scale (optional)

        Returns:
            trajectory: (B, k, num_steps+1, num_grasps, D) - Full sampling trajectory
        """
        # Handle CFG parameters
        cfg_params = {
            'use_cfg': use_cfg if use_cfg is not None else getattr(self, 'use_cfg', False),
            'guidance_scale': guidance_scale if guidance_scale is not None else getattr(self, 'guidance_scale', 1.0),
            'use_negative_guidance': use_negative_guidance if use_negative_guidance is not None else getattr(self, 'use_negative_guidance', False),
            'negative_guidance_scale': negative_guidance_scale if negative_guidance_scale is not None else getattr(self, 'negative_guidance_scale', 0.0),
        }

        # Generate k independent samples
        samples = []
        for _ in range(k):
            trajectory = self._sample_single(data, num_steps, **cfg_params)
            samples.append(trajectory)

        # Stack: (k, B, num_steps+1, ...) -> (B, k, num_steps+1, ...)
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def _sample_single(
        self,
        data: Dict,
        num_steps: Optional[int] = None,
        use_cfg: bool = False,
        guidance_scale: float = 1.0,
        use_negative_guidance: bool = False,
        negative_guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Sample a single trajectory from t=0 to t=1.

        Args:
            data: Dictionary with conditioning and shape information
            num_steps: Number of integration steps
            use_cfg: Enable classifier-free guidance
            guidance_scale: CFG scale
            use_negative_guidance: Enable negative guidance
            negative_guidance_scale: Negative guidance scale

        Returns:
            trajectory: (B, num_steps+1, num_grasps, D) - Sampling trajectory
        """
        # Get shape information from data
        if 'norm_pose' not in data:
            raise ValueError('norm_pose required for shape inference')

        norm_pose = data['norm_pose']
        if isinstance(norm_pose, torch.Tensor):
            B, orig_num_grasps, pose_dim = norm_pose.shape

            # Apply grasp count control (same as DDPM)
            target_num_grasps = orig_num_grasps
            if getattr(self, 'fix_num_grasps', False) and getattr(self, 'target_num_grasps', None) is not None:
                target_num_grasps = self.target_num_grasps
                if target_num_grasps != orig_num_grasps:
                    logging.info(f"FlowMatching: Adjusting grasp count from {orig_num_grasps} to {target_num_grasps}")

            # Sample from prior: x_0 ~ N(0, I)
            if target_num_grasps == orig_num_grasps:
                x0 = torch.randn_like(norm_pose, device=self.device)
            else:
                x0 = torch.randn(B, target_num_grasps, pose_dim, device=self.device)
        else:
            # Handle list/other types
            pose_dim = norm_pose[0].shape[-1]
            x0 = torch.randn(len(norm_pose), pose_dim, device=self.device)

        # Compute conditioning features
        condition_dict = self.velocity_model.condition(data)
        data.update(condition_dict)

        # Create velocity prediction function with optional CFG
        if use_cfg and not self.training:
            def velocity_fn(x_t, t_batch):
                return self._predict_velocity_cfg(
                    x_t, t_batch, data,
                    guidance_scale, use_negative_guidance, negative_guidance_scale
                )
        else:
            def velocity_fn(x_t, t_batch):
                return self.model_predict_velocity(x_t, t_batch, data)

        # Integrate ODE using the sampler
        steps = num_steps if num_steps is not None else self.num_sampling_steps
        trajectory = self._integrate_ode(x0, velocity_fn, steps)

        return trajectory

    def _integrate_ode(
        self,
        x0: torch.Tensor,
        velocity_fn,
        num_steps: int
    ) -> torch.Tensor:
        """
        Integrate the ODE dx/dt = v(x,t) from t=0 to t=1.

        Args:
            x0: (B, num_grasps, D) or (B, D) - Initial prior sample
            velocity_fn: Function that computes velocity given (x_t, t_batch)
            num_steps: Number of integration steps

        Returns:
            trajectory: (B, num_steps+1, num_grasps, D) - Full trajectory
        """
        dt = 1.0 / num_steps
        x_t = x0
        trajectory = [x_t]

        # Determine sampler type
        sampler_type = getattr(self, 'sampler_type', 'euler').lower()

        for step in range(num_steps):
            t = step / num_steps
            t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.float32)

            if sampler_type == 'heun':
                # Heun's method (2nd order)
                v1 = velocity_fn(x_t, t_batch)
                x_pred = x_t + dt * v1

                t_next = min((step + 1) / num_steps, 1.0)
                t_next_batch = torch.full((x_t.shape[0],), t_next, device=self.device, dtype=torch.float32)
                v2 = velocity_fn(x_pred, t_next_batch)

                x_t = x_t + dt * 0.5 * (v1 + v2)
            else:
                # Euler method (1st order)
                v_t = velocity_fn(x_t, t_batch)
                x_t = x_t + dt * v_t

            trajectory.append(x_t)

        # Stack trajectory: (num_steps+1, B, ...) -> (B, num_steps+1, ...)
        return torch.stack(trajectory, dim=1)

    @torch.no_grad()
    def _predict_velocity_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        data: Dict,
        guidance_scale: float,
        use_negative_guidance: bool,
        negative_guidance_scale: float
    ) -> torch.Tensor:
        """
        Predict velocity with classifier-free guidance.

        CFG formula:
            v_guided = v_uncond + w * (v_cond - v_uncond)

        With negative guidance:
            v_guided = v_uncond + w_pos * (v_pos - v_uncond) + w_neg * (v_uncond - v_neg)

        Args:
            x_t: (B, num_grasps, D) - Current state
            t: (B,) - Timesteps
            data: Conditioning data
            guidance_scale: Positive guidance weight
            use_negative_guidance: Whether to use negative prompts
            negative_guidance_scale: Negative guidance weight

        Returns:
            v_guided: (B, num_grasps, D) - Guided velocity
        """
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}")

        B, num_grasps, _ = x_t.shape

        # Prepare expanded inputs for batched prediction
        num_repeats = 3 if use_negative_guidance else 2
        x_t_expanded = x_t.unsqueeze(0).expand(num_repeats, *x_t.shape).reshape(-1, *x_t.shape[1:])
        t_expanded = t.unsqueeze(0).expand(num_repeats, *t.shape).reshape(-1)

        # Prepare CFG data (similar to DDPM)
        data_expanded = self._prepare_cfg_data(data, B, use_negative_guidance)

        # Predict velocities
        v_all = self.model_predict_velocity(x_t_expanded, t_expanded, data_expanded)

        if use_negative_guidance:
            v_uncond, v_pos, v_neg = v_all.chunk(3, dim=0)
            v_guided = v_uncond + guidance_scale * (v_pos - v_uncond) + negative_guidance_scale * (v_uncond - v_neg)
        else:
            v_uncond, v_pos = v_all.chunk(2, dim=0)
            v_guided = v_uncond + guidance_scale * (v_pos - v_uncond)

        if input_dim == 2:
            v_guided = v_guided.squeeze(1)

        return v_guided

    def _prepare_cfg_data(self, data: Dict, B: int, use_negative_guidance: bool) -> Dict:
        """
        Prepare data for classifier-free guidance (batched prediction).

        Creates [uncond, positive, (optional) negative] batches.

        Args:
            data: Original conditioning data
            B: Batch size
            use_negative_guidance: Whether to include negative prompts

        Returns:
            cfg_data: Expanded data dictionary
        """
        num_repeats = 3 if use_negative_guidance else 2
        cfg_data = {}

        def _repeat_tensor(tensor: torch.Tensor) -> torch.Tensor:
            expanded = tensor.unsqueeze(0).expand(num_repeats, *tensor.shape)
            return expanded.reshape(-1, *tensor.shape[1:])

        # Repeat scene and pose data
        for key in ['scene_pc', 'norm_pose', 'scene_cond']:
            if key in data and isinstance(data[key], torch.Tensor):
                cfg_data[key] = _repeat_tensor(data[key])
            elif key in data:
                cfg_data[key] = data[key] * num_repeats

        # Handle text conditioning
        if 'text_cond' in data and data['text_cond'] is not None:
            text_cond = data['text_cond']
            uncond_text = torch.zeros_like(text_cond)
            pos_text = text_cond

            if use_negative_guidance:
                neg_text = data.get('neg_pred', torch.zeros_like(text_cond))
                cfg_data['text_cond'] = torch.cat([uncond_text, pos_text, neg_text], dim=0)
            else:
                cfg_data['text_cond'] = torch.cat([uncond_text, pos_text], dim=0)
        else:
            cfg_data['text_cond'] = None

        # Handle masks and negative features
        if use_negative_guidance:
            for key in ['neg_pred', 'neg_text_features', 'text_mask']:
                if key in data and data[key] is not None:
                    cfg_data[key] = _repeat_tensor(data[key]) if isinstance(data[key], torch.Tensor) else data[key]
        elif 'text_mask' in data and data['text_mask'] is not None:
            cfg_data['text_mask'] = _repeat_tensor(data['text_mask']) if isinstance(data['text_mask'], torch.Tensor) else data['text_mask']

        return cfg_data

    # --------------------
    # Utilities
    # --------------------

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps for training. Alias for sample_time() for compatibility.

        Args:
            batch_size: Number of timesteps to sample

        Returns:
            t: (B,) - Continuous timesteps in [0, 1]
        """
        return self.sample_time(batch_size)
