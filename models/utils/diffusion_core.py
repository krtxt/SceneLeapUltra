import torch
from typing import Dict, Tuple
import logging

class DiffusionCoreMixin:
    """Core diffusion model logic mixin.

    This class is framework-independent and only requires subclasses to implement:
    - timesteps: int
    - schedule/buffer tensors: sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod, 
      sqrt_recipm1_alphas_cumprod, posterior_mean_coef1, posterior_mean_coef2, posterior_variance, posterior_log_variance_clipped
    - eps_model: network with .condition and __call__(x_t, t, data) methods
    - pred_x0: bool indicating model output type
    - use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale
    - device: torch.device
    - rand_t_type: str ('all' | 'half')
    """

    # Pose vector slice constants
    TRANSLATION_SLICE = slice(0, 3)
    QPOS_SLICE = slice(3, 19)
    ROTATION_SLICE = slice(19, None)

    # --------------------
    # Diffusion Core Logic
    # --------------------

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise. If x0 is [B, D], automatically handle as [B, 1, D]."""
        input_dim = x0.dim()
        if input_dim == 2:
            x0 = x0.unsqueeze(1)
            if noise.dim() == 2:
                noise = noise.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x0.dim()}. Expected 2 or 3.")

        B, num_grasps, _ = x0.shape
        t_expanded = t.unsqueeze(1).expand(-1, num_grasps)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_expanded].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_expanded].unsqueeze(-1)
        x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t.squeeze(1) if input_dim == 2 else x_t

    def model_predict(self, x_t: torch.Tensor, t: torch.Tensor, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict noise or x0 from noise data."""
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}.")

        B, num_grasps, _ = x_t.shape
        model_output = self.eps_model(x_t, t, data)

        t_expanded = t.unsqueeze(1).expand(-1, num_grasps)
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t_expanded].unsqueeze(-1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t_expanded].unsqueeze(-1)

        if self.pred_x0:
            pred_x0 = model_output
            pred_noise = (sqrt_recip * x_t - pred_x0) / sqrt_recipm1
        else:
            pred_noise = model_output
            pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * pred_noise

        if input_dim == 2:
            pred_noise = pred_noise.squeeze(1)
            pred_x0 = pred_x0.squeeze(1)
        return pred_noise, pred_x0

    def _compute_pred_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        """Compute x0 from noise."""
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
            pred_noise = pred_noise.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}.")

        B, num_grasps, _ = x_t.shape
        t_expanded = t.unsqueeze(1).expand(-1, num_grasps)
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t_expanded].unsqueeze(-1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t_expanded].unsqueeze(-1)
        pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * pred_noise
        return pred_x0.squeeze(1) if input_dim == 2 else pred_x0

    def _prepare_cfg_data(self, data: Dict, B: int) -> Dict:
        from .prediction import build_pred_dict_adaptive  

        cfg_data = {}
        num_repeats = 3 if self.use_negative_guidance else 2

        def _repeat_tensor(tensor: torch.Tensor) -> torch.Tensor:
            expanded = tensor.unsqueeze(0).expand(num_repeats, *tensor.shape)
            return expanded.reshape(-1, *tensor.shape[1:])

        for key in ['scene_pc', 'norm_pose', 'scene_cond']:
            if key in data and isinstance(data[key], torch.Tensor):
                cfg_data[key] = _repeat_tensor(data[key])
            elif key in data:
                cfg_data[key] = data[key] * num_repeats

        # 文本条件
        if 'text_cond' in data and data['text_cond'] is not None:
            text_cond = data['text_cond']
            uncond_text = torch.zeros_like(text_cond)
            pos_text = text_cond
            if self.use_negative_guidance:
                neg_text = data.get('neg_pred', torch.zeros_like(text_cond))
                cfg_data['text_cond'] = torch.cat([uncond_text, pos_text, neg_text], dim=0)
            else:
                cfg_data['text_cond'] = torch.cat([uncond_text, pos_text], dim=0)
        else:
            cfg_data['text_cond'] = None

        if self.use_negative_guidance:
            for key in ['neg_pred', 'neg_text_features', 'text_mask']:
                if key in data and data[key] is not None:
                    cfg_data[key] = _repeat_tensor(data[key]) if isinstance(data[key], torch.Tensor) else data[key]
        elif 'text_mask' in data and data['text_mask'] is not None:
             cfg_data['text_mask'] = _repeat_tensor(data['text_mask']) if isinstance(data['text_mask'], torch.Tensor) else data['text_mask']
             
        return cfg_data

    # --------------------
    # Sampling
    # --------------------

    @torch.no_grad()
    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, data: Dict):
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}.")

        pred_noise, pred_x0 = self.model_predict(x_t, t, data)
        B, num_grasps, _ = x_t.shape
        t_expanded = t.unsqueeze(1).expand(-1, num_grasps)
        coef1 = self.posterior_mean_coef1[t_expanded].unsqueeze(-1)
        coef2 = self.posterior_mean_coef2[t_expanded].unsqueeze(-1)
        model_mean = coef1 * pred_x0 + coef2 * x_t
        posterior_variance = self.posterior_variance[t_expanded].unsqueeze(-1)
        posterior_log_variance = self.posterior_log_variance_clipped[t_expanded].unsqueeze(-1)

        if input_dim == 2:
            model_mean = model_mean.squeeze(1)
            posterior_variance = posterior_variance.squeeze(1)
            posterior_log_variance = posterior_log_variance.squeeze(1)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_mean_variance_cfg(self, x_t: torch.Tensor, t: torch.Tensor, data: Dict, guidance_scale: float, use_negative_guidance: bool, negative_guidance_scale: float):
        input_dim = x_t.dim()
        if input_dim == 2:
            x_t = x_t.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Unsupported input dimension: {x_t.dim()}.")

        B, num_grasps, _ = x_t.shape
        x_t_expanded_size = 3 if use_negative_guidance else 2
        x_t_expanded = x_t.unsqueeze(0).expand(x_t_expanded_size, *x_t.shape).reshape(-1, *x_t.shape[1:])
        t_expanded = t.unsqueeze(0).expand(x_t_expanded_size, *t.shape).reshape(-1)
        data_expanded = self._prepare_cfg_data(data, B)

        pred_noise_all, pred_x0_all = self.model_predict(x_t_expanded, t_expanded, data_expanded)
        
        if use_negative_guidance:
            pred_noise_uncond, pred_noise_pos, pred_noise_neg = pred_noise_all.chunk(3, dim=0)
            pred_x0_uncond, pred_x0_pos, pred_x0_neg = pred_x0_all.chunk(3, dim=0)
            guided_noise = pred_noise_uncond + guidance_scale * (pred_noise_pos - pred_noise_uncond) + negative_guidance_scale * (pred_noise_uncond - pred_noise_neg)
            guided_x0 = pred_x0_uncond + guidance_scale * (pred_x0_pos - pred_x0_uncond) + negative_guidance_scale * (pred_x0_uncond - pred_x0_neg)
        else:
            pred_noise_uncond, pred_noise_pos = pred_noise_all.chunk(2, dim=0)
            pred_x0_uncond, pred_x0_pos = pred_x0_all.chunk(2, dim=0)
            guided_noise = pred_noise_uncond + guidance_scale * (pred_noise_pos - pred_noise_uncond)
            guided_x0 = pred_x0_uncond + guidance_scale * (pred_x0_pos - pred_x0_uncond)

        t_expanded_orig = t.unsqueeze(1).expand(-1, num_grasps)
        coef1 = self.posterior_mean_coef1[t_expanded_orig].unsqueeze(-1)
        coef2 = self.posterior_mean_coef2[t_expanded_orig].unsqueeze(-1)
        model_mean = coef1 * guided_x0 + coef2 * x_t
        posterior_variance = self.posterior_variance[t_expanded_orig].unsqueeze(-1)
        posterior_log_variance = self.posterior_log_variance_clipped[t_expanded_orig].unsqueeze(-1)

        if input_dim == 2:
            model_mean = model_mean.squeeze(1)
            posterior_variance = posterior_variance.squeeze(1)
            posterior_log_variance = posterior_log_variance.squeeze(1)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, data: Dict, use_cfg: bool, guidance_scale: float, use_negative_guidance: bool, negative_guidance_scale: float) -> torch.Tensor:
        B = x_t.shape[0]
        batch_timestep = torch.full((B,), t, device=self.device, dtype=torch.long)
        if use_cfg and not self.training:
            model_mean, model_variance, model_log_variance = self.p_mean_variance_cfg(x_t, batch_timestep, data, guidance_scale, use_negative_guidance, negative_guidance_scale)
        else:
            model_mean, model_variance, model_log_variance = self.p_mean_variance(x_t, batch_timestep, data)

        noise = torch.randn_like(x_t) if t > 0 else 0.0
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_x

    @torch.no_grad()
    def p_sample_loop(self, data: Dict, use_cfg: bool, guidance_scale: float, use_negative_guidance: bool, negative_guidance_scale: float):
        if 'norm_pose' not in data:
            raise ValueError('norm_pose required')
        norm_pose = data['norm_pose']
        if isinstance(norm_pose, torch.Tensor):
            B, orig_num_grasps, pose_dim = norm_pose.shape

            target_num_grasps = orig_num_grasps
            # Apply grasp count control at diffusion level
            if getattr(self, 'fix_num_grasps', False) and getattr(self, 'target_num_grasps', None) is not None:
                target_num_grasps = self.target_num_grasps
                if target_num_grasps != orig_num_grasps:
                    logging.info(f"Diffusion: Adjusting grasp count from {orig_num_grasps} to {target_num_grasps} "
                               f"({'training' if self.training else 'inference'})")

            if target_num_grasps == orig_num_grasps:
                x_t = torch.randn_like(norm_pose, device=self.device)
            else:
                x_t = torch.randn(B, target_num_grasps, pose_dim, device=self.device)
        else:
            # norm_pose 是列表或其他类型，仅保持最后维度一致
            pose_dim = norm_pose[0].shape[-1]
            x_t = torch.randn(len(norm_pose), pose_dim, device=self.device)

        condition_dict = self.eps_model.condition(data)
        data.update(condition_dict)

        all_x_t = [x_t]
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t, t, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
            all_x_t.append(x_t)
        return torch.stack(all_x_t, dim=1)

    @torch.no_grad()
    def sample(self, data: Dict, k: int = 1, use_cfg: bool = None, guidance_scale: float = None, use_negative_guidance: bool = None, negative_guidance_scale: float = None):
        cfg_params = {
            'use_cfg': use_cfg if use_cfg is not None else self.use_cfg,
            'guidance_scale': guidance_scale if guidance_scale is not None else self.guidance_scale,
            'use_negative_guidance': use_negative_guidance if use_negative_guidance is not None else self.use_negative_guidance,
            'negative_guidance_scale': negative_guidance_scale if negative_guidance_scale is not None else self.negative_guidance_scale,
        }
        ksamples = [self.p_sample_loop(data, **cfg_params) for _ in range(k)]
        return torch.stack(ksamples, dim=1)

    # --------------------
    # Utilities
    # --------------------

    def _sample_timesteps(self, batch_size: int):
        if self.rand_t_type == 'all':
            return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        elif self.rand_t_type == 'half':
            ts = torch.randint(0, self.timesteps, ((batch_size + 1) // 2,), device=self.device)
            if batch_size % 2:
                return torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            return torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise ValueError(f'Unsupported rand_t_type {self.rand_t_type}') 