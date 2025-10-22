import torch
from typing import Dict


class FlowMatchingCoreMixin:
    """Flow Matching 核心逻辑 Mixin，提供时间采样与流积分函数。"""

    def _sample_times(self, batch_size: int) -> torch.Tensor:
        sampler = getattr(self, "flow_cfg", {}).get("time_sampler", "uniform")
        time_eps = getattr(self, "flow_cfg", {}).get("time_eps", 0.0)
        device = self.device

        if sampler == "uniform":
            t = torch.rand(batch_size, device=device)
        elif sampler == "quadratic":
            t = torch.rand(batch_size, device=device) ** 2
        elif sampler == "sqrt":
            t = torch.sqrt(torch.rand(batch_size, device=device))
        else:
            raise ValueError(f"Unsupported time_sampler: {sampler}")

        if time_eps > 0:
            t = t * (1 - 2 * time_eps) + time_eps
        return t

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        scale = getattr(self, "time_embedding_scale", 1.0)
        return t * scale

    def _sample_base_noise(self, reference: torch.Tensor) -> torch.Tensor:
        std = getattr(self, "base_std", 1.0)
        return torch.randn_like(reference, device=self.device) * std

    def _interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * x0 + t * x1

    def _velocity_target(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        target = x1 - x0
        velocity_scale = getattr(self, "velocity_scale", 1.0)
        if velocity_scale != 1.0:
            target = target * velocity_scale
        return target

    def _prepare_cfg_data(self, data: Dict, use_negative_guidance: bool) -> Dict:
        cfg_data = {}
        num_repeats = 3 if use_negative_guidance else 2

        def _repeat_tensor(tensor: torch.Tensor) -> torch.Tensor:
            expanded = tensor.unsqueeze(0).expand(num_repeats, *tensor.shape)
            return expanded.reshape(-1, *tensor.shape[1:])

        for key in ["scene_pc", "norm_pose", "scene_cond"]:
            if key in data and isinstance(data[key], torch.Tensor):
                cfg_data[key] = _repeat_tensor(data[key])
            elif key in data:
                cfg_data[key] = data[key] * num_repeats

        if "text_cond" in data and data["text_cond"] is not None:
            text_cond = data["text_cond"]
            uncond_text = torch.zeros_like(text_cond)
            pos_text = text_cond
            if use_negative_guidance:
                neg_text = data.get("neg_pred", torch.zeros_like(text_cond))
                cfg_data["text_cond"] = torch.cat([uncond_text, pos_text, neg_text], dim=0)
            else:
                cfg_data["text_cond"] = torch.cat([uncond_text, pos_text], dim=0)
        else:
            cfg_data["text_cond"] = None

        if use_negative_guidance:
            for key in ["neg_pred", "neg_text_features", "text_mask"]:
                if key in data and data[key] is not None:
                    if isinstance(data[key], torch.Tensor):
                        cfg_data[key] = _repeat_tensor(data[key])
                    else:
                        cfg_data[key] = data[key]
        elif "text_mask" in data and data["text_mask"] is not None:
            value = data["text_mask"]
            if isinstance(value, torch.Tensor):
                cfg_data["text_mask"] = _repeat_tensor(value)
            else:
                cfg_data["text_mask"] = value * num_repeats

        return cfg_data

    def _model_forward(self, x: torch.Tensor, t: torch.Tensor, data: Dict) -> torch.Tensor:
        scaled_t = self._scale_timesteps(t)
        return self.eps_model(x, scaled_t, data)

    def _compute_velocity(
        self,
        x: torch.Tensor,
        t_scalar: torch.Tensor,
        data: Dict,
        use_cfg: bool,
        guidance_scale: float,
        use_negative_guidance: bool,
        negative_guidance_scale: float,
    ) -> torch.Tensor:
        if t_scalar.dim() == 0:
            t_batch = torch.full((x.shape[0],), t_scalar.item(), device=self.device)
        else:
            t_batch = t_scalar

        if use_cfg and not self.training:
            return self._compute_velocity_cfg(
                x,
                t_batch,
                data,
                guidance_scale,
                use_negative_guidance,
                negative_guidance_scale,
            )
        return self._model_forward(x, t_batch, data)

    def _compute_velocity_cfg(
        self,
        x: torch.Tensor,
        t_batch: torch.Tensor,
        data: Dict,
        guidance_scale: float,
        use_negative_guidance: bool,
        negative_guidance_scale: float,
    ) -> torch.Tensor:
        B, num_grasps, _ = x.shape
        expand = 3 if use_negative_guidance else 2
        x_expanded = x.unsqueeze(0).expand(expand, *x.shape).reshape(-1, *x.shape[1:])
        t_expanded = t_batch.unsqueeze(0).expand(expand, *t_batch.shape).reshape(-1)
        data_expanded = self._prepare_cfg_data(data, use_negative_guidance)

        velocities = self._model_forward(x_expanded, t_expanded, data_expanded)
        if use_negative_guidance:
            v_uncond, v_pos, v_neg = velocities.chunk(3, dim=0)
            guided = (
                v_uncond
                + guidance_scale * (v_pos - v_uncond)
                + negative_guidance_scale * (v_uncond - v_neg)
            )
        else:
            v_uncond, v_pos = velocities.chunk(2, dim=0)
            guided = v_uncond + guidance_scale * (v_pos - v_uncond)
        return guided

    def _flow_integrate(
        self,
        x0: torch.Tensor,
        data: Dict,
        steps: int,
        method: str,
        use_cfg: bool,
        guidance_scale: float,
        use_negative_guidance: bool,
        negative_guidance_scale: float,
        t_start: float,
        t_end: float,
    ) -> torch.Tensor:
        times = torch.linspace(t_start, t_end, steps + 1, device=self.device)
        x = x0
        path = [x]

        for i in range(steps):
            t_i = times[i]
            dt = times[i + 1] - t_i
            if method == "euler":
                v = self._compute_velocity(x, t_i, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                x = x + dt * v
            elif method == "heun":
                v1 = self._compute_velocity(x, t_i, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                x_euler = x + dt * v1
                v2 = self._compute_velocity(x_euler, t_i + dt, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                x = x + dt * 0.5 * (v1 + v2)
            elif method == "rk4":
                k1 = self._compute_velocity(x, t_i, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                k2 = self._compute_velocity(x + 0.5 * dt * k1, t_i + 0.5 * dt, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                k3 = self._compute_velocity(x + 0.5 * dt * k2, t_i + 0.5 * dt, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                k4 = self._compute_velocity(x + dt * k3, t_i + dt, data, use_cfg, guidance_scale, use_negative_guidance, negative_guidance_scale)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unsupported integrator method: {method}")
            path.append(x)

        return torch.stack(path, dim=1)
