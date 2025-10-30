import logging
from typing import Dict, Optional

import torch

from models.utils.set_ops import pairwise_cost


class SetMetricsEvaluator:
    """Evaluate set-level coverage, fidelity, and diversity metrics for grasp predictions."""

    def __init__(self, config: Dict, rot_type: str, cost_weights: Dict[str, float]):
        self.enabled = bool(config.get('enable', False))
        self.rot_type = rot_type
        self.cost_weights = {k: float(v) for k, v in (cost_weights or {}).items()}
        self.threshold = float(config.get('cov_threshold', 0.2))
        self.strict_threshold = float(config.get('strict_cov_threshold', self.threshold))
        self.max_samples = int(config.get('max_samples', 0) or 0)

        log_cfg = config.get('log', {}) if isinstance(config, dict) else {}
        self.log_cov = bool(log_cfg.get('cov', True))
        self.log_mmd = bool(log_cfg.get('mmd', True))
        self.log_nnd = bool(log_cfg.get('nnd', True))
        self.log_component_stats = bool(log_cfg.get('component_stats', False))
        self._component_stats_warned = False

    def _subsample(self, preds: torch.Tensor, gts: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.max_samples <= 0:
            return {
                'preds': preds,
                'gts': gts,
                'mask': mask,
            }

        pred_limit = min(self.max_samples, preds.shape[1])
        gt_limit = min(self.max_samples, gts.shape[1])

        return {
            'preds': preds[:, :pred_limit],
            'gts': gts[:, :gt_limit],
            'mask': mask[:, :gt_limit],
        }

    def _build_target_dict(self, norm_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'norm_pose': norm_pose}

    def compute(self, pred_dict: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if not self.enabled:
            return {}

        try:
            preds = pred_dict.get('pred_pose_norm')
            gts = batch.get('norm_pose')
            if preds is None or gts is None:
                logging.debug("[SetMetrics] Missing prediction or target poses; skipping metrics computation.")
                return {}

            if preds.dim() == 2:
                preds = preds.unsqueeze(0)
            if gts.dim() == 2:
                gts = gts.unsqueeze(0)

            valid_mask = (gts.abs().sum(dim=-1) > 0)
            subsampled = self._subsample(preds, gts, valid_mask)
            preds = subsampled['preds']
            gts = subsampled['gts']
            valid_mask = subsampled['mask']

            translation = pred_dict.get('translation_norm')
            if translation is not None and translation.dim() >= 3:
                translation = translation[:, :preds.shape[1]]
            qpos = pred_dict.get('qpos_norm')
            if qpos is not None and qpos.dim() >= 3:
                qpos = qpos[:, :preds.shape[1]]
            rotation = pred_dict.get('rotation')
            if rotation is not None and rotation.dim() >= 3:
                rotation = rotation[:, :preds.shape[1]]

            prediction_components = {
                'pred_pose_norm': preds,
                'translation_norm': translation,
                'qpos_norm': qpos,
                'rotation': rotation,
            }
            target_components = self._build_target_dict(gts)

            with torch.no_grad():
                return_components = self.log_component_stats and (self.log_cov or self.log_mmd)
                cost_result = pairwise_cost(
                    prediction_components,
                    target_components,
                    weights=self.cost_weights,
                    rot_type=self.rot_type,
                    valid_mask=valid_mask,
                    return_components=return_components,
                )

                if return_components:
                    cost_matrix, component_costs = cost_result
                else:
                    cost_matrix = cost_result
                    component_costs = None

                metrics = {}

                if self.log_cov or self.log_mmd:
                    gt_min = cost_matrix.min(dim=1).values
                    gt_min = torch.where(torch.isfinite(gt_min), gt_min, gt_min.new_zeros(gt_min.shape))
                    if self.log_component_stats and component_costs:
                        self._log_component_stats(component_costs, valid_mask)
                    if self.log_cov:
                        coverage = self._compute_coverage(gt_min, valid_mask)
                        metrics['cov@{:.2f}'.format(self.threshold)] = coverage
                        strict_cov = self._compute_strict_coverage(gt_min, valid_mask)
                        metrics['strict_cov@{:.2f}'.format(self.strict_threshold)] = strict_cov
                    if self.log_mmd:
                        mmd = self._compute_mmd(gt_min, valid_mask)
                        metrics['mmd'] = mmd

                if self.log_nnd:
                    nnd_stats = self._compute_nnd(preds, prediction_components)
                    metrics.update(nnd_stats)

                return metrics
        except Exception as exc:
            logging.error(f"[SetMetrics] Error computing metrics: {exc}")
            return {}

    def _log_component_stats(self, component_costs: Dict[str, torch.Tensor], valid_mask: torch.Tensor) -> None:
        try:
            mask = valid_mask.to(next(iter(component_costs.values())).device) if valid_mask is not None else None
            for name, matrix in component_costs.items():
                if matrix is None:
                    continue
                comp_min = matrix.min(dim=1).values
                comp_min = torch.where(torch.isfinite(comp_min), comp_min, comp_min.new_zeros(comp_min.shape))
                if mask is not None:
                    comp_min = torch.where(mask, comp_min, comp_min.new_zeros(comp_min.shape))
                    selected = comp_min[mask]
                else:
                    selected = comp_min.reshape(-1)
                if selected.numel() == 0:
                    mean_val = float('nan')
                    std_val = float('nan')
                else:
                    mean_val = selected.mean().item()
                    std_val = selected.std(unbiased=False).item()
                logging.info(
                    f"[SetMetrics] component '{name}' min-distance mean={mean_val:.6f}, std={std_val:.6f}"
                )
        except Exception as exc:
            if not self._component_stats_warned:
                logging.warning(f"[SetMetrics] Failed to log component stats: {exc}")
                self._component_stats_warned = True

    def _compute_coverage(self, distances: torch.Tensor, valid_mask: torch.Tensor) -> float:
        coverage_hits = (distances <= self.threshold)
        valid_mask = valid_mask.to(coverage_hits.device)
        covered = torch.where(valid_mask, coverage_hits, coverage_hits.new_zeros(coverage_hits.shape, dtype=torch.bool))
        denom = valid_mask.sum(dim=1).clamp(min=1)
        per_sample = covered.float().sum(dim=1) / denom.float()
        return per_sample.mean().item()

    def _compute_strict_coverage(self, distances: torch.Tensor, valid_mask: torch.Tensor) -> float:
        strict_hits = (distances <= self.strict_threshold)
        valid_mask = valid_mask.to(strict_hits.device)
        covered = torch.where(
            valid_mask,
            strict_hits,
            strict_hits.new_zeros(strict_hits.shape, dtype=torch.bool)
        )
        denom = valid_mask.sum(dim=1).clamp(min=1)
        per_sample = covered.float().sum(dim=1) / denom.float()
        return per_sample.mean().item()

    def _compute_mmd(self, distances: torch.Tensor, valid_mask: torch.Tensor) -> float:
        valid_mask = valid_mask.to(distances.device)
        masked = torch.where(valid_mask, distances, distances.new_zeros(distances.shape))
        denom = valid_mask.sum(dim=1).clamp(min=1)
        per_sample = masked.sum(dim=1) / denom.float()
        return per_sample.mean().item()

    def _compute_nnd(self, preds: torch.Tensor, prediction_components: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if preds.shape[1] <= 1:
            return {
                'nnd_mean': 0.0,
                'nnd_std': 0.0,
                'nnd_cv': 0.0,
            }

        self_cost = pairwise_cost(
            prediction_components,
            {'norm_pose': preds},
            weights=self.cost_weights,
            rot_type=self.rot_type,
            valid_mask=None,
        )
        diag = torch.eye(self_cost.shape[1], device=self_cost.device, dtype=torch.bool).unsqueeze(0)
        self_cost = self_cost.masked_fill(diag, float('inf'))
        nearest = self_cost.min(dim=2).values
        nearest = torch.where(torch.isfinite(nearest), nearest, nearest.new_zeros(nearest.shape))

        mean_val = nearest.mean().item()
        std_val = nearest.std().item()
        cv = std_val / (mean_val + 1e-8)

        return {
            'nnd_mean': mean_val,
            'nnd_std': std_val,
            'nnd_cv': cv,
        }
