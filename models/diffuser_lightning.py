import torch
from typing import Any, Dict, Tuple
import logging

from models.base_lightning import BaseGraspGeneratorLightning
from models.decoder import build_decoder
from models.loss import GraspLossPose
from models.utils.diffusion_utils import make_schedule_ddpm
from utils.hand_helper import process_hand_pose
from .utils.diffusion_core import DiffusionCoreMixin
from .utils.log_colors import ENDC, GREEN
from .utils.prediction import build_pred_dict_adaptive

class DDPMLightning(DiffusionCoreMixin, BaseGraspGeneratorLightning):
    def __init__(self, cfg):
        super().__init__()
        logging.info(f"{GREEN}Initializing DDPMLightning model{ENDC}")
        self.save_hyperparameters()
        
        self.use_negative_prompts = cfg.get('use_negative_prompts', True)
        
        self.eps_model = build_decoder(cfg.decoder, cfg)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights
        self.rot_type = cfg.rot_type
        self.batch_size = cfg.batch_size
        self.print_freq = cfg.print_freq
        self.use_score = cfg.use_score
        self.score_pretrain = cfg.score_pretrain
        
        self.timesteps = cfg.steps
        self.schedule_cfg = cfg.schedule_cfg
        self.rand_t_type = cfg.rand_t_type
        self.pred_x0 = cfg.pred_x0
        self.mode = cfg.mode
        
        for k, v in make_schedule_ddpm(self.timesteps, **self.schedule_cfg).items():
            self.register_buffer(k, v)
            
        self.optimizer_cfg = cfg.optimizer
        self.scheduler = cfg.scheduler

        self.use_cfg = cfg.use_cfg
        self.guidance_scale = cfg.guidance_scale
        self.use_negative_guidance = cfg.use_negative_guidance and self.use_negative_prompts
        self.negative_guidance_scale = cfg.negative_guidance_scale
        
        # Grasp count control (diffusion-level configuration)
        self.fix_num_grasps = cfg.get('fix_num_grasps', False)
        self.target_num_grasps = cfg.get('target_num_grasps', None)

        # WandB optimization parameters
        wandb_opt = cfg.get('wandb_optimization', {})
        self._log_gradients = wandb_opt.get('log_gradients', False)
        self._gradient_freq = wandb_opt.get('gradient_freq', 1000)
        self._monitor_system = wandb_opt.get('monitor_system', False)
        self._system_freq = wandb_opt.get('system_freq', 500)

    # =====================
    # PyTorch Lightning Hooks
    # =====================

    # training_step 逻辑继承自 BaseGraspGeneratorLightning

    # 验证阶段逻辑继承自 BaseGraspGeneratorLightning

    # 测试与推理接口复用 BaseGraspGeneratorLightning 的实现

    def forward_train_instance(self, data: Dict):
        """Forward pass for training a single instance."""
        data = process_hand_pose(data, rot_type=self.rot_type, mode=self.mode)
        B = data['norm_pose'].shape[0]
        
        ts = self._sample_timesteps(B)
        noise = torch.randn_like(data['norm_pose'], device=self.device)
        x_t = self.q_sample(x0=data['norm_pose'], t=ts, noise=noise)

        condition = self.eps_model.condition(data)
        data["cond"] = condition
        output = self.eps_model(x_t, ts, data)

        if self.pred_x0:
            pred_dict = {"pred_pose_norm": output, "noise": noise}
        else:
            pred_x0 = self._compute_pred_x0_from_noise(x_t, ts, output)
            pred_dict = {"pred_noise": output, "pred_pose_norm": pred_x0, "noise": noise}

        outputs, targets = self.criterion.get_hand_model_pose(pred_dict, data)
        outputs, targets = self.criterion.get_hand(outputs, targets)
        return outputs, targets

    # =====================
    # Helper Methods
    # =====================

    def _compute_loss(self, batch: Dict, mode: str = 'train') -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Unified loss computation logic for training and validation."""
        # Step 1: Process hand pose
        processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        norm_pose = processed_batch['norm_pose']
        B = norm_pose.shape[0]

        # Log input statistics
        if torch.isnan(norm_pose).any():
            logging.error(f"[NaN Detection] NaN in norm_pose after process_hand_pose")
            logging.error(f"  norm_pose shape: {norm_pose.shape}")
            logging.error(f"  NaN count: {torch.isnan(norm_pose).sum().item()}/{norm_pose.numel()}")
            raise RuntimeError("NaN detected in normalized pose input")

        # Step 2: Sample timesteps
        ts = self._sample_timesteps(B)
        logging.debug(f"[Diffusion] Sampled timesteps: min={ts.min().item()}, max={ts.max().item()}, mean={ts.float().mean().item():.2f}")

        # Step 3: Generate noise and sample noisy data
        noise = torch.randn_like(norm_pose, device=self.device)
        if torch.isnan(noise).any() or torch.isinf(noise).any():
            logging.error(f"[NaN Detection] Invalid noise generated")
            logging.error(f"  noise shape: {noise.shape}, NaN: {torch.isnan(noise).sum().item()}, Inf: {torch.isinf(noise).sum().item()}")
            raise RuntimeError("Invalid noise generated")

        x_t = self.q_sample(x0=norm_pose, t=ts, noise=noise)
        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            logging.error(f"[NaN Detection] NaN/Inf in x_t after q_sample")
            logging.error(f"  x_t shape: {x_t.shape}")
            logging.error(f"  x_t stats: min={x_t[~torch.isnan(x_t) & ~torch.isinf(x_t)].min():.6f if (~torch.isnan(x_t) & ~torch.isinf(x_t)).any() else 'all invalid'}, max={x_t[~torch.isnan(x_t) & ~torch.isinf(x_t)].max():.6f if (~torch.isnan(x_t) & ~torch.isinf(x_t)).any() else 'all invalid'}")
            logging.error(f"  NaN count: {torch.isnan(x_t).sum().item()}, Inf count: {torch.isinf(x_t).sum().item()}")
            logging.error(f"  norm_pose stats: min={norm_pose.min():.6f}, max={norm_pose.max():.6f}, mean={norm_pose.mean():.6f}")
            logging.error(f"  noise stats: min={noise.min():.6f}, max={noise.max():.6f}, mean={noise.mean():.6f}, std={noise.std():.6f}")
            logging.error(f"  timesteps: {ts}")
            raise RuntimeError("NaN/Inf detected in noisy sample x_t")

        logging.debug(f"[Diffusion] x_t stats: min={x_t.min():.6f}, max={x_t.max():.6f}, mean={x_t.mean():.6f}, std={x_t.std():.6f}")

        # Step 4: Compute conditioning features
        logging.debug(f"[Conditioning] Computing conditioning features...")
        condition_dict = self.eps_model.condition(processed_batch)

        # Validate conditioning outputs
        if 'scene_cond' in condition_dict and condition_dict['scene_cond'] is not None:
            scene_cond = condition_dict['scene_cond']
            if torch.isnan(scene_cond).any():
                logging.error(f"[NaN Detection] NaN in scene_cond from conditioning")
                logging.error(f"  scene_cond shape: {scene_cond.shape}")
                logging.error(f"  NaN count: {torch.isnan(scene_cond).sum().item()}/{scene_cond.numel()}")
                raise RuntimeError("NaN detected in scene conditioning")
            logging.debug(f"[Conditioning] scene_cond stats: shape={scene_cond.shape}, min={scene_cond.min():.6f}, max={scene_cond.max():.6f}, mean={scene_cond.mean():.6f}")

        if 'text_cond' in condition_dict and condition_dict['text_cond'] is not None:
            text_cond = condition_dict['text_cond']
            if torch.isnan(text_cond).any():
                logging.error(f"[NaN Detection] NaN in text_cond from conditioning")
                logging.error(f"  text_cond shape: {text_cond.shape}")
                logging.error(f"  NaN count: {torch.isnan(text_cond).sum().item()}/{text_cond.numel()}")
                raise RuntimeError("NaN detected in text conditioning")
            logging.debug(f"[Conditioning] text_cond stats: shape={text_cond.shape}, min={text_cond.min():.6f}, max={text_cond.max():.6f}, mean={text_cond.mean():.6f}")

        processed_batch.update(condition_dict)

        # Step 5: Forward through model
        logging.debug(f"[Model Forward] Passing through eps_model...")
        output = self.eps_model(x_t, ts, processed_batch)

        # Validate model output
        if torch.isnan(output).any():
            logging.error(f"[NaN Detection] NaN in model output")
            logging.error(f"  output shape: {output.shape}")
            logging.error(f"  NaN count: {torch.isnan(output).sum().item()}/{output.numel()}")
            raise RuntimeError("NaN detected in model output")
        if torch.isinf(output).any():
            logging.error(f"[NaN Detection] Inf in model output")
            logging.error(f"  output shape: {output.shape}")
            logging.error(f"  Inf count: {torch.isinf(output).sum().item()}/{output.numel()}")
            raise RuntimeError("Inf detected in model output")

        logging.debug(f"[Model Forward] output stats: shape={output.shape}, min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}, std={output.std():.6f}")

        if self.pred_x0:
            pred_dict = {"pred_pose_norm": output, "noise": noise}
        else:
            pred_x0 = self._compute_pred_x0_from_noise(x_t, ts, output)
            pred_dict = {"pred_noise": output, "pred_pose_norm": pred_x0, "noise": noise}

        if self.use_negative_prompts and 'neg_pred' in condition_dict and condition_dict['neg_pred'] is not None:
            pred_dict['neg_pred'] = condition_dict['neg_pred']
            pred_dict['neg_text_features'] = condition_dict['neg_text_features']

        loss_dict = self.criterion(pred_dict, processed_batch, mode=mode)
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)

        return loss, loss_dict, processed_batch

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Ensure lazily-initialized submodules (e.g., text encoder) exist BEFORE Lightning
        restores the state_dict, otherwise keys under `eps_model.text_encoder.*` will be
        reported as Unexpected.
        Also keep the original special handling for `score_heads`.
        """
        # 1) Make sure the text encoder is constructed so its parameters are present
        # Only initialize if text conditioning is enabled
        try:
            if (hasattr(self, 'eps_model') and 
                hasattr(self.eps_model, '_ensure_text_encoder') and
                hasattr(self.eps_model, 'use_text_condition') and
                self.eps_model.use_text_condition):
                self.eps_model._ensure_text_encoder()
                logging.info("Text encoder initialized for checkpoint loading")
            elif hasattr(self, 'eps_model') and hasattr(self.eps_model, 'use_text_condition'):
                logging.info(f"Skipping text encoder initialization (use_text_condition={self.eps_model.use_text_condition})")
        except Exception as e:
            logging.warning(f"Failed to ensure text encoder before loading checkpoint: {e}")

        # 2) Handle optional score model loading behavior
        if hasattr(self, 'use_score') and self.use_score and not self.score_pretrain:
            model_state_dict = self.state_dict()
            checkpoint_state_dict = checkpoint['state_dict']

            # Load all weights except for score_heads
            new_state_dict = {k: v for k, v in checkpoint_state_dict.items() if 'score_heads' not in k}
            model_state_dict.update(new_state_dict)
            self.load_state_dict(model_state_dict)
            logging.info("Loaded checkpoint weights, excluding score_heads.")
        else:
            # Standard checkpoint loading (Lightning will proceed with strict=True)
            logging.info("Loading full checkpoint.")

    def _check_state_dict(self, dict1, dict2):
        """Helper to check for inconsistencies between two state dicts."""
        keys1, keys2 = set(dict1.keys()), set(dict2.keys())
        if keys1 != keys2:
            logging.warning("State dict key mismatch!")
            logging.warning(f"Only in checkpoint: {keys1 - keys2}")
            logging.warning(f"Only in model: {keys2 - keys1}")
            return True
        
        for key in keys1:
            if dict1[key].shape != dict2[key].shape:
                logging.warning(f"Shape mismatch for key '{key}': {dict1[key].shape} vs {dict2[key].shape}")
                return True
        return False

    def _build_pred_dict_adaptive(self, pred_x0):
        """(Deprecated) Backward compatibility wrapper for build_pred_dict_adaptive."""
        logging.warning("_build_pred_dict_adaptive is deprecated; use build_pred_dict_adaptive from models.utils.prediction instead.")
        return build_pred_dict_adaptive(pred_x0)