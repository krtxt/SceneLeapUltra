import pytorch_lightning as pl
import torch
import torch.nn as nn
from functools import partial
from typing import Dict, Any
from collections import defaultdict

from models.backbone import build_backbone
from models.utils.helpers import GenericMLP
from models.loss import GraspLossPose
from utils.hand_helper import process_hand_pose, process_hand_pose_test, denorm_hand_pose_robust
import logging
from .utils.log_colors import HEADER, BLUE, GREEN, YELLOW, RED, ENDC, BOLD, UNDERLINE

class GraspCVAELightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        logging.info(f"{GREEN}Initializing GraspCVAELightning model{ENDC}")
        self.save_hyperparameters() # Saves cfg to self.hparams

        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)
        self.criterion = GraspLossPose(cfg.criterion)
        self.loss_weights = cfg.criterion.loss_weights

        self.rot_type = cfg.rot_type
        self.mode = cfg.mode

        self.latent_dim = cfg.encoder.latent_dim
        self.batch_size = cfg.batch_size # Assuming batch_size is in cfg
        self.print_freq = cfg.get('print_freq', 50) # Default print frequency

        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        self.validation_step_outputs = [] # For accumulating validation outputs

    def resample(self, means, log_var):
        batch_size = means.shape[0]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_dim], device=means.device)
        z = eps * std + means
        return z

    def _calculate_loss(self, pred_dict, input_dict, mode='train'):
        loss_dict = self.criterion(pred_dict, input_dict, mode=mode)
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)
        return loss, loss_dict

    def forward(self, batch: Dict, mode: str = 'train') -> Dict:
        """
        Performs a forward pass through the CVAE model.
        This method encapsulates the logic from the original GraspCVAE's 
        forward_train and forward_inference.
        """
        scene_pc = batch["scene_pc"]
        
        if mode == 'train':
            hand_pc = self.criterion.hand_model(
                batch["hand_model_pose"],
                with_surface_points=True,
            )["surface_points"]
            means, log_var, _, fc = self.encoder(hand_pc, scene_pc)
            z = self.resample(means, log_var)
            pred_pose_norm = self.decoder(z, fc)
            
            return {
                "pred_pose_norm": pred_pose_norm,
                "log_var": log_var,
                "means": means,
                "z": z
            }
        else: # 'val' or 'test'
            batch_size = scene_pc.shape[0]
            _, fc = self.encoder.object_backbone(scene_pc) # Note: object_backbone typo in original
            fc = self.encoder.GMP(fc).squeeze(-1)
            z = torch.randn([batch_size, self.latent_dim], device=fc.device)
            pred_pose_norm = self.decoder(z, fc)

            return {
                "pred_pose_norm": pred_pose_norm,
                "log_var": torch.zeros_like(z), # Dummy values for inference
                "means": torch.zeros_like(z),   # Dummy values for inference
                "matched": {
                    "log_var": torch.zeros_like(z),
                    "means": torch.zeros_like(z)
                }
            }

    def training_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
        pred_dict = self.forward(batch, mode='train')
        loss_dict = self.criterion(pred_dict, batch, mode='train')
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)

        self.log("train/total_loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch["scene_pc"].shape[0], sync_dist=True)
        self.log("train/lr", self.optimizers().param_groups[0]['lr'], prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batch["scene_pc"].shape[0])

        # 定义希望在进度条上显示的关键指标
        # 基于 diffuser 的截图和 CVAE 日志的常见损失项
        prog_bar_keys = ["hand_chamfer", "translation", "rotation", "qpos"]

        for k, v_val in loss_dict.items():
            show_on_prog_bar = k in prog_bar_keys
            self.log(f"train/{k}", v_val, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=batch["scene_pc"].shape[0], sync_dist=True)

        if batch_idx % self.print_freq == 0:
            empty_formatter = logging.Formatter('')
            root_logger = logging.getLogger()
            original_formatters = [handler.formatter for handler in root_logger.handlers]
            for handler in root_logger.handlers:
                handler.setFormatter(empty_formatter)
            logging.info("")
            for handler, formatter in zip(root_logger.handlers, original_formatters):
                handler.setFormatter(formatter)
                
            logging.info(f'{HEADER}🌅 Epoch {self.current_epoch} – Batch [{batch_idx}/{len(self.trainer.train_dataloader)}] ☕️ Training... 🚀{ENDC}')
            logging.info(f'{GREEN}{"Total Loss:":<21s} {loss.item():.4f}{ENDC}')
            for k, v in loss_dict.items():
                logging.info(f'{BLUE}{k.title() + ":":<21s} {v.item():.4f}{ENDC}')
        
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_dict = self.forward(batch, mode='val')
        loss_dict = self.criterion(pred_dict, batch, mode='val')
        loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)
        

        self.log_dict({f"val/{k}": v for k,v in loss_dict.items()}, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch["scene_pc"].shape[0], sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batch["scene_pc"].shape[0], sync_dist=True)
        
        self.validation_step_outputs.append({
            "loss": loss.item(),
            "loss_dict": {k: v.item() for k, v in loss_dict.items()}
        })
        return {"loss": loss, "loss_dict": loss_dict}

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.tensor([x["loss"] for x in self.validation_step_outputs]).mean().item()
        
        val_detailed_loss = {}
        if self.validation_step_outputs[0]["loss_dict"]: # Check if loss_dict is not empty
            for k in self.validation_step_outputs[0]["loss_dict"].keys():
                val_detailed_loss[k] = torch.tensor([x["loss_dict"][k] for x in self.validation_step_outputs]).mean().item()
        
        empty_formatter = logging.Formatter('')
        root_logger = logging.getLogger()
        original_formatters = [handler.formatter for handler in root_logger.handlers]
        for handler in root_logger.handlers:
            handler.setFormatter(empty_formatter)
        logging.info("")
        for handler, formatter in zip(root_logger.handlers, original_formatters):
            handler.setFormatter(formatter)
        logging.info(f'{GREEN}🎯 Epoch {self.current_epoch} – Validation Complete! 🎉✨{ENDC}')
        
        avg_loss_str = f"{avg_loss:.4f}"
        logging.info(f'{BLUE}{"Avg Total Loss:":<21s} {avg_loss_str}{ENDC}')
        
        for k, v in val_detailed_loss.items():
            v_str = f"{v:.4f}"
            logging.info(f'{BLUE}{("Avg " + k.title() + ":"):<21s} {v_str}{ENDC}')
        
        self.log('val/loss', avg_loss, prog_bar=False, logger=True, on_epoch=True, batch_size=self.batch_size, sync_dist=True) # Log aggregated val loss with key 'val/loss' and show on prog_bar
        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.metric_results = [] # Using a list to store detailed metrics per sample/batch

    def test_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        scene_pc = batch["scene_pc"]
        batch_size = scene_pc.shape[0]
        
        # In test mode, we use the inference part of the forward method
        # which generates z from a normal distribution.
        _, fc = self.encoder.object_backbone(scene_pc) # Note: object_backbone typo in original
        fc = self.encoder.GMP(fc).squeeze(-1)
        z = torch.randn([batch_size, self.latent_dim], device=fc.device)
        pred_pose_norm = self.decoder(z, fc)

        # The criterion's test method might expect a slightly different pred_dict
        # or might return metrics directly. We adapt this from the original GraspCVAE.test
        pred_dict_for_test = {
            "pred_pose_norm": pred_pose_norm, 
            "log_var": torch.zeros_like(z), # Dummy as per original test
            "means": torch.zeros_like(z),    # Dummy as per original test
            "matched": {
                "log_var": torch.zeros_like(z),
                "means": torch.zeros_like(z)
            }
        }
        
        # The original GraspCVAE.test calls self.criterion.test
        # We assume self.criterion.test returns (metric_dict, pred_hand, target_hand)
        # For Lightning, we primarily care about logging metrics.
        # If pred_hand and target_hand are needed for visualization or further analysis,
        # they can be returned and handled in on_test_epoch_end or by a callback.
        metric_dict, metric_details = self.criterion(pred_dict_for_test, batch, mode='test')
        
        self.log_dict({f"test/{k}": v for k,v in metric_dict.items()}, logger=True, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.metric_results.append(metric_details) # Store for aggregation if needed
        return metric_dict

    def on_test_epoch_end(self):
        if not self.metric_results:
            logging.info(f"{YELLOW}No test outputs to aggregate.{ENDC}")
            return

        # Example aggregation: mean of each metric
        aggregated_metrics = {}
        if self.metric_results:
            # Assuming metric_results contains dicts from metric_details
            # metric_details is a dict of dicts: {sample_key: {metric_name: value}}
            # We need to collect all metric_name values across all samples
            all_metrics_by_name = defaultdict(list)
            for batch_detail in self.metric_results:
                for sample_key, metrics_for_sample in batch_detail.items():
                    for metric_name, value in metrics_for_sample.items():
                        all_metrics_by_name[metric_name].append(value)

            for k, values in all_metrics_by_name.items():
                if values:
                    if isinstance(values[0], torch.Tensor):
                        aggregated_metrics[f"test_epoch_mean_{k}"] = torch.stack(values).mean().item()
                    elif isinstance(values[0], (int, float)):
                        aggregated_metrics[f"test_epoch_mean_{k}"] = sum(values) / len(values)


        empty_formatter = logging.Formatter('')
        root_logger = logging.getLogger()
        original_formatters = [handler.formatter for handler in root_logger.handlers]
        for handler in root_logger.handlers:
            handler.setFormatter(empty_formatter)
        logging.info("")
        for handler, formatter in zip(root_logger.handlers, original_formatters):
            handler.setFormatter(formatter)
            
        logging.info(f"{BOLD}📊 Test Results Summary:{ENDC}")
        for k, v in aggregated_metrics.items():
            v_str = f"{v:.4f}"
            logging.info(f'{BLUE}{k.title().replace("_", " ") + ":":<30s}{ENDC} {GREEN}{v_str}{ENDC}')
        
        self.log_dict(aggregated_metrics, logger=True) # Log aggregated test metrics
        self.metric_results.clear()

    def configure_optimizers(self):
        if self.optimizer_cfg.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.get("weight_decay", 0.0) # Provide default
            )
        elif self.optimizer_cfg.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.get("weight_decay", 0.0001) # Provide default
            )
        else:
            raise NotImplementedError(f"Optimizer {self.optimizer_cfg.name} not implemented.")

        if self.scheduler_cfg.name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.t_max,
                eta_min=self.scheduler_cfg.get("min_lr", 0.0) # Provide default
            )
        elif self.scheduler_cfg.name.lower() == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_cfg.step_size,
                gamma=self.scheduler_cfg.gamma
            )
        else:
            # Return only optimizer if no scheduler or scheduler not recognized
            logging.warning(f"Scheduler {self.scheduler_cfg.name} not recognized or not specified. Using optimizer only.")
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.scheduler_cfg.get("monitor", "val/total_loss_epoch"), # Default monitor
                "interval": "epoch",
                "frequency": 1
            }
        }

    def forward_get_pose_matched(self, data: Dict, k=4):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred = self.forward(data, mode='val')["pred_pose_norm"]

        pred_dict = {
                "pred_pose_norm": pred,
                "qpos_norm": pred[..., 3:19],
                "translation_norm": pred[..., :3],
                "rotation": pred[..., 19:]
            }
        
        matched_pred, matched_targets, outputs, targets = self.criterion.infer_norm_process_dict_get_pose_matched(pred_dict, data)

        return matched_pred, matched_targets, outputs, targets

    def forward_get_pose_raw(self, data: Dict, k=4):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        scene_pc = data["scene_pc"]
        batch_size = scene_pc.shape[0]

        # Get object features (fc_obj) from the encoder's object_backbone.
        # This logic mirrors the 'val'/'test' mode of the main forward method.
        # We use torch.no_grad() here as we are only extracting features for conditioning the decoder,
        # and gradients through this part of the encoder might not be needed for this specific sampling function.
        with torch.no_grad():
            _, fc_obj_features = self.encoder.object_backbone(scene_pc)
            fc_obj = self.encoder.GMP(fc_obj_features).squeeze(-1) # fc_obj shape: [B, condition_dim]

        # Repeat fc_obj k times to condition k different pose samples for each input item.
        # fc_obj has shape [batch_size, C_dim].
        # fc_obj_repeated will have shape [batch_size * k, C_dim].
        fc_obj_repeated = fc_obj.unsqueeze(1).expand(-1, k, -1).reshape(batch_size * k, fc_obj.shape[-1])

        # Sample k different latent codes (z_samples) for each input item.
        # z_samples shape: [batch_size * k, latent_dim]
        z_samples = torch.randn([batch_size * k, self.latent_dim], device=fc_obj.device)

        # Decode k poses for each input item using the respective z_sample and repeated fc_obj.
        # pred_pose_norm_flat shape: [batch_size * k, output_dim]
        pred_pose_norm_flat = self.decoder(z_samples, fc_obj_repeated)

        # Reshape predictions from [batch_size * k, output_dim] to [batch_size, k, output_dim].
        output_dim = pred_pose_norm_flat.shape[-1] # Get the actual output dimension from the decoder's output
        pred_pose_norm_k_samples = pred_pose_norm_flat.view(batch_size, k, output_dim)
        
        # Denormalize the k predicted poses for each input item.
        # It is assumed that denorm_hand_pose_robust can handle the [B, k, DIM] input shape,
        # by applying its transformations to the last dimension(s) for each of the B*k poses.
        hand_model_poses_k_samples = denorm_hand_pose_robust(pred_pose_norm_k_samples, self.rot_type, self.mode)

        return hand_model_poses_k_samples


class Encoder(nn.Module):

    def __init__(self, encoder_cfg):
        super().__init__()
        self.hand_backbone = build_backbone(encoder_cfg.hand_backbone)
        self.object_backbone = build_backbone(encoder_cfg.object_backbone)
        self.GMP = torch.nn.AdaptiveMaxPool1d(1)

        input_size = encoder_cfg.encoder_dim + encoder_cfg.condition_dim
        build_mlp = partial(
            GenericMLP,
            input_dim=input_size,
            hidden_dims=[encoder_cfg.hidden_dim],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=False,
            dropout=encoder_cfg.dropout_prob,
        )
        self.linear_means = build_mlp(output_dim=encoder_cfg.latent_dim)
        self.linear_log_var = build_mlp(output_dim=encoder_cfg.latent_dim)

    def forward(self, x, c):
        _, fx = self.hand_backbone(x)
        _, fc = self.object_backbone(c)
        fx = self.GMP(fx).squeeze(-1)
        fc = self.GMP(fc).squeeze(-1)
        out = torch.cat((fx, fc), dim=-1) 
        means = self.linear_means(out)
        log_vars = self.linear_log_var(out)
        return means, log_vars, fx, fc

class Decoder(nn.Module):

    def __init__(self, decoder_cfg):
        super().__init__()
        self.rot_type = decoder_cfg.rot_type
        if self.rot_type == 'quat':
            self.output_dim = 23
        elif self.rot_type == 'r6d':
            self.output_dim = 25
        else:
            raise ValueError(f'rot_type {self.rot_type} not supported, must be one of [quat, r6d]')
        input_size = decoder_cfg.input_dim + decoder_cfg.condition_dim
        build_mlp = partial(
            GenericMLP,
            input_dim=input_size,
            hidden_dims=[decoder_cfg.hidden_dim],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=False,
            dropout=decoder_cfg.dropout_prob,
        )
        self.MLP = build_mlp(output_dim=self.output_dim)

    def forward(self, z, c):
        z_c = torch.cat((z, c), dim=-1)
        x = self.MLP(z_c)
        return x