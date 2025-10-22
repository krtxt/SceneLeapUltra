import logging
from statistics import mean
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from models.utils.log_colors import BLUE, ENDC, GREEN, HEADER
from models.utils.prediction import build_pred_dict_adaptive
from utils.hand_helper import (
    denorm_hand_pose_robust,
    process_hand_pose,
    process_hand_pose_test,
)


class BaseGraspGeneratorLightning(pl.LightningModule):
    """通用的抓取生成 LightningModule 基类。

    子类需要实现以下接口：
      - self.criterion: GraspLossPose 实例
      - self.loss_weights: dict, 用于验证/测试阶段的损失加权
      - self.batch_size: int, 当前批大小（用于日志记录）
      - self.print_freq: int, 训练阶段的日志频率
      - self.optimizer_cfg, self.scheduler: 优化器与调度器配置
      - self.use_cfg / self.guidance_scale / self.use_negative_guidance / self.negative_guidance_scale:
        控制条件引导
      - self.sample(batch, k=1, ...): 推理采样接口
      - self._compute_loss(batch, mode): 训练与验证阶段计算损失
    """

    def __init__(self) -> None:
        super().__init__()

    # =====================
    # PyTorch Lightning Hooks
    # =====================

    def training_step(self, batch: Dict, batch_idx: int):
        """复用 DDPM 训练日志逻辑，依赖子类实现 _compute_loss。"""
        loss, loss_dict, processed_batch = self._compute_loss(batch, mode="train")

        norm_pose = processed_batch["norm_pose"]
        B, num_grasps, _ = norm_pose.shape
        total_samples = B * num_grasps

        train_log_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        train_log_dict["train/total_loss"] = loss
        self.log_dict(
            train_log_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            batch_size=total_samples,
            sync_dist=True,
        )

        optimizer = self.optimizers()
        if optimizer is not None:
            self.log(
                "train/lr",
                optimizer.param_groups[0]["lr"],
                prog_bar=False,
                logger=True,
                on_step=True,
                batch_size=total_samples,
                sync_dist=True,
            )

        if getattr(self, "_log_gradients", False) and batch_idx % getattr(
            self, "_gradient_freq", 1000
        ) == 0:
            total_norm = 0.0
            param_count = 0
            for p in self.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_norm = total_norm ** 0.5
                self.log(
                    "train/grad_norm",
                    total_norm,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    batch_size=total_samples,
                    sync_dist=True,
                )

        if getattr(self, "_monitor_system", False) and batch_idx % getattr(
            self, "_system_freq", 500
        ) == 0:
            if torch.cuda.is_available():
                self.log(
                    "system/gpu_memory_allocated_gb",
                    torch.cuda.memory_allocated() / 1024 ** 3,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    batch_size=total_samples,
                )
                self.log(
                    "system/gpu_memory_reserved_gb",
                    torch.cuda.memory_reserved() / 1024 ** 3,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    batch_size=total_samples,
                )

            if hasattr(self, "_last_log_time"):
                import time

                current_time = time.time()
                time_diff = current_time - self._last_log_time
                if time_diff > 0:
                    self.log(
                        "system/samples_per_sec",
                        total_samples / time_diff,
                        prog_bar=False,
                        logger=True,
                        on_step=True,
                        batch_size=total_samples,
                    )
            import time

            self._last_log_time = time.time()

        if batch_idx % getattr(self, "print_freq", 250) == 0 and getattr(
            self.trainer, "is_global_zero", True
        ):
            empty_formatter = logging.Formatter("")
            root_logger = logging.getLogger()
            original_formatters = [handler.formatter for handler in root_logger.handlers]

            for handler in root_logger.handlers:
                handler.setFormatter(empty_formatter)
            logging.info("")
            for handler, formatter in zip(root_logger.handlers, original_formatters):
                handler.setFormatter(formatter)

            total_batches = len(self.trainer.train_dataloader)
            logging.info(
                f"{HEADER}Epoch {self.current_epoch} - Batch [{batch_idx}/{total_batches}]{ENDC}"
            )
            logging.info(f"{GREEN}{'Loss:':<21s} {loss.item():.4f}{ENDC}")
            for k, v in loss_dict.items():
                logging.info(f"{BLUE}{k.title() + ':':<21s} {v.item():.4f}{ENDC}")

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)

        B, num_grasps, _ = pred_x0.shape
        batch_size = B * num_grasps

        loss_dict = self.criterion(pred_dict, batch, mode="val")

        if hasattr(self.criterion, "use_standardized_val_loss") and self.criterion.use_standardized_val_loss:
            standard_loss_dict = loss_dict.pop("_standard_losses", {})
            std_weights = getattr(self.criterion, "std_val_weights", {})
            loss = sum(
                v * std_weights.get(k, 0)
                for k, v in loss_dict.items()
                if k in std_weights and std_weights[k] > 0
            )
            standard_loss = sum(
                v * self.loss_weights[k]
                for k, v in standard_loss_dict.items()
                if k in self.loss_weights
            )
            self.log(
                "val/loss",
                loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/standardized_loss",
                loss,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                "val/standard_loss",
                standard_loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.validation_step_outputs.append(
                {
                    "loss": loss.item(),
                    "standardized_loss": loss.item(),
                    "standard_loss": standard_loss.item(),
                    "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                    "standard_loss_dict": {
                        k: v.item() for k, v in standard_loss_dict.items()
                    },
                }
            )
        else:
            loss = sum(
                v * self.loss_weights[k]
                for k, v in loss_dict.items()
                if k in self.loss_weights
            )
            self.log(
                "val/loss",
                loss,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.validation_step_outputs.append(
                {
                    "loss": loss.item(),
                    "loss_dict": {k: v.item() for k, v in loss_dict.items()},
                }
            )

        return {"loss": loss, "loss_dict": loss_dict}

    def on_validation_epoch_end(self):
        val_loss = [x["loss"] for x in self.validation_step_outputs]
        avg_loss = mean(val_loss)

        using_standardized = (
            hasattr(self.criterion, "use_standardized_val_loss")
            and self.criterion.use_standardized_val_loss
            and len(self.validation_step_outputs) > 0
            and "standardized_loss" in self.validation_step_outputs[0]
        )

        if using_standardized:
            std_loss = [x["standardized_loss"] for x in self.validation_step_outputs]
            standard_loss = [x["standard_loss"] for x in self.validation_step_outputs]

            avg_std_loss = mean(std_loss)
            avg_standard_loss = mean(standard_loss)

            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean(
                        [x["loss_dict"][k] for x in self.validation_step_outputs]
                    )

            standard_detailed_loss = {}
            if self.validation_step_outputs and "standard_loss_dict" in self.validation_step_outputs[0]:
                for k in self.validation_step_outputs[0]["standard_loss_dict"].keys():
                    standard_detailed_loss[k] = mean(
                        [x["standard_loss_dict"][k] for x in self.validation_step_outputs]
                    )

            std_loss_std = (
                torch.std(torch.tensor(std_loss)).item() if len(std_loss) > 1 else 0.0
            )
            std_loss_min = min(std_loss) if std_loss else 0.0
            std_loss_max = max(std_loss) if std_loss else 0.0

            from models.utils.logging_helpers import log_validation_summary

            log_validation_summary(
                epoch=self.current_epoch,
                num_batches=len(self.validation_step_outputs),
                avg_loss=avg_std_loss,
                loss_std=std_loss_std,
                loss_min=std_loss_min,
                loss_max=std_loss_max,
                val_detailed_loss=val_detailed_loss,
            )

            val_log_dict = {f"val/std_{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update(
                {
                    "val/standardized_total_loss": avg_std_loss,
                    "val/std_loss_std": std_loss_std,
                    "val/std_loss_min": std_loss_min,
                    "val/std_loss_max": std_loss_max,
                }
            )

            standard_log_dict = {f"val/orig_{k}": v for k, v in standard_detailed_loss.items()}
            standard_log_dict.update({"val/original_total_loss": avg_standard_loss})
            val_log_dict.update(standard_log_dict)

            self.log_dict(
                val_log_dict,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )

            self.log(
                "val_loss",
                avg_std_loss,
                prog_bar=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )
            self.log(
                "val_loss_standardized",
                avg_std_loss,
                prog_bar=False,
                batch_size=self.batch_size,
                sync_dist=True,
            )
            self.log(
                "val_loss_original",
                avg_standard_loss,
                prog_bar=False,
                batch_size=self.batch_size,
                sync_dist=True,
            )
        else:
            val_detailed_loss = {}
            if self.validation_step_outputs:
                for k in self.validation_step_outputs[0]["loss_dict"].keys():
                    val_detailed_loss[k] = mean(
                        [x["loss_dict"][k] for x in self.validation_step_outputs]
                    )

            num_batches = len(self.validation_step_outputs)
            loss_std = (
                torch.std(torch.tensor(val_loss)).item() if len(val_loss) > 1 else 0.0
            )
            loss_min = min(val_loss) if val_loss else 0.0
            loss_max = max(val_loss) if val_loss else 0.0

            from models.utils.logging_helpers import log_validation_summary

            log_validation_summary(
                epoch=self.current_epoch,
                num_batches=num_batches,
                avg_loss=avg_loss,
                loss_std=loss_std,
                loss_min=loss_min,
                loss_max=loss_max,
                val_detailed_loss=val_detailed_loss,
            )

            val_log_dict = {f"val/{k}": v for k, v in val_detailed_loss.items()}
            val_log_dict.update(
                {
                    "val/total_loss": avg_loss,
                    "val/loss_std": loss_std,
                    "val/loss_min": loss_min,
                    "val/loss_max": loss_max,
                }
            )
            self.log_dict(
                val_log_dict,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )

            self.log(
                "val_loss",
                avg_loss,
                prog_bar=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )

        self.log(
            "val/epoch",
            float(self.current_epoch),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )
        self.log(
            "val/num_batches",
            float(len(self.validation_step_outputs)),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            batch_size=self.batch_size,
            sync_dist=True,
        )

        if hasattr(self, "lr_schedulers") and self.lr_schedulers():
            current_lr = (
                self.lr_schedulers().get_last_lr()[0]
                if hasattr(self.lr_schedulers(), "get_last_lr")
                else self.optimizers().param_groups[0]["lr"]
            )
            self.log(
                "val/lr",
                current_lr,
                prog_bar=False,
                logger=True,
                on_epoch=True,
                batch_size=self.batch_size,
                sync_dist=True,
            )

        self.validation_step_outputs.clear()

    def on_test_start(self):
        self.metric_results = []

    def test_step(self, batch: Dict, batch_idx: int):
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        B, num_grasps, _ = pred_x0.shape

        metric_dict, metric_details = self.criterion(pred_dict, batch, mode="test")
        self.metric_results.append(metric_details)
        return metric_dict

    def test_step_teaser(self, batch: Dict):
        B = self.batch_size
        batch = process_hand_pose_test(batch, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(batch)[:, 0, -1]

        pred_dict = build_pred_dict_adaptive(pred_x0)
        metric_dict, metric_details = self.criterion.forward_metric(pred_dict, batch)
        return metric_dict, metric_details

    # ========================
    # Optimizer Configuration
    # ========================

    def configure_optimizers(self):
        optimizer_name = self.optimizer_cfg.name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )
        else:
            raise NotImplementedError(f"Optimizer not supported: {self.optimizer_cfg.name}")

        self.last_epoch = self.current_epoch - 1 if self.current_epoch else -1
        if getattr(self, "use_score", False) and not getattr(self, "score_pretrain", False):
            logging.info("Using score model without pretraining; optimizer state will not be loaded.")
            if hasattr(self.trainer, "fit_loop"):
                self.trainer.fit_loop.epoch_progress.current.completed = 0
            self.last_epoch = -1

        logging.info(f"Setting scheduler last_epoch to: {self.last_epoch}")

        scheduler_name = self.scheduler.name.lower()
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler.t_max,
                eta_min=self.scheduler.min_lr,
                last_epoch=self.last_epoch,
            )
        elif scheduler_name == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                self.scheduler.step_size,
                gamma=self.scheduler.step_gamma,
                last_epoch=self.last_epoch,
            )
        else:
            raise NotImplementedError(f"Scheduler not supported: {self.scheduler.name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    # =====================
    # Inference Methods
    # =====================

    def forward_infer(self, data: Dict, k: int = 1, timestep: int = -1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, :, timestep]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        preds_hand, targets_hand = self.criterion.infer_norm_process_dict(pred_dict, data)
        return preds_hand, targets_hand

    def forward_infer_step(self, data: Dict, k: int = 1, timesteps: Optional[list] = None):
        if timesteps is None:
            timesteps = [1, 3, 5, 7, 9, 11, 91, 93, 95, 97, -1]
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)
        results = []
        for timestep in timesteps:
            pred_x0_t = pred_x0[:, :, timestep]
            pred_dict = build_pred_dict_adaptive(pred_x0_t)
            preds_hand, _ = self.criterion.infer_norm_process_dict(pred_dict, data)
            results.append(preds_hand)
        return results

    def forward_get_pose(self, data: Dict, k: int = 1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, :, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        outputs, targets = self.criterion.infer_norm_process_dict_get_pose(pred_dict, data)
        return outputs, targets

    def forward_get_pose_matched(self, data: Dict, k: int = 1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_x0 = self.sample(data, k=k)[:, 0, -1]
        pred_dict = build_pred_dict_adaptive(pred_x0)
        matched_preds, matched_targets, outputs, targets = self.criterion.forward_infer(
            pred_dict, data
        )
        return matched_preds, matched_targets, outputs, targets

    def forward_get_pose_raw(self, data: Dict, k: int = 1):
        data = process_hand_pose_test(data, rot_type=self.rot_type, mode=self.mode)
        pred_pose_norm = self.sample(data, k=k)[:, :, -1]
        hand_model_pose = denorm_hand_pose_robust(pred_pose_norm, self.rot_type, self.mode)
        return hand_model_pose

    def forward_train_instance(self, data: Dict):
        data = process_hand_pose(data, rot_type=self.rot_type, mode=self.mode)
        raise NotImplementedError("forward_train_instance 需由子类实现")

    # =====================
    # Helper Methods
    # =====================

    def _compute_loss(self, batch: Dict, mode: str = "train") -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        raise NotImplementedError("_compute_loss 需由子类实现")

    def sample(
        self,
        data: Dict,
        k: int = 1,
        use_cfg: Optional[bool] = None,
        guidance_scale: Optional[float] = None,
        use_negative_guidance: Optional[bool] = None,
        negative_guidance_scale: Optional[float] = None,
    ):
        raise NotImplementedError("sample 需由子类实现")

    def _build_pred_dict_adaptive(self, pred_x0):
        logging.warning(
            "_build_pred_dict_adaptive is deprecated; use build_pred_dict_adaptive from models.utils.prediction instead."
        )
        return build_pred_dict_adaptive(pred_x0)
