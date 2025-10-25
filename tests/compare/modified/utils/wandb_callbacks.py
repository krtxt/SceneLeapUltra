"""
WandB回调函数模块

提供用于PyTorch Lightning的WandB可视化和指标监控回调函数。
支持分布式训练，只在主进程执行记录操作。

主要功能：
1. WandbVisualizationCallback: 模型可视化和样本预测记录
2. WandbMetricsCallback: 参数和梯度直方图记录

特性：
- 分布式训练兼容（只在主进程记录）
- 可配置的记录频率
- 错误处理和日志记录
- 内存友好的实现
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from utils.distributed_utils import is_main_process

logger = logging.getLogger(__name__)


class WandbVisualizationCallback(Callback):
    """
    WandB可视化回调，用于记录模型架构、损失曲线、样本可视化等
    
    只在主进程执行，确保分布式训练兼容性
    """
    
    def __init__(
        self,
        log_model_graph: bool = True,
        log_sample_predictions: bool = True,
        sample_log_freq: int = 5,
        max_samples_to_log: int = 4
    ):
        """
        初始化可视化回调
        
        Args:
            log_model_graph: 是否记录模型架构图
            log_sample_predictions: 是否记录样本预测结果
            sample_log_freq: 样本记录频率（epoch）
            max_samples_to_log: 最大记录样本数
        """
        super().__init__()
        self.log_model_graph = log_model_graph
        self.log_sample_predictions = log_sample_predictions
        self.sample_log_freq = sample_log_freq
        self.max_samples_to_log = max_samples_to_log
        self._model_graph_logged = False
        
        logger.info(f"WandbVisualizationCallback initialized:")
        logger.info(f"  - Model graph logging: {log_model_graph}")
        logger.info(f"  - Sample predictions: {log_sample_predictions}")
        logger.info(f"  - Sample log frequency: {sample_log_freq} epochs")
        logger.info(f"  - Max samples to log: {max_samples_to_log}")
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """训练开始时记录模型架构图"""
        if not is_main_process():
            return
            
        if self.log_model_graph and not self._model_graph_logged:
            self._log_model_graph(trainer, pl_module)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """验证epoch结束时记录样本预测结果"""
        if not is_main_process():
            return
            
        if (self.log_sample_predictions and 
            trainer.current_epoch % self.sample_log_freq == 0):
            self._log_sample_predictions(trainer, pl_module)
    
    def _log_model_graph(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """记录模型架构图"""
        try:
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                # 获取一个样本batch用于模型图记录
                dataloader = trainer.train_dataloader
                if dataloader is not None:
                    sample_batch = next(iter(dataloader))
                    
                    # 将batch移到正确的设备
                    if hasattr(sample_batch, 'keys'):
                        for key in sample_batch:
                            if torch.is_tensor(sample_batch[key]):
                                sample_batch[key] = sample_batch[key].to(pl_module.device)
                    
                    # 记录模型图
                    trainer.logger.experiment.watch(
                        pl_module, 
                        log="all", 
                        log_freq=100,
                        log_graph=True
                    )
                    
                    logger.info("Model graph logged to WandB")
                    self._model_graph_logged = True
                    
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def _log_sample_predictions(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """记录样本预测结果"""
        try:
            if not (trainer.logger and hasattr(trainer.logger, 'experiment')):
                return
                
            # 获取验证数据加载器
            val_dataloader = trainer.val_dataloaders
            if val_dataloader is None:
                return
                
            # 获取一个batch的样本
            sample_batch = next(iter(val_dataloader))
            
            # 将batch移到正确的设备
            if hasattr(sample_batch, 'keys'):
                for key in sample_batch:
                    if torch.is_tensor(sample_batch[key]):
                        sample_batch[key] = sample_batch[key].to(pl_module.device)
            
            # 限制样本数量
            batch_size = len(sample_batch[list(sample_batch.keys())[0]])
            num_samples = min(self.max_samples_to_log, batch_size)
            
            # 提取子集
            sample_subset = {}
            for key, value in sample_batch.items():
                if torch.is_tensor(value):
                    sample_subset[key] = value[:num_samples]
                else:
                    sample_subset[key] = value[:num_samples] if hasattr(value, '__getitem__') else value
            
            # 生成预测
            pl_module.eval()
            with torch.no_grad():
                # 这里可以根据具体模型调整预测逻辑
                if hasattr(pl_module, 'validation_step'):
                    predictions = pl_module.validation_step(sample_subset, 0)
                else:
                    predictions = pl_module(sample_subset)
            
            # 创建可视化
            visualization = self._create_prediction_visualization(
                sample_subset, predictions, trainer.current_epoch
            )
            
            if visualization:
                trainer.logger.experiment.log(visualization)
                logger.info(f"Logged {num_samples} sample predictions to WandB")
                
        except Exception as e:
            logger.warning(f"Failed to log sample predictions: {e}")
    
    def _create_prediction_visualization(
        self, 
        batch: Dict[str, Any], 
        predictions: Any, 
        epoch: int
    ) -> Optional[Dict[str, Any]]:
        """
        创建预测结果可视化
        
        Args:
            batch: 输入batch
            predictions: 模型预测结果
            epoch: 当前epoch
            
        Returns:
            可视化字典，用于wandb记录
        """
        try:
            vis_dict = {}
            
            # 记录基本信息
            vis_dict[f"predictions/epoch"] = epoch
            vis_dict[f"predictions/batch_size"] = len(batch[list(batch.keys())[0]])
            
            # 这里可以根据具体的模型输出格式添加更多可视化
            # 例如：损失值、预测精度等
            if isinstance(predictions, dict):
                for key, value in predictions.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        vis_dict[f"predictions/{key}"] = value.item()
            
            return vis_dict
            
        except Exception as e:
            logger.warning(f"Failed to create prediction visualization: {e}")
            return None


class WandbMetricsCallback(Callback):
    """
    WandB指标回调，用于记录额外的训练指标和统计信息
    
    只在主进程执行，确保分布式训练兼容性
    """
    
    def __init__(
        self,
        log_histograms: bool = True,
        histogram_freq: int = 10
    ):
        """
        初始化指标回调
        
        Args:
            log_histograms: 是否记录参数直方图
            histogram_freq: 直方图记录频率（epoch）
        """
        super().__init__()
        self.log_histograms = log_histograms
        self.histogram_freq = histogram_freq
        
        logger.info(f"WandbMetricsCallback initialized:")
        logger.info(f"  - Histogram logging: {log_histograms}")
        logger.info(f"  - Histogram frequency: {histogram_freq} epochs")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """训练epoch结束时记录参数和梯度直方图"""
        if not is_main_process():
            return
            
        if (self.log_histograms and 
            trainer.current_epoch % self.histogram_freq == 0):
            self._log_parameter_histograms(pl_module, trainer.current_epoch)
    
    def _log_parameter_histograms(self, pl_module: pl.LightningModule, epoch: int) -> None:
        """记录参数和梯度直方图"""
        try:
            if not (hasattr(pl_module.trainer, 'logger') and 
                   hasattr(pl_module.trainer.logger, 'experiment')):
                return
                
            histograms = {}
            
            # 记录参数直方图
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.data is not None:
                    # 清理参数名称，使其适合wandb
                    clean_name = name.replace('.', '/')
                    histograms[f"params/{clean_name}"] = param.data.cpu().numpy()
                    
                    # 记录梯度直方图（如果存在）
                    if param.grad is not None:
                        histograms[f"grads/{clean_name}"] = param.grad.data.cpu().numpy()
            
            # 记录到wandb
            if histograms:
                pl_module.trainer.logger.experiment.log(histograms, step=epoch)
                logger.info(f"Logged {len(histograms)} parameter/gradient histograms to WandB")
                
        except Exception as e:
            logger.warning(f"Failed to log parameter histograms: {e}")
