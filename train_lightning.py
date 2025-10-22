import logging
import os
from pathlib import Path
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelSummary,
    RichProgressBar
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from datasets.scenedex_datamodule import SceneLeapDataModule
from utils.wandb_callbacks import WandbVisualizationCallback, WandbMetricsCallback
# from models.diffuser_lightning_copy import DDPMLightning
from models.diffuser_lightning import DDPMLightning
from models.flow_matching_lightning import FlowMatchingLightning
from models.cvae import GraspCVAELightning
from utils.logging_utils import setup_basic_logging, setup_file_logging
from utils.git_utils import get_git_head_hash
from utils.backup_utils import backup_code
from utils.distributed_utils import (
    setup_distributed_environment,
    setup_environment_variables,
    adjust_batch_size_for_distributed,
    adjust_learning_rate_for_distributed,
    get_trainer_kwargs,
    log_distributed_info,
    is_main_process
)

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg) -> None:
    setup_basic_logging()

    # 设置分布式训练环境
    try:
        setup_environment_variables(cfg)
        dist_config = setup_distributed_environment(cfg)
        log_distributed_info(dist_config)

        # 根据分布式配置调整批次大小和学习率
        cfg = adjust_batch_size_for_distributed(cfg, dist_config)
        cfg = adjust_learning_rate_for_distributed(cfg, dist_config)

    except Exception as e:
        logging.error(f"分布式环境设置失败: {e}")
        fallback = cfg.get('distributed', {}).get('fallback_to_single_gpu', True)
        if fallback:
            logging.warning("回退到单GPU训练")
            dist_config = {
                'enabled': False,
                'strategy': 'auto',
                'devices': 1,
                'num_nodes': 1,
                'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu'
            }
        else:
            raise e

    # 保存命令行参数的副本
    cli_overrides = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    # 处理 resume 训练的情况
    if cfg.get("resume", False):
        if not cfg.get("checkpoint_path"):
            raise ValueError("checkpoint_path must be provided when resume=True")
            
        checkpoint_path = Path(cfg.checkpoint_path)
        original_exp_dir = checkpoint_path.parent.parent
        original_config_path = original_exp_dir / "config" / "whole_config.yaml"
        
        if original_config_path.exists():
            logging.info(f"Loading config from: {original_config_path}")
            with open(original_config_path, 'r') as f:
                whole_cfg = OmegaConf.load(f)
                
            # 配置合并顺序：命令行 > whole_config > 原始config
            cfg = OmegaConf.merge(cfg, whole_cfg)  # 先加载原始config和whole_config
            cfg = OmegaConf.merge(cfg, cli_overrides)  # 最后用命令行参数覆盖
            
            # 强制更新关键参数
            cfg.save_root = str(original_exp_dir)
            cfg.checkpoint_path = str(checkpoint_path)
            
            checkpoint_epoch = int(Path(cfg.checkpoint_path).stem.split('=')[1].split('-')[0])
            logging.info(f"Resume training from epoch {checkpoint_epoch}")
    
    # 创建输出目录（所有进程都需要）
    save_dir = Path(cfg.save_root)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置文件logging（所有进程都需要，但文件名不同）
    setup_file_logging(save_dir, mode='train', is_resume=cfg.get("resume", False))
    logger = logging.getLogger(__name__)

    # Lightning logs目录（所有进程都需要）
    lightning_log_dir = save_dir / 'lightning_logs'
    lightning_log_dir.mkdir(exist_ok=True)

    # 以下操作只在主进程执行
    if is_main_process():
        # 记录git版本
        git_hash = get_git_head_hash()
        if git_hash:
            logging.info(f"Current git commit: {git_hash}")
        else:
            logger.warning("Unable to get git commit hash")

        # 备份代码
        logging.info("Backing up code...")
        backup_code(save_dir)

        # 保存配置
        config_dir = save_dir / "config"
        config_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, config_dir / "whole_config.yaml")
        logging.info(f"Saved config to: {config_dir / 'whole_config.yaml'}")

    # 在分布式训练中，需要同步所有进程，确保主进程完成文件操作后其他进程再继续
    if dist_config['enabled'] and torch.distributed.is_available():
        # 等待主进程完成文件操作
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    # 设置随机种子
    pl.seed_everything(cfg.seed, workers=True)
    logging.info(f"Set random seed to: {cfg.seed}")

    # 初始化wandb logger
    wandb_logger = None
    if cfg.get("wandb", {}).get("enabled", False):
        # 生成实验名称
        exp_name = cfg.get("wandb", {}).get("name", None)
        if exp_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = cfg.model.name
            exp_name = f"{model_name}_{timestamp}"

        # 准备wandb配置
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        # 添加实验管理信息
        experiment_info = {
            "experiment_name": exp_name,
            "model_type": cfg.model.name,
            "dataset": cfg.data_cfg.name if hasattr(cfg.data_cfg, 'name') else "unknown",
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.model.optimizer.lr if hasattr(cfg.model, 'optimizer') else "unknown",
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "git_commit": get_git_head_hash() or "unknown",
            "save_root": str(save_dir),
            "distributed": dist_config.get('enabled', False),
            "num_gpus": dist_config.get('devices', 1) if isinstance(dist_config.get('devices'), int) else len(dist_config.get('devices', [1])),
        }

        # 合并配置
        wandb_config.update(experiment_info)

        # 准备标签
        tags = cfg.get("wandb", {}).get("tags", [])
        tags.extend([
            cfg.model.name,
            f"batch_{cfg.batch_size}",
            f"lr_{cfg.model.optimizer.lr}" if hasattr(cfg.model, 'optimizer') else "lr_unknown",
            "distributed" if dist_config.get('enabled', False) else "single_gpu"
        ])

        # 根据PyTorch Lightning最佳实践，WandbLogger会自动处理分布式训练
        # 只有rank 0进程会实际记录到wandb，其他进程的logger会是None
        wandb_logger = WandbLogger(
            project=cfg.get("wandb", {}).get("project", "scene-leap-plus-diffusion-grasp"),
            name=exp_name,
            group=cfg.get("wandb", {}).get("group", None),
            tags=list(set(tags)),  # 去重
            notes=cfg.get("wandb", {}).get("notes", ""),
            config=wandb_config,
            save_dir=str(lightning_log_dir),
            log_model=cfg.get("wandb", {}).get("save_model", False),
            settings=wandb.Settings(init_timeout=300)
        )

        logging.info(f"Initialized WandB logger for project: {cfg.get('wandb', {}).get('project', 'scene-leap-plus-diffusion-grasp')}")
        logging.info(f"Experiment name: {exp_name}")
    else:
        logging.info("WandB logging disabled in configuration")
    
    # 初始化模型和数据模块
    logging.info("Initializing model and data module...")

    # 准备模型配置，包含wandb优化配置
    model_cfg = cfg.model
    if cfg.get("wandb", {}).get("enabled", False):
        # 创建一个包含wandb配置的新配置对象
        wandb_opt = cfg.get("wandb", {}).get("optimization", {})
        wandb_opt.monitor_system = cfg.get("wandb", {}).get("monitor_system", False)

        # 使用OmegaConf.merge来安全地添加wandb配置
        additional_cfg = OmegaConf.create({"wandb_optimization": wandb_opt})
        model_cfg = OmegaConf.merge(model_cfg, additional_cfg)

    if cfg.model.name == "GraspCVAE":
        model = GraspCVAELightning(model_cfg)
    elif cfg.model.name == "GraspDiffuser":
        model = DDPMLightning(model_cfg)
    elif cfg.model.name == "GraspFlowMatching":
        model = FlowMatchingLightning(model_cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}. Supported: GraspCVAE, GraspDiffuser, GraspFlowMatching")
    # logging.info(f"Initialized model: {model}")
    datamodule = SceneLeapDataModule(cfg.data_cfg)
    
    # 设置callbacks
    logging.info("Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            dirpath=str(save_dir / "checkpoints"),
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=cfg.save_top_n,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(save_dir / "checkpoints"),
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
        DeviceStatsMonitor(),
        ModelSummary(max_depth=3),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="blue",
                progress_bar_finished="green",
                progress_bar_pulse="green1",
                batch_progress="pink",
                time="indigo",
                processing_speed="purple",
                metrics="yellow",
            )
        ),
    ]

    # 添加WandB可视化回调（如果启用wandb且启用了可视化）
    if cfg.get("wandb", {}).get("enabled", False):
        wandb_opt = cfg.get("wandb", {}).get("optimization", {})

        # 可视化回调（根据配置决定是否启用）
        if wandb_opt.get("enable_visualization", False):
            viz_callback = WandbVisualizationCallback(
                log_model_graph=True,
                log_sample_predictions=True,
                sample_log_freq=wandb_opt.get("visualization_freq", 20),
                max_samples_to_log=4
            )
            callbacks.append(viz_callback)
            logging.info("Added WandB visualization callback")

        # 指标回调（根据配置决定是否启用直方图）
        if wandb_opt.get("log_histograms", False):
            metrics_callback = WandbMetricsCallback(
                log_histograms=True,
                histogram_freq=wandb_opt.get("histogram_freq", 50)
            )
            callbacks.append(metrics_callback)
            logging.info("Added WandB metrics callback with histograms")

        logging.info("WandB callbacks configured for minimal bandwidth usage")
    
    # # 在trainer初始化之前添加profiler配置
    # profiler = PyTorchProfiler(
    #     dirpath=str(save_dir / "profiler_logs"),  # 保存到实验目录下的profiler_logs文件夹
    #     filename="profiler_output",
    #     group_by_input_shapes=True,
    #     export_to_chrome=True,
    #     row_limit=100,
    #     # sort_by_key="cuda_time_total",
    #     record_module_names=True,
    # )
    
    # 获取分布式训练配置
    trainer_kwargs = get_trainer_kwargs(cfg, dist_config)

    # 从配置中获取trainer参数
    trainer_cfg = cfg.get('trainer', {})

    # 初始化trainer
    logging.info("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get('max_epochs', cfg.epochs),
        check_val_every_n_epoch=trainer_cfg.get('check_val_every_n_epoch', 1),
        logger=wandb_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=trainer_cfg.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_cfg.get('accumulate_grad_batches',
                                               cfg.get('distributed', {}).get('accumulate_grad_batches', 1)),
        benchmark=trainer_cfg.get('benchmark', True),
        log_every_n_steps=trainer_cfg.get('log_every_n_steps', 20),
        sync_batchnorm=trainer_cfg.get('sync_batchnorm', False),
        # profiler=profiler,
        **trainer_kwargs
    )
    
    # 开始训练
    try:
        logging.info("Starting training...")
        if cfg.get("resume", False):
            logging.info(f"Resuming from checkpoint: {cfg.checkpoint_path}")
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint_path)
        else:
            trainer.fit(model, datamodule=datamodule)
        logging.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise e
    finally:
        logging.info("Cleaning up WandB...")
        wandb.finish()

if __name__ == "__main__":
    logging.info("Starting training...")
    main()