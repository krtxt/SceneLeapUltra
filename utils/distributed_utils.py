"""
This module provides utility functions for setting up and managing
distributed training environments using PyTorch and PyTorch Lightning.
"""

import logging
import os
import socket
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig


def is_distributed_available() -> bool:
    """Checks if the distributed package is available."""
    return dist.is_available()


def get_available_gpus() -> int:
    """Returns the number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def find_free_port() -> int:
    """Finds and returns an available port on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_environment(cfg: DictConfig) -> Dict[str, Any]:
    """
    Configures the distributed training environment based on the provided configuration.

    Args:
        cfg: The OmegaConf configuration object.

    Returns:
        A dictionary containing the distributed training configuration.
    """
    dist_config = {
        "enabled": False,
        "strategy": "auto",
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
    }

    distributed_cfg = cfg.get("distributed", {})
    enabled_flag = distributed_cfg.get("enabled", "auto")
    available_gpus = get_available_gpus()
    logging.info(f"Found {available_gpus} available GPUs.")

    # Determine if distributed training should be enabled
    if enabled_flag == "auto":
        should_enable = available_gpus > 1 and is_distributed_available()
        dist_config["enabled"] = should_enable
        if should_enable:
            logging.info("Auto-enabling distributed training.")
        else:
            logging.info(
                f"Auto-disabling distributed training (GPUs: {available_gpus}, Distributed available: {is_distributed_available()})."
            )
    elif enabled_flag is True:
        if not is_distributed_available():
            raise RuntimeError(
                "Distributed training is not available but was explicitly enabled in the config."
            )
        if available_gpus < 2:
            logging.warning(
                f"Forcing distributed training with only {available_gpus} GPU(s) available."
            )
        dist_config["enabled"] = True
    else:
        dist_config["enabled"] = False
        logging.info("Distributed training is disabled by configuration.")

    if not dist_config["enabled"]:
        return dist_config

    # Configure distributed parameters
    strategy = distributed_cfg.get("strategy", "ddp")
    devices = distributed_cfg.get("devices", "auto")

    # Configure devices
    if devices == "auto":
        dist_config["devices"] = available_gpus
    elif isinstance(devices, int):
        if devices > available_gpus:
            logging.warning(
                f"Requested {devices} GPUs, but only {available_gpus} are available. Using all available GPUs."
            )
            dist_config["devices"] = available_gpus
        else:
            dist_config["devices"] = devices
    elif isinstance(devices, (list, tuple)):
        valid_devices = [d for d in devices if d < available_gpus]
        if len(valid_devices) != len(devices):
            logging.warning(
                f"Some specified GPUs are not available. Using valid devices: {valid_devices}"
            )
        dist_config["devices"] = valid_devices
    else:
        raise ValueError(f"Invalid 'devices' configuration: {devices}")

    # Configure strategy
    supported_strategies = ["ddp", "fsdp", "ddp_sharded"]
    if strategy in supported_strategies:
        dist_config["strategy"] = strategy
    else:
        logging.warning(
            f"Unknown distributed strategy '{strategy}'. Defaulting to 'ddp'."
        )
        dist_config["strategy"] = "ddp"

    # Configure other parameters
    dist_config["num_nodes"] = distributed_cfg.get("num_nodes", 1)
    dist_config["precision"] = distributed_cfg.get("precision", 32)
    dist_config["sync_batchnorm"] = distributed_cfg.get("sync_batchnorm", False)
    dist_config["find_unused_parameters"] = distributed_cfg.get(
        "find_unused_parameters", False
    )

    return dist_config


def setup_environment_variables(cfg: DictConfig) -> None:
    """Sets up environment variables required for distributed training."""
    distributed_cfg = cfg.get("distributed", {})

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        logging.info("Setting MASTER_ADDR=localhost")

    if "MASTER_PORT" not in os.environ:
        port = find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        logging.info(f"Setting MASTER_PORT={port}")

    backend = distributed_cfg.get("backend", "nccl")
    if torch.cuda.is_available() and backend == "nccl":
        os.environ["NCCL_DEBUG"] = "INFO"

    timeout = distributed_cfg.get("timeout", 1800)
    os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = str(timeout)


def adjust_batch_size_for_distributed(
    cfg: DictConfig, dist_config: Dict[str, Any]
) -> DictConfig:
    """
    Adjusts the batch size for each device to maintain the effective global batch size.

    Args:
        cfg: The original configuration.
        dist_config: The distributed training configuration.

    Returns:
        The configuration with the adjusted batch size.
    """
    if not dist_config.get("enabled", False):
        return cfg

    num_devices = dist_config.get("devices", 1)
    if isinstance(num_devices, (list, tuple)):
        num_devices = len(num_devices)

    if num_devices <= 1:
        return cfg

    original_batch_size = cfg.batch_size
    per_device_batch_size = original_batch_size // num_devices

    if per_device_batch_size < 1:
        logging.warning(
            f"Original batch size {original_batch_size} is too small for {num_devices} devices. "
            f"Setting per-device batch size to 1."
        )
        per_device_batch_size = 1

    effective_batch_size = per_device_batch_size * num_devices
    if effective_batch_size != original_batch_size:
        logging.info(
            f"Adjusting batch size from {original_batch_size} to {effective_batch_size} "
            f"({per_device_batch_size} per device)."
        )

    # Update batch size in the main config and data-specific configs
    cfg.batch_size = per_device_batch_size
    if hasattr(cfg, "data") and cfg.data is not None:
        for split in ["train", "val", "test"]:
            if hasattr(cfg.data, split) and getattr(cfg.data, split) is not None:
                getattr(cfg.data, split).batch_size = per_device_batch_size

    return cfg


def adjust_learning_rate_for_distributed(
    cfg: DictConfig, dist_config: Dict[str, Any]
) -> DictConfig:
    """
    Adjusts the learning rate based on the number of devices (learning rate scaling).

    Args:
        cfg: The original configuration.
        dist_config: The distributed training configuration.

    Returns:
        The configuration with the adjusted learning rate.
    """
    if not dist_config.get("enabled", False):
        return cfg

    num_devices = dist_config.get("devices", 1)
    if isinstance(num_devices, (list, tuple)):
        num_devices = len(num_devices)

    if num_devices <= 1:
        return cfg

    scaling_method = cfg.get("distributed", {}).get("lr_scaling", "sqrt")
    if scaling_method not in ["linear", "sqrt"]:
        return cfg

    try:
        original_lr = cfg.model.optimizer.lr
    except AttributeError:
        logging.warning(
            "Could not find 'model.optimizer.lr' in config. Skipping learning rate scaling."
        )
        return cfg

    if scaling_method == "linear":
        adjusted_lr = original_lr * num_devices
    else:  # 'sqrt'
        adjusted_lr = original_lr * (num_devices**0.5)

    if adjusted_lr != original_lr:
        logging.info(
            f"Scaling learning rate from {original_lr:.2e} to {adjusted_lr:.2e} "
            f"(method: {scaling_method}, devices: {num_devices})."
        )
        cfg.model.optimizer.lr = adjusted_lr

    return cfg


def get_trainer_kwargs(cfg: DictConfig, dist_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs keyword arguments for the PyTorch Lightning Trainer.

    Args:
        cfg: The application configuration.
        dist_config: The distributed training configuration.

    Returns:
        A dictionary of keyword arguments for the Trainer.
    """
    trainer_cfg = cfg.get("trainer", {})

    accelerator = trainer_cfg.get("accelerator", dist_config["accelerator"])
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    devices = trainer_cfg.get("devices", dist_config["devices"])
    if not dist_config["enabled"] and devices != 1:
        devices = 1

    trainer_kwargs = {
        "accelerator": accelerator,
        "devices": devices,
        "num_nodes": dist_config.get("num_nodes", 1),
        "precision": trainer_cfg.get("precision", dist_config.get("precision", 32)),
    }

    if dist_config["enabled"]:
        strategy_name = trainer_cfg.get("strategy", dist_config["strategy"])
        if strategy_name == "auto":
            strategy_name = "ddp"

        if strategy_name == "ddp":
            from pytorch_lightning.strategies import DDPStrategy

            ddp_kwargs = {
                "find_unused_parameters": trainer_cfg.get(
                    "find_unused_parameters",
                    dist_config.get("find_unused_parameters", False),
                ),
                "gradient_as_bucket_view": True,
            }
            trainer_kwargs["strategy"] = DDPStrategy(**ddp_kwargs)
        elif strategy_name in ["fsdp", "ddp_sharded"]:
            trainer_kwargs["strategy"] = strategy_name
        else:
            logging.warning(
                f"Unsupported strategy '{strategy_name}'. Defaulting to 'ddp'."
            )
            trainer_kwargs["strategy"] = "ddp"

    return trainer_kwargs


def is_main_process() -> bool:
    """
    Checks if the current process is the main process (rank 0).
    This is safe to call in non-distributed environments.
    """
    if "RANK" in os.environ and int(os.environ["RANK"]) == 0:
        return True
    if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) == 0:
        return True
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # If no distributed environment variables are set, assume it's the main process.
    return "RANK" not in os.environ and "LOCAL_RANK" not in os.environ


def log_distributed_info(dist_config: Dict[str, Any]) -> None:
    """Logs a summary of the distributed training configuration on the main process."""
    if not is_main_process():
        return

    if dist_config.get("enabled", False):
        devices = dist_config["devices"]
        if isinstance(devices, (list, tuple)):
            device_info = f"GPUs {devices}"
        else:
            device_info = f"{devices} GPUs"

        logging.info("=" * 50)
        logging.info("Distributed Training Configuration")
        logging.info(f"  - Strategy:         {dist_config['strategy']}")
        logging.info(f"  - Devices:          {device_info}")
        logging.info(f"  - Nodes:            {dist_config['num_nodes']}")
        logging.info(f"  - Precision:        {dist_config['precision']}")
        logging.info(
            f"  - Sync Batch Norm:  {dist_config.get('sync_batchnorm', False)}"
        )
        logging.info("=" * 50)
    else:
        logging.info("Training on a single device (distributed mode disabled).")
