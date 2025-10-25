"""
Distributed Training Utilities for SceneLeapPro Dataset

This module provides utility functions for distributed training support,
including process detection, synchronization, and coordination.
"""

import logging
import os
import time
from typing import Optional

import torch


def is_distributed_training() -> bool:
    """
    Check if running in distributed training environment.

    Returns:
        bool: True if in distributed training environment
    """
    return (
        "LOCAL_RANK" in os.environ
        or "RANK" in os.environ
        or (torch.distributed.is_available() and torch.distributed.is_initialized())
    )


def is_main_process() -> bool:
    """
    Check if current process is main process (rank 0).

    Returns:
        bool: True if main process
    """
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) == 0
    elif "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        # Non-distributed environment, default to main process
        return True


def distributed_barrier():
    """
    Synchronization barrier in distributed training.
    Only applies barrier if distributed training is active.
    """
    if is_distributed_training() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank_info() -> str:
    """
    Get rank information for logging purposes.

    Returns:
        str: Rank information string
    """
    if is_distributed_training():
        return f"rank {os.environ.get('LOCAL_RANK', 'unknown')}"
    else:
        return "single process"


def get_world_size() -> int:
    """
    Get world size in distributed training.

    Returns:
        int: World size (number of processes)
    """
    if is_distributed_training():
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
    return 1


def wait_for_directory_creation(
    directory_path: str, timeout: int = 30, check_interval: float = 0.1
) -> bool:
    """
    Wait for directory creation by main process in distributed training.

    Args:
        directory_path: Path to directory to wait for
        timeout: Timeout in seconds
        check_interval: Check interval in seconds

    Returns:
        bool: True if directory was created within timeout
    """
    if is_main_process():
        # Main process doesn't need to wait
        return True

    start_time = time.time()
    while not os.path.exists(directory_path) and (time.time() - start_time) < timeout:
        time.sleep(check_interval)

    return os.path.exists(directory_path)


def ensure_directory_exists(directory_path: str, timeout: int = 30) -> bool:
    """
    Ensure directory exists in distributed training environment.
    Main process creates directory, others wait for it.

    Args:
        directory_path: Path to directory
        timeout: Timeout for waiting in seconds

    Returns:
        bool: True if directory exists or was created successfully
    """
    # In distributed training, only main process creates directory
    if not is_distributed_training() or is_main_process():
        os.makedirs(directory_path, exist_ok=True)

    # In distributed training, wait for main process to create directory
    if is_distributed_training():
        distributed_barrier()
        if not is_main_process():
            if not wait_for_directory_creation(directory_path, timeout):
                logging.error(
                    f"Directory {directory_path} was not created by main process within timeout"
                )
                return False

    return os.path.exists(directory_path)


def log_distributed_info(message: str, level: int = logging.INFO):
    """
    Log message with distributed training context.

    Args:
        message: Message to log
        level: Logging level
    """
    rank_info = get_rank_info()
    logging.log(level, f"[{rank_info}] {message}")


def should_create_cache() -> bool:
    """
    Determine if current process should create cache files.
    In distributed training, only main process creates cache.

    Returns:
        bool: True if current process should create cache
    """
    return not is_distributed_training() or is_main_process()


def get_distributed_info() -> dict:
    """
    Get comprehensive distributed training information.

    Returns:
        dict: Dictionary containing distributed training info
    """
    info = {
        "distributed": is_distributed_training(),
        "is_main_process": is_main_process(),
        "world_size": get_world_size(),
        "rank_info": get_rank_info(),
    }

    if is_distributed_training():
        info.update(
            {
                "local_rank": os.environ.get("LOCAL_RANK", "unknown"),
                "rank": os.environ.get("RANK", "unknown"),
                "world_size_env": os.environ.get("WORLD_SIZE", "unknown"),
            }
        )

    return info
