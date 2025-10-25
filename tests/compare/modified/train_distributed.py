#!/usr/bin/env python3
"""
A launcher script for distributed training.
Supports multiple launch methods, including torchrun, SLURM, and manual execution.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from utils.logging_utils import setup_basic_logging


def setup_logging():
    """Initializes basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def detect_environment() -> dict:
    """
    Detects the current execution environment (SLURM, torchrun, or manual).
    
    Returns:
        A dictionary indicating the detected environment.
    """
    env_info = {
        'slurm': 'SLURM_JOB_ID' in os.environ,
        'torchrun': 'LOCAL_RANK' in os.environ,
        'manual': False
    }
    
    if not any(env_info.values()):
        env_info['manual'] = True
    
    return env_info


def get_gpu_count() -> int:
    """
    Gets the number of available GPUs.
    
    Returns:
        The number of CUDA devices, or 0 if PyTorch is not installed.
    """
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        logging.warning("PyTorch is not installed. Cannot detect GPU count.")
        return 0


def build_torchrun_command(args: argparse.Namespace) -> list:
    """
    Constructs the command for launching a job with torchrun.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        A list of strings representing the command, or None on failure.
    """
    gpu_count = get_gpu_count()
    
    if gpu_count == 0:
        logging.error("No available GPUs detected for torchrun.")
        return None
    
    nproc_per_node = args.gpus or gpu_count
    
    cmd = [
        'torchrun',
        f'--nproc_per_node={nproc_per_node}',
        f'--nnodes={args.nodes}',
    ]
    
    if args.nodes > 1:
        if not args.master_addr:
            logging.error("--master_addr is required for multi-node training.")
            return None
        cmd.extend([
            f'--rdzv_endpoint={args.master_addr}:{args.master_port}',
            f'--rdzv_backend=c10d',
            f'--rdzv_id={args.job_id}',
        ])
    
    cmd.append('train_lightning.py')
    
    if args.config_overrides:
        cmd.extend(args.config_overrides)
    
    # Force-enable distributed training settings via Hydra override
    cmd.append('distributed.enabled=true')
    cmd.append(f'distributed.devices={nproc_per_node}')
    cmd.append(f'distributed.num_nodes={args.nodes}')
    
    return cmd


def build_slurm_command(args: argparse.Namespace) -> list:
    """
    Constructs the command for launching a job with SLURM's srun.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        A list of strings representing the command.
    """
    gpus_per_node = args.gpus or get_gpu_count()
    cmd = [
        'srun',
        f'--ntasks-per-node={gpus_per_node}',
        f'--nodes={args.nodes}',
        f'--gres=gpu:{gpus_per_node}',
        'python', 'train_lightning.py'
    ]
    
    if args.config_overrides:
        cmd.extend(args.config_overrides)
        
    cmd.append('distributed.enabled=true')
    cmd.append(f'distributed.devices={gpus_per_node}')
    cmd.append(f'distributed.num_nodes={args.nodes}')
    
    return cmd


def run_manual_distributed(args: argparse.Namespace) -> subprocess.CompletedProcess:
    """
    Manually launches a distributed training job.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        The result of the subprocess run.
    """
    gpu_count = get_gpu_count()
    
    if gpu_count < 2:
        logging.warning(f"Only {gpu_count} GPU(s) available. Starting single-GPU training.")
        cmd = ['python', 'train_lightning.py']
        if args.config_overrides:
            cmd.extend(args.config_overrides)
        return subprocess.run(cmd, check=False)
    
    os.environ['MASTER_ADDR'] = args.master_addr or 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    cmd = build_torchrun_command(args)
    if cmd is None:
        sys.exit(1)
    
    logging.info(f"Executing command: {' '.join(cmd)}")
    return subprocess.run(cmd, check=False)


def main():
    """Main entry point for the script."""
    # setup_logging()
    setup_basic_logging()
    
    parser = argparse.ArgumentParser(description='Distributed Training Launcher')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use per node (default: auto-detect)')
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--master_addr', type=str, help='Address of the master node (required for multi-node)')
    parser.add_argument('--master_port', type=int, default=29501, help='Port of the master node (default: 29501)')
    parser.add_argument('--job_id', type=str, default='default_job', help='A unique job ID (default: default_job)')
    parser.add_argument(
        '--launcher', 
        choices=['auto', 'torchrun', 'slurm', 'manual'], 
        default='auto', 
        help='Launcher to use (default: auto-detect)'
    )
    parser.add_argument(
        'config_overrides', 
        nargs='*', 
        help='Hydra config overrides (e.g., data.batch_size=64 trainer.max_epochs=100)'
    )
    
    args = parser.parse_args()
    
    env_info = detect_environment()
    logging.info(f"Detected environment: {env_info}")
    
    launcher = args.launcher
    if launcher == 'auto':
        if env_info['slurm']:
            launcher = 'slurm'
        elif env_info['torchrun']:
            # Already in a torchrun environment, so we just execute the script
            launcher = 'passthrough' 
        else:
            launcher = 'manual'
    
    logging.info(f"Using launcher: {launcher}")
    
    result = None
    if launcher == 'passthrough':
        # Already in a distributed environment, just run the training script
        cmd = ['python', 'train_lightning.py'] + args.config_overrides
        logging.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
    elif launcher == 'torchrun':
        cmd = build_torchrun_command(args)
        if cmd:
            logging.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)
    elif launcher == 'slurm':
        cmd = build_slurm_command(args)
        logging.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
    elif launcher == 'manual':
        result = run_manual_distributed(args)
    else:
        logging.error(f"Unknown launcher: {launcher}")
        return 1
        
    return result.returncode if result else 1


if __name__ == '__main__':
    sys.exit(main())
