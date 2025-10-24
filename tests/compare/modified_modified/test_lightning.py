import json
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from datasets.scenedex_datamodule import SceneLeapDataModule
from models.diffuser_lightning import DDPMLightning
from models.cvae import GraspCVAELightning
from utils.logging_utils import setup_basic_logging, setup_file_logging
from utils.git_utils import get_git_head_hash
from utils.backup_utils import backup_code

os.environ["HYDRA_FULL_ERROR"] = "1"

class TestResultsCallback(pl.Callback):
    def __init__(self, save_path: Path):
        super().__init__()
        self.save_path = save_path
        self.final_test_metrics = {
            "mean_q1": 0.0,
            "mean_pen": 0.0,
            "max_pen": 0.0,
            "mean_valid_q1": 0.0
        }

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Aggregate per-sample metrics into an extended summary.
        Adds: success_rate, std/min/max, and best_* metrics (per-sample best grasp).
        """
        if not getattr(pl_module, 'metric_results', None):
            logging.info("No test outputs to aggregate.")
            return

        all_q1: list = []
        all_pen: list = []
        all_valid_q1: list = []
        # Group grasps by sample (strip trailing '_grasp{j}')
        sample_groups = {}

        for batch_result in pl_module.metric_results:
            for sample_key, sample_metrics in batch_result.items():
                q1 = float(sample_metrics.get('q1', 0.0))
                pen = float(sample_metrics.get('pen', 0.0))
                vq1 = float(sample_metrics.get('valid_q1', 0.0))
                all_q1.append(q1)
                all_pen.append(pen)
                all_valid_q1.append(vq1)

                base_key = sample_key.rsplit('_grasp', 1)[0] if '_grasp' in sample_key else sample_key
                g = sample_groups.setdefault(base_key, {'q1': [], 'pen': []})
                g['q1'].append(q1)
                g['pen'].append(pen)

        # Safety
        def _mean(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0
        def _std(x):
            return float(np.std(x)) if len(x) > 0 else 0.0
        def _min(x):
            return float(np.min(x)) if len(x) > 0 else 0.0
        def _max(x):
            return float(np.max(x)) if len(x) > 0 else 0.0

        thres_pen = 0.0
        try:
            thres_pen = float(getattr(pl_module.criterion, 'q1_cfg', {}).get('thres_pen', 0.0))
        except Exception:
            pass

        # Overall metrics
        mean_q1 = _mean(all_q1)
        mean_pen = _mean(all_pen)
        max_pen = _max(all_pen)
        mean_valid_q1 = _mean(all_valid_q1)
        std_q1 = _std(all_q1)
        std_pen = _std(all_pen)
        min_q1 = _min(all_q1)
        max_q1 = _max(all_q1)
        min_pen = _min(all_pen)
        success_rate = float(np.mean(np.array(all_pen) < thres_pen)) if len(all_pen) > 0 else 0.0

        # Best-grasp metrics per sample
        best_q1_list = []
        best_pen_list = []
        best_valid_q1_list = []
        for _, vals in sample_groups.items():
            q1_arr = np.array(vals['q1'], dtype=float)
            pen_arr = np.array(vals['pen'], dtype=float)
            if q1_arr.size == 0:
                continue
            valid_mask = pen_arr < thres_pen
            if valid_mask.any():
                # among valid, pick max Q1
                valid_indices = np.where(valid_mask)[0]
                local_idx = int(np.argmax(q1_arr[valid_indices]))
                best_idx = int(valid_indices[local_idx])
            else:
                # otherwise, pick minimum penetration
                best_idx = int(np.argmin(pen_arr))
            best_q1 = float(q1_arr[best_idx])
            best_pen = float(pen_arr[best_idx])
            best_q1_list.append(best_q1)
            best_pen_list.append(best_pen)
            best_valid_q1_list.append(best_q1 if best_pen < thres_pen else 0.0)

        best_mean_q1 = _mean(best_q1_list)
        best_mean_pen = _mean(best_pen_list)
        best_max_pen = _max(best_pen_list)
        best_mean_valid_q1 = _mean(best_valid_q1_list)
        best_success_rate = float(np.mean(np.array(best_pen_list) < thres_pen)) if len(best_pen_list) > 0 else 0.0

        # Compose final metrics
        self.final_test_metrics = {
            # Overall
            "mean_q1": mean_q1,
            "mean_pen": mean_pen,
            "max_pen": max_pen,
            "mean_valid_q1": mean_valid_q1,
            # Extended overall
            "std_q1": std_q1,
            "std_pen": std_pen,
            "min_q1": min_q1,
            "max_q1": max_q1,
            "min_pen": min_pen,
            "success_rate": success_rate,
            # Best grasp per sample
            "best_mean_q1": best_mean_q1,
            "best_mean_pen": best_mean_pen,
            "best_max_pen": best_max_pen,
            "best_mean_valid_q1": best_mean_valid_q1,
            "best_success_rate": best_success_rate,
            # Threshold used
            "thres_pen": thres_pen,
        }

        detailed_results = {
            "summary_metrics": self.final_test_metrics,
            "per_sample_metrics": pl_module.metric_results,
        }

        results_file = self.save_path / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logging.info(f"Saved detailed test results to: {results_file}")
        pl_module.metric_results.clear()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: OmegaConf) -> None:
    setup_basic_logging()
    
    if not cfg.get("checkpoint_path") and not cfg.get("train_root"):
        raise ValueError("Must provide either checkpoint_path or train_root for testing")
    
    checkpoint_paths = []
    
    if cfg.get("train_root"):
        train_root = Path(cfg.train_root)
        checkpoint_dir = train_root / "checkpoints"
        checkpoint_paths = sorted(checkpoint_dir.glob("*val_loss*.ckpt"))
        
        if not checkpoint_paths:
            logging.warning(f"No checkpoints containing 'val_loss' found in {checkpoint_dir}")
            return
        
        logging.info(f"Found {len(checkpoint_paths)} checkpoints to test:")
        for path in checkpoint_paths:
            logging.info(f"  - {path.name}")
    else:
        checkpoint_paths = [Path(cfg.checkpoint_path)]
    
    for ckpt_path in checkpoint_paths:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing checkpoint: {ckpt_path.name}")
        
        original_exp_dir = ckpt_path.parent.parent
        original_config_path = original_exp_dir / "config" / "whole_config.yaml"
        
        if not original_config_path.exists():
            logging.error(f"Original config file not found: {original_config_path}")
            continue
        
        with open(original_config_path, 'r') as f:
            original_cfg = OmegaConf.load(f)
        
        test_cfg = OmegaConf.merge(original_cfg, cfg)
        test_cfg.checkpoint_path = str(ckpt_path)
        
        test_save_dir = original_exp_dir / "test_results" / ckpt_path.stem
        test_save_dir.mkdir(parents=True, exist_ok=True)
        
        setup_file_logging(test_save_dir, mode='test')
        
        git_hash = get_git_head_hash()
        if git_hash:
            logging.info(f"Current git commit: {git_hash}")
        
        config_dir = test_save_dir / "config"
        config_dir.mkdir(exist_ok=True)
        OmegaConf.save(test_cfg, config_dir / "test_config.yaml")
        
        pl.seed_everything(test_cfg.seed, workers=True)
        
        try:
            if test_cfg.model.name == "GraspCVAE":
                model = GraspCVAELightning(test_cfg.model)
            elif test_cfg.model.name == "GraspDiffuser":
                model = DDPMLightning(test_cfg.model)
            else:
                raise ValueError(f"Unknown model name: {test_cfg.model.name}")
            datamodule = SceneLeapDataModule(test_cfg.data)
            
            callbacks = [
                TestResultsCallback(test_save_dir),
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
            
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=callbacks,
                enable_progress_bar=True,
                benchmark=True,
            )
            
            trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))
            logging.info(f"Test results saved to: {test_save_dir}")
            
        except Exception as e:
            logging.error(f"Testing {ckpt_path.name} failed: {str(e)}", exc_info=True)
            continue
    
    logging.info("All tests completed!")

if __name__ == "__main__":
    main()