from typing import Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from . import build_datasets


class SceneLeapDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.data_cfg = data_cfg
        self.save_hyperparameters()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            logging.info("Building train and validation datasets...")
            self.train_dataset, self.val_dataset, _ = build_datasets(self.data_cfg, stage="fit")
            logging.info(f"Train dataset size: {len(self.train_dataset)}")
            logging.info(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            logging.info("Building test dataset...")
            _, _, self.test_dataset = build_datasets(self.data_cfg, stage="test")
            if self.test_dataset is None:
                raise ValueError("Test dataset is None!")
            logging.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        use_distributed = hasattr(self.trainer, 'world_size') and self.trainer.world_size > 1
        num_workers = self.data_cfg["train"]["num_workers"]
        if use_distributed:
            # Adjust workers for DDP: each process gets a subset of workers.
            num_workers = max(1, num_workers // self.trainer.world_size)

        dataloader_kwargs = {
            'batch_size': self.data_cfg["train"]["batch_size"],
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': True,
            'collate_fn': self.train_dataset.collate_fn,  # Collate_fn to handle negative prompts.
            'persistent_workers': True if num_workers > 0 else False,
            'prefetch_factor': 4 if num_workers > 0 else None
        }

        if use_distributed:
            # Use DistributedSampler for DDP.
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                drop_last=True
            )
            dataloader_kwargs['sampler'] = sampler
        else:
            # Shuffle for single-process training.
            dataloader_kwargs['shuffle'] = True

        return DataLoader(self.train_dataset, **dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        use_distributed = hasattr(self.trainer, 'world_size') and self.trainer.world_size > 1
        num_workers = self.data_cfg["val"]["num_workers"]
        if use_distributed:
            num_workers = max(1, num_workers // self.trainer.world_size)

        dataloader_kwargs = {
            'batch_size': self.data_cfg["val"]["batch_size"],
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': True,
            'collate_fn': self.val_dataset.collate_fn,
            'persistent_workers': True if num_workers > 0 else False,
            'prefetch_factor': 4 if num_workers > 0 else None
        }

        if use_distributed:
            # Use DistributedSampler for validation to ensure each process sees different data.
            sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
                drop_last=False
            )
            dataloader_kwargs['sampler'] = sampler
            del dataloader_kwargs['shuffle']  # Sampler is mutually exclusive with shuffle.

        return DataLoader(self.val_dataset, **dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        use_distributed = hasattr(self.trainer, 'world_size') and self.trainer.world_size > 1
        num_workers = self.data_cfg["test"]["num_workers"]
        if use_distributed:
            num_workers = max(1, num_workers // self.trainer.world_size)

        dataloader_kwargs = {
            'batch_size': self.data_cfg["test"]["batch_size"],
            'shuffle': False,
            'num_workers': num_workers,
            'pin_memory': True,
            'collate_fn': self.test_dataset.collate_fn,
            'persistent_workers': True if num_workers > 0 else False,
            'prefetch_factor': 4 if num_workers > 0 else None
        }

        if use_distributed:
            # Use DistributedSampler for the test set as well.
            sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
                drop_last=False
            )
            dataloader_kwargs['sampler'] = sampler
            del dataloader_kwargs['shuffle']

        return DataLoader(self.test_dataset, **dataloader_kwargs)
