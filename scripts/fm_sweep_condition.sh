#!/bin/bash
set -e
set -x

python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_01_baseline \
    wandb.group=fm_sweep_condition \
    wandb.name=exp_01_baseline 

python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_02_use_adaln_zero_multi_1028 \
    wandb.enabled=False \
    model.decoder.use_adaln_zero=True \
    model.decoder.use_scene_pooling=False \
    model.decoder.adaln_mode=multi

python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_03_use_geometric_bias \
    wandb.group=fm_sweep_condition \
    wandb.name=exp_03_use_geometric_bias \
    model.decoder.use_geometric_bias=True 

python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_04_use_global_local_conditioning \
    wandb.group=fm_sweep_condition \
    wandb.name=exp_04_use_global_local_conditioning \
    model.decoder.use_global_local_conditioning=True 


python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_05_use_t_aware_conditioning_retry1028 \
    wandb.group=fm_sweep_condition \
    wandb.name=exp_05_use_t_aware_conditioning_retry1028 \
    model.decoder.use_t_aware_conditioning=True 
