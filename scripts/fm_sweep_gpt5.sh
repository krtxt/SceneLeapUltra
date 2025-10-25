#!/bin/bash
set -e
set -x

# =========================================================
# Flow Matching (FM) 超参扫面（单卡串行，约 49 个实验）
# 约定：
# - 所有实验均保持 epochs=500，因此统一修正 scheduler.t_max=500
# - 结果目录统一为 ./experiments/fm_sweep_gpt5/exp_xx_*
# - Weights & Biases 分组统一为 wandb.group=fm_sweep_gpt5
# - 除显式改动外，其它配置保持你的默认设置不变
# =========================================================

# --- 实验组 0: 基线 ---
# 修正了 scheduler.t_max=500 以匹配 config.yaml 中的 epochs=500
# echo "--- 1. Running Baseline (t_max=500) ---"
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_01_baseline \
#     wandb.group=fm_sweep_gpt5 \
#     wandb.name=exp_01_baseline \
#     model.scheduler.t_max=500

# # --- 实验组 1: 优化器学习率（基于 baseline）---
# echo "--- Group 1: Optimizer LR Grid ---"

# # Exp 2: LR=1e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_02_lr_1e-4 \
#     wandb.group=fm_sweep_gpt5 \
#     wandb.name=exp_02_lr_1e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0001

# # Exp 3: LR=3e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_03_lr_3e-4 \
#     wandb.group=fm_sweep_gpt5 \
#     wandb.name=exp_03_lr_3e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0003

# # Exp 4: LR=5e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_04_lr_5e-4 \
#     wandb.group=fm_sweep_gpt5 \
#     wandb.name=exp_04_lr_5e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0005

# # Exp 5: LR=1e-3
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_05_lr_1e-3 \
#     wandb.group=fm_sweep_gpt5 \
#     wandb.name=exp_05_lr_1e-3 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.001

# # --- 实验组 2: 路径(path) × 时间采样(t_sampler)（固定 solver=rk4@32）---
# echo "--- Group 2: FM Path × Time Sampler ---"

# # 线性 OT
# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_06_path-linear_ot_ts-uniform \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_06_path-linear_ot_ts-uniform \
#     model.scheduler.t_max=500 \
#     model.fm.path=linear_ot model.fm.t_sampler=uniform \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_07_path-linear_ot_ts-cosine \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_07_path-linear_ot_ts-cosine \
#     model.scheduler.t_max=500 \
#     model.fm.path=linear_ot model.fm.t_sampler=cosine \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_08_path-linear_ot_ts-beta \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_08_path-linear_ot_ts-beta \
#     model.scheduler.t_max=500 \
#     model.fm.path=linear_ot model.fm.t_sampler=beta \
#     model.solver.type=rk4 model.solver.nfe=32

# # VP
# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_09_path-vp_ts-uniform \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_09_path-vp_ts-uniform \
#     model.scheduler.t_max=500 \
#     model.fm.path=vp model.fm.t_sampler=uniform \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_10_path-vp_ts-cosine \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_10_path-vp_ts-cosine \
#     model.scheduler.t_max=500 \
#     model.fm.path=vp model.fm.t_sampler=cosine \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_11_path-vp_ts-beta \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_11_path-vp_ts-beta \
#     model.scheduler.t_max=500 \
#     model.fm.path=vp model.fm.t_sampler=beta \
#     model.solver.type=rk4 model.solver.nfe=32

# # VE
# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_12_path-ve_ts-uniform \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_12_path-ve_ts-uniform \
#     model.scheduler.t_max=500 \
#     model.fm.path=ve model.fm.t_sampler=uniform \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_13_path-ve_ts-cosine \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_13_path-ve_ts-cosine \
#     model.scheduler.t_max=500 \
#     model.fm.path=ve model.fm.t_sampler=cosine \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_14_path-ve_ts-beta \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_14_path-ve_ts-beta \
#     model.scheduler.t_max=500 \
#     model.fm.path=ve model.fm.t_sampler=beta \
#     model.solver.type=rk4 model.solver.nfe=32

# cosine 采样下开启 t_weight=cosine（各 path）
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_15_path-linear_ot_ts-cosine_tw-cosine \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_15_path-linear_ot_ts-cosine_tw-cosine \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.fm.t_sampler=cosine model.fm.t_weight=cosine \
    model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_16_path-vp_ts-cosine_tw-cosine \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_16_path-vp_ts-cosine_tw-cosine \
#     model.scheduler.t_max=500 \
#     model.fm.path=vp model.fm.t_sampler=cosine model.fm.t_weight=cosine \
#     model.solver.type=rk4 model.solver.nfe=32

# python train_lightning.py trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gpt5/exp_17_path-ve_ts-cosine_tw-cosine \
#     wandb.group=fm_sweep_gpt5 wandb.name=exp_17_path-ve_ts-cosine_tw-cosine \
#     model.scheduler.t_max=500 \
#     model.fm.path=ve model.fm.t_sampler=cosine model.fm.t_weight=cosine \
#     model.solver.type=rk4 model.solver.nfe=32

# --- 实验组 3: 采样 ODE 求解器（固定 path=linear_ot）---
echo "--- Group 3: ODE Solver ---"

# Heun：NFE 16/32/64
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_18_solver-heun_nfe16 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_18_solver-heun_nfe16 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=heun model.solver.nfe=16

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_19_solver-heun_nfe32 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_19_solver-heun_nfe32 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=heun model.solver.nfe=32

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_20_solver-heun_nfe64 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_20_solver-heun_nfe64 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=heun model.solver.nfe=64

# RK4：NFE 16/64
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_21_solver-rk4_nfe16 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_21_solver-rk4_nfe16 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk4 model.solver.nfe=16

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_22_solver-rk4_nfe64 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_22_solver-rk4_nfe64 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk4 model.solver.nfe=64

# RK45（自适应）：rtol 扫描
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_23_solver-rk45_rtol1e-3 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_23_solver-rk45_rtol1e-3 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk45 model.solver.rtol=1e-3 model.solver.atol=1e-6

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_24_solver-rk45_rtol5e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_24_solver-rk45_rtol5e-4 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk45 model.solver.rtol=5e-4 model.solver.atol=1e-6

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_25_solver-rk45_rtol1e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_25_solver-rk45_rtol1e-4 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk45 model.solver.rtol=1e-4 model.solver.atol=1e-7

# RK45：最小步长
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_26_solver-rk45_rtol5e-4_min1e-3 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_26_solver-rk45_rtol5e-4_min1e-3 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk45 model.solver.rtol=5e-4 model.solver.min_step=1e-3

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_27_solver-rk45_rtol5e-4_min1e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_27_solver-rk45_rtol5e-4_min1e-4 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk45 model.solver.rtol=5e-4 model.solver.min_step=1e-4

# 积分方向（对比）
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_28_solver-rk4_reverse_true \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_28_solver-rk4_reverse_true \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk4 model.solver.nfe=32 model.solver.reverse_time=true

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_29_solver-rk4_reverse_false \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_29_solver-rk4_reverse_false \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot \
    model.solver.type=rk4 model.solver.nfe=32 model.solver.reverse_time=false

# --- 实验组 4: Classifier-Free Guidance（固定 path=linear_ot, rk4@32）---
echo "--- Group 4: CFG Method × Scale ---"

# method ∈ {basic, clipped, rescaled, adaptive} × scale ∈ {1.5, 3.0, 5.0}
# basic
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_30_cfg-basic_scale-1p5 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_30_cfg-basic_scale-1p5 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=basic model.guidance.scale=1.5

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_31_cfg-basic_scale-3p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_31_cfg-basic_scale-3p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=basic model.guidance.scale=3.0

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_32_cfg-basic_scale-5p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_32_cfg-basic_scale-5p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=basic model.guidance.scale=5.0

# clipped
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_33_cfg-clipped_scale-1p5 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_33_cfg-clipped_scale-1p5 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=clipped model.guidance.scale=1.5

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_34_cfg-clipped_scale-3p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_34_cfg-clipped_scale-3p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=clipped model.guidance.scale=3.0

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_35_cfg-clipped_scale-5p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_35_cfg-clipped_scale-5p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=clipped model.guidance.scale=5.0

# rescaled
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_36_cfg-rescaled_scale-1p5 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_36_cfg-rescaled_scale-1p5 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=rescaled model.guidance.scale=1.5

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_37_cfg-rescaled_scale-3p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_37_cfg-rescaled_scale-3p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=rescaled model.guidance.scale=3.0

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_38_cfg-rescaled_scale-5p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_38_cfg-rescaled_scale-5p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=rescaled model.guidance.scale=5.0

# adaptive
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_39_cfg-adaptive_scale-1p5 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_39_cfg-adaptive_scale-1p5 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=adaptive model.guidance.scale=1.5

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_40_cfg-adaptive_scale-3p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_40_cfg-adaptive_scale-3p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=adaptive model.guidance.scale=3.0

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_41_cfg-adaptive_scale-5p0 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_41_cfg-adaptive_scale-5p0 \
    model.scheduler.t_max=500 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32 \
    model.guidance.enable_cfg=true model.guidance.method=adaptive model.guidance.scale=5.0

# --- 实验组 5: 优化器 × 调度器组合（6 个）---
echo "--- Group 5: Optimizer × Scheduler Combos ---"

# CosineAnnealing（4 个 LR）
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_42_sch-cosine_lr-6e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_42_sch-cosine_lr-6e-4 \
    model.scheduler.t_max=500 model.scheduler.min_lr=1e-6 model.scheduler.name=cosine \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0006

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_43_sch-cosine_lr-4e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_43_sch-cosine_lr-4e-4 \
    model.scheduler.t_max=500 model.scheduler.min_lr=1e-6 model.scheduler.name=cosine \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0004

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_44_sch-cosine_lr-3e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_44_sch-cosine_lr-3e-4 \
    model.scheduler.t_max=500 model.scheduler.min_lr=1e-6 model.scheduler.name=cosine \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0003

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_45_sch-cosine_lr-2e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_45_sch-cosine_lr-2e-4 \
    model.scheduler.t_max=500 model.scheduler.min_lr=1e-6 model.scheduler.name=cosine \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0002

# StepLR（2 个 LR）
python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_46_sch-steplr_lr-4e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_46_sch-steplr_lr-4e-4 \
    model.scheduler.name=steplr model.scheduler.step_size=100 model.scheduler.step_gamma=0.5 \
    model.scheduler.t_max=500 \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0004

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_47_sch-steplr_lr-3e-4 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_47_sch-steplr_lr-3e-4 \
    model.scheduler.name=steplr model.scheduler.step_size=100 model.scheduler.step_gamma=0.5 \
    model.scheduler.t_max=500 \
    model.optimizer.name=adamw model.optimizer.weight_decay=1e-3 model.optimizer.lr=0.0003

# --- 实验组 6: 随机流（SFM）强度 ---
echo "--- Group 6: Stochastic Flow Matching (sigma) ---"

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_48_sfm_sigma-0p05 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_48_sfm_sigma-0p05 \
    model.scheduler.t_max=500 \
    model.fm.variant=sfm model.fm.sigma=0.05 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32

python train_lightning.py trainer.devices=1 \
    save_root=./experiments/fm_sweep_gpt5/exp_49_sfm_sigma-0p10 \
    wandb.group=fm_sweep_gpt5 wandb.name=exp_49_sfm_sigma-0p10 \
    model.scheduler.t_max=500 \
    model.fm.variant=sfm model.fm.sigma=0.10 \
    model.fm.path=linear_ot model.solver.type=rk4 model.solver.nfe=32
