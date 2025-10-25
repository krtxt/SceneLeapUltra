#!/bin/bash
set -e
set -x

# # --- 实验组 0: 基线实验 ---
# # 修正了 scheduler.t_max=500 以匹配 config.yaml 中的 epochs=500
# echo "--- 1. Running Baseline (t_max=500) ---"
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_01_baseline \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_01_baseline \
#     model.scheduler.t_max=500

# # --- 实验组 1: 优化器与调度器 ---
# # 均基于 baseline (t_max=500)
# echo "--- Group 1: Optimizer & Scheduler ---"

# # Exp 2: LR=1e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_02_lr_1e-4 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_02_lr_1e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0001

# # Exp 3: LR=3e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_03_lr_3e-4 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_03_lr_3e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0003

# # Exp 4: LR=5e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_04_lr_5e-4 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_04_lr_5e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.0005

# # Exp 5: LR=1e-3
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_05_lr_1e-3 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_05_lr_1e-3 \
#     model.scheduler.t_max=500 \
#     model.optimizer.lr=0.001

# # Exp 6: WD=1e-4
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_06_wd_1e-4 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_06_wd_1e-4 \
#     model.scheduler.t_max=500 \
#     model.optimizer.weight_decay=0.0001

# # Exp 7: WD=1e-2
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_07_wd_1e-2 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_07_wd_1e-2 \
#     model.scheduler.t_max=500 \
#     model.optimizer.weight_decay=0.01

# # Exp 8: Optim=adam
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_08_optim_adam \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_08_optim_adam \
#     model.scheduler.t_max=500 \
#     model.optimizer.name=adam

# # Exp 9: Min_LR=1e-7
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_09_min_lr_1e-7 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_09_min_lr_1e-7 \
#     model.scheduler.t_max=500 \
#     model.scheduler.min_lr=1.0e-7

# # Exp 10: Min_LR=1e-5
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_10_min_lr_1e-5 \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_10_min_lr_1e-5 \
#     model.scheduler.t_max=500 \
#     model.scheduler.min_lr=1.0e-5

# Exp 11: Sched=steplr
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_11_sched_steplr \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_11_sched_steplr \
    model.scheduler.name=steplr \
    model.scheduler.step_size=100 \
    model.scheduler.step_gamma=0.5 \
    model.scheduler.t_max=null \
    model.scheduler.min_lr=null

# --- 实验组 2: Flow Matching 核心参数 ---
echo "--- Group 2: FM Core Parameters ---"

# # Exp 12: Path=vp
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_12_path_vp \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_12_path_vp \
#     model.scheduler.t_max=500 \
#     model.fm.path=vp

# # Exp 13: Path=ve
# python train_lightning.py \
#     trainer.devices=1 \
#     save_root=./experiments/fm_sweep_gemini/exp_13_path_ve \
#     wandb.group=fm_sweep_gemini \
#     wandb.name=exp_13_path_ve \
#     model.scheduler.t_max=500 \
#     model.fm.path=ve

# Exp 14: T-Sampler=cosine
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_14_tsampler_cosine \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_14_tsampler_cosine \
    model.scheduler.t_max=500 \
    model.fm.t_sampler=cosine

# Exp 15: T-Sampler=beta
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_15_tsampler_beta \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_15_tsampler_beta \
    model.scheduler.t_max=500 \
    model.fm.t_sampler=beta

# Exp 16: T-Weight=cosine
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_16_tweight_cosine \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_16_tweight_cosine \
    model.scheduler.t_max=500 \
    model.fm.t_weight=cosine

# Exp 17: T-Weight=beta
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_17_tweight_beta \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_17_tweight_beta \
    model.scheduler.t_max=500 \
    model.fm.t_weight=beta

# Exp 18: SFM Sigma=0.01
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_18_sfm_0.01 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_18_sfm_0.01 \
    model.scheduler.t_max=500 \
    model.fm.sfm.sigma=0.01

# Exp 19: SFM Sigma=0.1
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_19_sfm_0.1 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_19_sfm_0.1 \
    model.scheduler.t_max=500 \
    model.fm.sfm.sigma=0.1

# Exp 20: SFM Sigma=0.5
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_20_sfm_0.5 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_20_sfm_0.5 \
    model.scheduler.t_max=500 \
    model.fm.sfm.sigma=0.5

# --- 实验组 3: 求解器参数 (仅影响验证) ---
echo "--- Group 3: Solver Parameters (Validation/Inference) ---"

# Exp 21: Solver=heun
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_21_solver_heun \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_21_solver_heun \
    model.scheduler.t_max=500 \
    model.solver.type=heun

# Exp 22: Solver=rk45
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_22_solver_rk45 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_22_solver_rk45 \
    model.scheduler.t_max=500 \
    model.solver.type=rk45

# Exp 23: NFE=16
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_23_nfe_16 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_23_nfe_16 \
    model.scheduler.t_max=500 \
    model.solver.nfe=16

# Exp 24: NFE=64
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_24_nfe_64 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_24_nfe_64 \
    model.scheduler.t_max=500 \
    model.solver.nfe=64

# Exp 25: NFE=128
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_25_nfe_128 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_25_nfe_128 \
    model.scheduler.t_max=500 \
    model.solver.nfe=128

# Exp 26: RK45 Loose Tol
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_26_rk45_loose \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_26_rk45_loose \
    model.scheduler.t_max=500 \
    model.solver.type=rk45 \
    model.solver.rtol=1e-2 \
    model.solver.atol=1e-4

# Exp 27: RK45 Strict Tol
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_27_rk45_strict \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_27_rk45_strict \
    model.scheduler.t_max=500 \
    model.solver.type=rk45 \
    model.solver.rtol=1e-4 \
    model.solver.atol=1e-6

# --- 实验组 4: Classifier-Free Guidance ---
echo "--- Group 4: Classifier-Free Guidance (CFG) ---"

# Exp 28: Enable CFG (Scale 3.0, Drop 0.1)
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_28_cfg_true_scale_3.0 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_28_cfg_true_scale_3.0 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true

# Exp 29: CFG Scale 1.0
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_29_cfg_scale_1.0 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_29_cfg_scale_1.0 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.scale=1.0

# Exp 30: CFG Scale 1.5
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_30_cfg_scale_1.5 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_30_cfg_scale_1.5 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.scale=1.5

# Exp 31: CFG Scale 5.0
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_31_cfg_scale_5.0 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_31_cfg_scale_5.0 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.scale=5.0

# Exp 32: CFG Scale 7.5
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_32_cfg_scale_7.5 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_32_cfg_scale_7.5 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.scale=7.5

# Exp 33: CFG Drop 0.05
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_33_cfg_drop_0.05 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_33_cfg_drop_0.05 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.cond_drop_prob=0.05

# Exp 34: CFG Drop 0.20
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_34_cfg_drop_0.20 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_34_cfg_drop_0.20 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.cond_drop_prob=0.20

# --- 实验组 5: 组合与探索 ---
echo "--- Group 5: Combinations & Refinements ---"

# Exp 35: Cosine Sampler + Cosine Weight
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_35_cosine_sampler_weight \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_35_cosine_sampler_weight \
    model.scheduler.t_max=500 \
    model.fm.t_sampler=cosine \
    model.fm.t_weight=cosine

# Exp 36: Beta Sampler + Beta Weight
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_36_beta_sampler_weight \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_36_beta_sampler_weight \
    model.scheduler.t_max=500 \
    model.fm.t_sampler=beta \
    model.fm.t_weight=beta

# Exp 37: CFG + PC Correction
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_37_cfg_pc_true \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_37_cfg_pc_true \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.pc_correction=true

# Exp 38: CFG + PC Correction (2 steps)
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_38_cfg_pc_num_2 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_38_cfg_pc_num_2 \
    model.scheduler.t_max=500 \
    model.guidance.enable_cfg=true \
    model.guidance.pc_correction=true \
    model.guidance.num_corrections=2

# Exp 39: RK45 max_step=1/16
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_39_rk45_maxstep_1_16 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_39_rk45_maxstep_1_16 \
    model.scheduler.t_max=500 \
    model.solver.type=rk45 \
    model.solver.max_step=0.0625

# Exp 40: RK45 max_step=1/8
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_40_rk45_maxstep_1_8 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_40_rk45_maxstep_1_8 \
    model.scheduler.t_max=500 \
    model.solver.type=rk45 \
    model.solver.max_step=0.125

# Exp 41: LR=5e-5
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_41_lr_5e-5 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_41_lr_5e-5 \
    model.scheduler.t_max=500 \
    model.optimizer.lr=0.00005

# Exp 42: LR=1e-5
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_gemini/exp_42_lr_1e-5 \
    wandb.group=fm_sweep_gemini \
    wandb.name=exp_42_lr_1e-5 \
    model.scheduler.t_max=500 \
    model.optimizer.lr=0.00001

echo "--- All 42 experiments completed ---"