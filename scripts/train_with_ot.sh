#!/bin/bash
#
# Flow Matching 训练脚本 - 使用 Optimal Transport 配对
#
# 使用方式：
#   bash scripts/train_with_ot.sh
#
# 功能：对比有无OT配对的训练效果

set -e

# 激活环境
source ~/.bashrc
conda activate DexGrasp

# 基础配置
DATASET="sceneleappro"
ROT_TYPE="r6d"
MODE="camera_centric_scene_mean_normalized"
BATCH_SIZE=32
NUM_WORKERS=8
GPUS=1

# 训练配置
EPOCHS=50
SAVE_ROOT="./experiments/ot_ablation"

echo "========================================="
echo "Flow Matching with Optimal Transport"
echo "========================================="
echo "Dataset: $DATASET"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Save Root: $SAVE_ROOT"
echo ""

# ========================================
# 实验1: Baseline (无OT配对)
# ========================================
echo "[实验1] Baseline - 随机索引配对"
echo "-----------------------------------------"

python train_lightning.py \
    model=flow_matching \
    data_cfg=$DATASET \
    rot_type=$ROT_TYPE \
    mode=$MODE \
    batch_size=$BATCH_SIZE \
    num_workers=$NUM_WORKERS \
    gpus=$GPUS \
    epochs=$EPOCHS \
    save_root="${SAVE_ROOT}/baseline_no_ot" \
    model.fm.optimal_transport.enable=false \
    wandb.name="FM_baseline_no_ot" \
    wandb.tags="[flow_matching,baseline,no_ot]"

echo "✅ 实验1完成"
echo ""

# ========================================
# 实验2: OT配对 (reg=0.1)
# ========================================
echo "[实验2] OT配对 - reg=0.1 (平衡)"
echo "-----------------------------------------"

python train_lightning.py \
    model=flow_matching \
    data_cfg=$DATASET \
    rot_type=$ROT_TYPE \
    mode=$MODE \
    batch_size=$BATCH_SIZE \
    num_workers=$NUM_WORKERS \
    gpus=$GPUS \
    epochs=$EPOCHS \
    save_root="${SAVE_ROOT}/ot_reg01" \
    model.fm.optimal_transport.enable=true \
    model.fm.optimal_transport.reg=0.1 \
    model.fm.optimal_transport.num_iters=50 \
    model.fm.optimal_transport.start_epoch=0 \
    wandb.name="FM_ot_reg01" \
    wandb.tags="[flow_matching,optimal_transport,reg_01]"

echo "✅ 实验2完成"
echo ""

# ========================================
# 实验3: OT配对 (reg=0.05, 更精确)
# ========================================
echo "[实验3] OT配对 - reg=0.05 (更精确)"
echo "-----------------------------------------"

python train_lightning.py \
    model=flow_matching \
    data_cfg=$DATASET \
    rot_type=$ROT_TYPE \
    mode=$MODE \
    batch_size=$BATCH_SIZE \
    num_workers=$NUM_WORKERS \
    gpus=$GPUS \
    epochs=$EPOCHS \
    save_root="${SAVE_ROOT}/ot_reg005" \
    model.fm.optimal_transport.enable=true \
    model.fm.optimal_transport.reg=0.05 \
    model.fm.optimal_transport.num_iters=100 \
    model.fm.optimal_transport.start_epoch=0 \
    wandb.name="FM_ot_reg005" \
    wandb.tags="[flow_matching,optimal_transport,reg_005]"

echo "✅ 实验3完成"
echo ""

# ========================================
# 实验4: 延迟启用OT (先稳定再优化)
# ========================================
echo "[实验4] 延迟OT - 从epoch 5开始"
echo "-----------------------------------------"

python train_lightning.py \
    model=flow_matching \
    data_cfg=$DATASET \
    rot_type=$ROT_TYPE \
    mode=$MODE \
    batch_size=$BATCH_SIZE \
    num_workers=$NUM_WORKERS \
    gpus=$GPUS \
    epochs=$EPOCHS \
    save_root="${SAVE_ROOT}/ot_delayed" \
    model.fm.optimal_transport.enable=true \
    model.fm.optimal_transport.reg=0.1 \
    model.fm.optimal_transport.num_iters=50 \
    model.fm.optimal_transport.start_epoch=5 \
    wandb.name="FM_ot_delayed" \
    wandb.tags="[flow_matching,optimal_transport,delayed]"

echo "✅ 实验4完成"
echo ""

# ========================================
# 完成
# ========================================
echo "========================================="
echo "✅ 所有实验完成！"
echo "========================================="
echo ""
echo "结果保存在: $SAVE_ROOT"
echo ""
echo "查看WandB对比:"
echo "  https://wandb.ai/your-project"
echo ""
echo "对比指标:"
echo "  - train/loss: 训练损失"
echo "  - train/ot_matched_dist: OT配对距离"
echo "  - train/ot_improvement: OT改进百分比"
echo "  - val/loss: 验证损失"
echo ""

