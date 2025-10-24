#!/bin/bash

# SceneLeapPro 原始训练脚本 (兼容性保留)
#
# ⚠️  推荐使用: ./train_distributed.sh
#
# 该脚本主要用于向后兼容，新项目建议使用 train_distributed.sh
# train_distributed.sh 提供更好的多GPU支持和Hydra语法

# 默认参数
GPUS=""
USE_DISTRIBUTED="auto"
BATCH_SIZE=""
LEARNING_RATE=""
SAVE_ROOT=""
EXTRA_ARGS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --distributed)
            USE_DISTRIBUTED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --save_root)
            SAVE_ROOT="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 检测可用GPU数量
if [ -z "$GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "检测到 $AVAILABLE_GPUS 个可用GPU"

        # 如果有多个GPU，默认使用所有GPU
        if [ "$AVAILABLE_GPUS" -gt 1 ]; then
            GPUS=$AVAILABLE_GPUS
            echo "将使用所有 $GPUS 个GPU进行训练"
        else
            GPUS=1
            echo "使用单GPU训练"
        fi
    else
        echo "警告: 无法检测GPU，默认使用单GPU"
        GPUS=1
    fi
fi

# 设置CUDA_VISIBLE_DEVICES
if [ "$GPUS" -eq 1 ]; then
    # 单GPU训练，使用GPU 0
    export CUDA_VISIBLE_DEVICES=0
    echo "设置 CUDA_VISIBLE_DEVICES=0"
else
    # 多GPU训练，暴露所有需要的GPU
    GPU_LIST=$(seq -s, 0 $((GPUS-1)))
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    echo "设置 CUDA_VISIBLE_DEVICES=$GPU_LIST"
fi

# 构建训练命令
TRAIN_CMD="python train_lightning.py"

# 添加分布式配置
if [ "$GPUS" -gt 1 ]; then
    TRAIN_CMD="$TRAIN_CMD distributed.enabled=$USE_DISTRIBUTED"
    TRAIN_CMD="$TRAIN_CMD distributed.devices=$GPUS"
    TRAIN_CMD="$TRAIN_CMD trainer.devices=$GPUS"
    TRAIN_CMD="$TRAIN_CMD trainer.strategy=ddp"
fi

# 添加可选参数
if [ ! -z "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD batch_size=$BATCH_SIZE"
fi

if [ ! -z "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD model.optimizer.lr=$LEARNING_RATE"
fi

if [ ! -z "$SAVE_ROOT" ]; then
    TRAIN_CMD="$TRAIN_CMD save_root=$SAVE_ROOT"
else
    # 默认保存路径
    TRAIN_CMD="$TRAIN_CMD save_root=experiments/test_7_5_diff_1"
fi

# 添加额外参数
TRAIN_CMD="$TRAIN_CMD $EXTRA_ARGS"

echo "训练配置:"
echo "  GPU数量: $GPUS"
echo "  分布式训练: $USE_DISTRIBUTED"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ ! -z "$BATCH_SIZE" ]; then
    echo "  批次大小: $BATCH_SIZE"
fi
if [ ! -z "$LEARNING_RATE" ]; then
    echo "  学习率: $LEARNING_RATE"
fi
echo ""
echo "执行命令: $TRAIN_CMD"
echo ""

# 执行训练
eval $TRAIN_CMD