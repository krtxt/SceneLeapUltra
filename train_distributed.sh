# xiantuo@Oppenheimer:~$ ssh -N -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -R 1101:localhost:22 root@38.150.2.96 -p 38268

#!/bin/bash

# SceneLeapPro 分布式训练脚本 (推荐使用)
# 支持单GPU和多GPU训练，使用标准Hydra命令行语法
#
# 🚀 快速开始:
#   自动检测GPU: ./train_distributed.sh
#   指定GPU数量: ./train_distributed.sh --gpus 4
#   调整参数:    ./train_distributed.sh --gpus 4 batch_size=128 model.optimizer.lr=0.002
#
# 📊 Batch Size 逻辑:
#   batch_size=128 表示全局有效batch_size
#   4GPU训练时，每个GPU处理 128÷4=32 个样本
#
# 🎯 Learning Rate 逻辑:
#   model.optimizer.lr=0.001 表示基础学习率
#   系统会根据GPU数量自动缩放 (默认sqrt缩放)
#
# 📝 配置优先级:
#   命令行Hydra参数 > config.yaml > 默认值

set -e

# 分布式训练专用参数（非Hydra参数）
GPUS=""
NODES=1
MASTER_ADDR="localhost"
MASTER_PORT=29501
JOB_ID="sceneleap_$(date +%Y%m%d_%H%M%S)"

# Hydra配置覆盖参数
HYDRA_OVERRIDES=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --job_id)
            JOB_ID="$2"
            shift 2
            ;;
        --help)
            echo "SceneLeapPro 分布式训练脚本"
            echo ""
            echo "分布式参数:"
            echo "  --gpus N              使用N个GPU (默认: 自动检测)"
            echo "  --nodes N             节点数量 (默认: 1)"
            echo "  --master_addr ADDR    主节点地址 (多节点必需)"
            echo "  --master_port PORT    主节点端口 (默认: 29500, 自动检测可用端口)"
            echo "  --job_id ID           作业ID (默认: 自动生成)"
            echo ""
            echo "Hydra配置覆盖 (直接传递给train_lightning.py):"
            echo "  batch_size=N          批次大小"
            echo "  model.optimizer.lr=X  学习率"
            echo "  epochs=N              训练轮数"
            echo "  save_root=PATH        保存路径"
            echo "  distributed.lr_scaling=METHOD  学习率缩放方法"
            echo ""
            echo "示例:"
            echo "  ./train_distributed.sh --gpus 4 batch_size=128 model.optimizer.lr=0.002"
            echo "  ./train_distributed.sh batch_size=64 epochs=500 save_root=experiments/test"
            exit 0
            ;;
        *)
            # 所有其他参数都作为Hydra配置覆盖
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES $1"
            shift
            ;;
    esac
done

# 查找可用端口的函数
find_free_port() {
    local start_port=$1
    local port=$start_port
    while netstat -ln 2>/dev/null | grep -q ":$port " || ss -ln 2>/dev/null | grep -q ":$port "; do
        port=$((port + 1))
        # 避免无限循环
        if [ $port -gt $((start_port + 100)) ]; then
            echo "错误: 无法找到可用端口 (尝试了 $start_port 到 $port)"
            exit 1
        fi
    done
    echo $port
}

# 检测GPU数量
if [ -z "$GPUS" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "自动检测到 $GPUS 个GPU"
    else
        echo "错误: 无法检测GPU数量，请手动指定 --gpus 参数"
        exit 1
    fi
fi

# 检查GPU数量
if [ "$GPUS" -lt 1 ]; then
    echo "错误: GPU数量必须大于0"
    exit 1
fi

# 自动查找可用端口
if netstat -ln 2>/dev/null | grep -q ":$MASTER_PORT " || ss -ln 2>/dev/null | grep -q ":$MASTER_PORT "; then
    echo "⚠️  端口 $MASTER_PORT 已被占用，自动查找可用端口..."
    MASTER_PORT=$(find_free_port $MASTER_PORT)
    echo "✅ 使用端口: $MASTER_PORT"
fi

# 自动添加分布式配置到Hydra覆盖参数
if [ "$GPUS" -gt 1 ]; then
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES distributed.enabled=true"
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES distributed.devices=$GPUS"
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES trainer.devices=$GPUS"
    HYDRA_OVERRIDES="$HYDRA_OVERRIDES trainer.strategy=ddp"
fi

echo "🚀 分布式训练配置:"
echo "  GPU数量: $GPUS"
echo "  节点数量: $NODES"
echo "  主节点地址: $MASTER_ADDR"
echo "  主节点端口: $MASTER_PORT"
echo "  作业ID: $JOB_ID"
echo "  CUDA设备: $CUDA_VISIBLE_DEVICES"
echo ""
echo "📝 Hydra配置覆盖:"
echo "  $HYDRA_OVERRIDES"
echo ""

# 检查是否在SLURM环境中
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "检测到SLURM环境，作业ID: $SLURM_JOB_ID"
    
    # 在SLURM环境中使用srun
    srun --ntasks-per-node=$GPUS \
         --nodes=$NODES \
         --gres=gpu:$GPUS \
         python train_lightning.py \
         distributed.num_nodes=$NODES \
         $HYDRA_OVERRIDES

elif [ ! -z "$LOCAL_RANK" ]; then
    echo "检测到torchrun环境"
    
    # 已经在torchrun环境中，直接运行
    python train_lightning.py \
        distributed.num_nodes=$NODES \
        $HYDRA_OVERRIDES

else
    echo "使用torchrun启动分布式训练"
    
    # 构建torchrun命令
    TORCHRUN_CMD="torchrun --nproc_per_node=$GPUS --nnodes=$NODES"
    
    # 添加端口配置（单节点和多节点都需要）
    TORCHRUN_CMD="$TORCHRUN_CMD --master_port=$MASTER_PORT"
    
    # 多节点配置
    if [ "$NODES" -gt 1 ]; then
        TORCHRUN_CMD="$TORCHRUN_CMD --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
        TORCHRUN_CMD="$TORCHRUN_CMD --rdzv_backend=c10d"
        TORCHRUN_CMD="$TORCHRUN_CMD --rdzv_id=$JOB_ID"
    fi
    
    # 添加训练脚本和参数
    TORCHRUN_CMD="$TORCHRUN_CMD train_lightning.py"
    TORCHRUN_CMD="$TORCHRUN_CMD distributed.num_nodes=$NODES"
    TORCHRUN_CMD="$TORCHRUN_CMD $HYDRA_OVERRIDES"
    
    echo "执行命令: $TORCHRUN_CMD"
    echo ""
    
    # 执行训练
    eval $TORCHRUN_CMD
fi

echo "分布式训练完成"


# bash train_distributed.sh --gpus 1 save_root="./experiments/testtest_ptv3" model.steps=100 batch_size=40 'checkpoint_path="experiments/testtest_ptv3/checkpoints/epoch=34-val_loss=56.94.ckpt"'