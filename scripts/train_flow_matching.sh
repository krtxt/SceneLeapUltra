#!/bin/bash
# Flow Matching 训练启动脚本
#
# 使用方法:
#   bash scripts/train_flow_matching.sh [config_name]
#
# 示例:
#   bash scripts/train_flow_matching.sh baseline
#   bash scripts/train_flow_matching.sh with_cfg

set -e

# 激活环境
source ~/.bashrc
conda activate DexGrasp

# 配置名称 (默认: baseline)
CONFIG=${1:-baseline}

echo "========================================"
echo "Flow Matching 训练"
echo "配置: $CONFIG"
echo "========================================"

case $CONFIG in
    baseline)
        echo "使用基础配置 (RK4, NFE=32, 无CFG)"
        python train_lightning.py \
            model=flow_matching \
            model.name=GraspFlowMatching \
            model.fm.path=linear_ot \
            model.fm.t_sampler=cosine \
            model.solver.type=rk4 \
            model.solver.nfe=32 \
            model.guidance.enable_cfg=false \
            data=sceneleapplus \
            batch_size=96 \
            epochs=500 \
            save_root=./experiments/fm_baseline
        ;;
    
    with_cfg)
        echo "使用CFG配置 (RK4, NFE=32, CFG=3.0)"
        python train_lightning.py \
            model=flow_matching \
            model.name=GraspFlowMatching \
            model.fm.path=linear_ot \
            model.fm.t_sampler=cosine \
            model.solver.type=rk4 \
            model.solver.nfe=32 \
            model.guidance.enable_cfg=true \
            model.guidance.scale=3.0 \
            model.guidance.diff_clip=5.0 \
            use_text_condition=true \
            data=sceneleapplus \
            batch_size=96 \
            epochs=500 \
            save_root=./experiments/fm_with_cfg
        ;;
    
    fast)
        echo "使用快速配置 (Heun, NFE=16)"
        python train_lightning.py \
            model=flow_matching \
            model.name=GraspFlowMatching \
            model.fm.path=linear_ot \
            model.fm.t_sampler=uniform \
            model.solver.type=heun \
            model.solver.nfe=16 \
            model.guidance.enable_cfg=false \
            data=sceneleapplus \
            batch_size=128 \
            epochs=300 \
            save_root=./experiments/fm_fast
        ;;
    
    high_quality)
        echo "使用高质量配置 (RK4, NFE=64, CFG=5.0)"
        python train_lightning.py \
            model=flow_matching \
            model.name=GraspFlowMatching \
            model.fm.path=linear_ot \
            model.fm.t_sampler=beta \
            model.fm.t_weight=cosine \
            model.solver.type=rk4 \
            model.solver.nfe=64 \
            model.guidance.enable_cfg=true \
            model.guidance.scale=5.0 \
            model.guidance.diff_clip=5.0 \
            use_text_condition=true \
            data=sceneleapplus \
            batch_size=64 \
            epochs=500 \
            save_root=./experiments/fm_high_quality
        ;;
    
    test)
        echo "使用测试配置 (快速验证)"
        python train_lightning.py \
            model=flow_matching \
            model.name=GraspFlowMatching \
            model.fm.path=linear_ot \
            model.solver.type=rk4 \
            model.solver.nfe=32 \
            data=sceneleapplus \
            batch_size=32 \
            epochs=10 \
            save_root=./experiments/fm_test
        ;;
    
    *)
        echo "未知配置: $CONFIG"
        echo ""
        echo "可用配置:"
        echo "  baseline     - 基础配置 (推荐默认)"
        echo "  with_cfg     - 启用CFG的配置"
        echo "  fast         - 快速训练配置"
        echo "  high_quality - 高质量配置"
        echo "  test         - 测试配置 (10 epochs)"
        echo ""
        echo "使用方法: bash scripts/train_flow_matching.sh [config_name]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "训练完成！"
echo "========================================"

