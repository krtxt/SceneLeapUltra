echo "实验1.1: 当前基线配置验证 (100步)"
bash train_distributed.sh --gpus 4 \
    save_root="./experiments/baseline_steps_100"

# 1.2 扩散步数影响测试
echo "实验1.2: 扩散步数50步 - 快速推理配置"
bash train_distributed.sh --gpus 4 \
    save_root="./experiments/steps_50" \
    model.steps=50

echo "实验1.3: 扩散步数200步 - 高质量配置"
bash train_distributed.sh --gpus 4 \
    save_root="./experiments/steps_200" \
    model.steps=200

# echo "实验1.4: 扩散步数500步 - 极高质量配置"
# bash train_distributed.sh --gpus 4 \
#     save_root="./experiments/steps_500" \
#     model.steps=500

echo "实验1.5: 扩散步数25步 - 超快速配置"
bash train_distributed.sh --gpus 4 \
    save_root="./experiments/steps_25" \
    model.steps=25



# # 2.1 学习率搜索
echo "实验2.1: 低学习率配置"
bash train_distributed.sh --gpus 4 \
    save_root="./experiments/lr_5e5" \
    model.optimizer.lr=5e-5

# echo "实验2.2: 高学习率配置"
# bash train_distributed.sh --gpus 4 \
#     save_root="./experiments/lr_2e4" \
#     model.optimizer.lr=2e-4

# echo "实验2.3: 中高学习率配置"
# bash train_distributed.sh --gpus 4 \
#     save_root="./experiments/lr_1.5e4" \
#     model.optimizer.lr=1.5e-4