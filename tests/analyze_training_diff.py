#!/usr/bin/env python3
"""
深度分析训练差异的可能原因
"""

import sys
import torch
import pytorch_lightning as pl
import yaml
from pathlib import Path

print("="*80)
print("训练差异诊断报告")
print("="*80)

# 1. 检查PyTorch环境
print("\n1. PyTorch环境信息")
print("-"*80)
print(f"PyTorch版本: {torch.__version__}")
print(f"PyTorch Lightning版本: {pl.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 2. 检查关键配置
print("\n2. 训练配置对比")
print("-"*80)

config_path = Path("/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra/experiments/diffuser_objcentric_mini_pointnet2_moreepochs/config/whole_config.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

print(f"随机种子: {cfg.get('seed', 'NOT SET')}")
print(f"Batch size: {cfg.get('batch_size', 'NOT SET')}")
print(f"Learning rate: {cfg.get('model', {}).get('optimizer', {}).get('lr', 'NOT SET')}")
print(f"Optimizer: {cfg.get('model', {}).get('optimizer', {}).get('name', 'NOT SET')}")
print(f"Scheduler: {cfg.get('model', {}).get('scheduler', {}).get('name', 'NOT SET')}")
print(f"Benchmark: {cfg.get('trainer', {}).get('benchmark', 'NOT SET')}")
print(f"Precision: {cfg.get('trainer', {}).get('precision', 'NOT SET')}")
print(f"Gradient clip: {cfg.get('trainer', {}).get('gradient_clip_val', 'NOT SET')}")

# 3. 可能的非确定性来源
print("\n3. 非确定性因素分析")
print("-"*80)

potential_issues = []

# 检查benchmark设置
if cfg.get('trainer', {}).get('benchmark', True):
    potential_issues.append({
        'issue': 'cuDNN Benchmark 开启',
        'impact': '高',
        'description': 'benchmark=True会让cuDNN选择最快但可能非确定性的算法',
        'solution': '设置 trainer.benchmark=False 或使用 torch.use_deterministic_algorithms(True)'
    })

# 检查数据加载器
if cfg.get('data_cfg', {}).get('train', {}).get('num_workers', 0) > 0:
    potential_issues.append({
        'issue': '多进程数据加载',
        'impact': '中',
        'description': 'num_workers>0 时，不同进程的随机数生成器可能导致非确定性',
        'solution': '确保 pl.seed_everything(seed, workers=True) 被调用'
    })

# 检查Flash Attention
if cfg.get('model', {}).get('decoder', {}).get('use_flash_attention', False):
    potential_issues.append({
        'issue': 'Flash Attention',
        'impact': '中',
        'description': 'Flash Attention 可能有非确定性的CUDA kernels',
        'solution': '临时设置 use_flash_attention=false 验证'
    })

# 检查数据缓存
use_cached = cfg.get('data_cfg', {}).get('train', {}).get('use_cached', False)
if not use_cached:
    potential_issues.append({
        'issue': '未使用数据缓存',
        'impact': '高',
        'description': '每次训练动态加载数据，如果数据处理有随机性或顺序不确定，会导致不同结果',
        'solution': '使用 use_cached=true 或检查数据处理pipeline的确定性'
    })

# 检查采样策略
sampling_strategy = cfg.get('exhaustive_sampling_strategy', 'NOT SET')
if sampling_strategy != 'NOT SET':
    potential_issues.append({
        'issue': f'数据采样策略: {sampling_strategy}',
        'impact': '中',
        'description': '不同的采样策略可能导致不同的训练数据顺序',
        'solution': '确保采样策略的确定性实现'
    })

# 打印所有潜在问题
for i, issue in enumerate(potential_issues, 1):
    print(f"\n问题 {i}: {issue['issue']}")
    print(f"  影响级别: {issue['impact']}")
    print(f"  描述: {issue['description']}")
    print(f"  建议: {issue['solution']}")

# 4. 关键建议
print("\n4. 关键调查建议")
print("-"*80)

recommendations = [
    "1. **CUDA非确定性** - benchmark=True是最可能的原因",
    "   建议: 设置 trainer.benchmark=false 重新训练验证",
    "",
    "2. **数据文件变化** - 检查数据集文件是否被修改",
    f"   数据路径: {cfg.get('data_cfg', {}).get('train', {}).get('succ_grasp_dir', 'NOT SET')}",
    "   建议: 对比数据文件的MD5哈希值",
    "",
    "3. **第三方库重编译** - PointNet2 CUDA扩展可能被重新编译",
    "   建议: 检查 third_party/pointnet2/ 的编译时间戳",
    "",
    "4. **PyTorch/CUDA版本差异** - 不同版本的行为可能不同",
    "   建议: 对比两次训练的环境版本",
    "",
    "5. **checkpoint恢复** - 检查是否意外从不同checkpoint恢复",
    f"   Resume: {cfg.get('resume', False)}",
    f"   Checkpoint: {cfg.get('checkpoint_path', 'None')}",
]

for rec in recommendations:
    print(rec)

# 5. 立即验证步骤
print("\n5. 立即验证步骤")
print("-"*80)
print("运行以下命令进行快速验证:")
print()
print("# Step 1: 检查数据文件")
print("find /home/xiantuo/source/grasp/SceneLeapUltra/data/1022_mini_succ_collect -type f -exec md5sum {} \\; | sort > data_checksum.txt")
print()
print("# Step 2: 检查PointNet2编译时间")
print("ls -lh third_party/pointnet2/build/")
print()
print("# Step 3: 使用确定性设置重新训练几个epoch")
print("python train_lightning.py \\")
print("  model=diffuser/diffuser \\")
print("  model.decoder.backbone=pointnet2 \\")
print("  trainer.benchmark=false \\")
print("  trainer.max_epochs=10 \\")
print("  save_root=./experiments/deterministic_test")

print("\n" + "="*80)

