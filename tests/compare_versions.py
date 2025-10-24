#!/usr/bin/env python3
"""
脚本用于比较两个版本的代码，找出可能导致训练结果不同的差异。
"""

import os
import hashlib
from pathlib import Path
import difflib

# 定义要比较的关键文件列表
KEY_FILES = [
    # 核心模型文件
    "models/decoder/dit.py",
    "models/backbone/pointnet2.py",
    "models/diffuser_lightning.py",
    
    # 训练相关
    "train_lightning.py",
    
    # 数据处理
    "datasets/objectcentric_grasp_dataset.py",
    "datasets/objectcentric_grasp_cached.py",
    "datasets/scenedex_datamodule.py",
    "datasets/utils/pointcloud_utils.py",
    
    # 工具函数
    "utils/hand_helper.py",
    "models/utils/diffusion_utils.py",
    "models/utils/diffusion_core.py",
    
    # 损失函数
    "models/loss.py",
]

def compute_file_hash(filepath):
    """计算文件的MD5哈希值"""
    if not os.path.exists(filepath):
        return None
    
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def compare_files(file1, file2):
    """比较两个文件并返回差异"""
    if not os.path.exists(file1) or not os.path.exists(file2):
        return None
    
    with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
        lines1 = f1.readlines()
    
    with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
        lines2 = f2.readlines()
    
    # 使用difflib生成差异
    diff = list(difflib.unified_diff(lines1, lines2, 
                                     fromfile=file1, 
                                     tofile=file2,
                                     lineterm=''))
    
    return diff if len(diff) > 0 else None

def main():
    # 定义两个版本的根目录
    backup_root = Path("/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra/experiments/diffuser_objcentric_mini_pointnet2_moreepochs/backups/v1")
    current_root = Path("/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra")
    
    print("=" * 80)
    print("比较训练环境关键文件")
    print("=" * 80)
    print(f"\n原始版本路径: {backup_root}")
    print(f"当前版本路径: {current_root}\n")
    
    differences_found = []
    identical_files = []
    missing_files = []
    
    for key_file in KEY_FILES:
        backup_file = backup_root / key_file
        current_file = current_root / key_file
        
        print(f"\n检查: {key_file}")
        print("-" * 80)
        
        # 检查文件是否存在
        if not backup_file.exists():
            print(f"  ❌ 备份版本文件不存在")
            missing_files.append((key_file, "backup"))
            continue
        
        if not current_file.exists():
            print(f"  ❌ 当前版本文件不存在")
            missing_files.append((key_file, "current"))
            continue
        
        # 计算哈希值
        backup_hash = compute_file_hash(str(backup_file))
        current_hash = compute_file_hash(str(current_file))
        
        if backup_hash == current_hash:
            print(f"  ✅ 文件完全相同 (MD5: {backup_hash[:8]}...)")
            identical_files.append(key_file)
        else:
            print(f"  ⚠️  文件存在差异!")
            print(f"     备份版本 MD5: {backup_hash[:16]}...")
            print(f"     当前版本 MD5: {current_hash[:16]}...")
            
            # 生成详细差异
            diff = compare_files(str(backup_file), str(current_file))
            if diff:
                differences_found.append((key_file, diff))
                print(f"     发现 {len(diff)} 行差异")
    
    # 输出汇总
    print("\n")
    print("=" * 80)
    print("汇总报告")
    print("=" * 80)
    print(f"\n✅ 相同文件数: {len(identical_files)}")
    print(f"⚠️  不同文件数: {len(differences_found)}")
    print(f"❌ 缺失文件数: {len(missing_files)}")
    
    if differences_found:
        print("\n" + "=" * 80)
        print("发现以下文件存在差异:")
        print("=" * 80)
        
        for filename, diff in differences_found:
            print(f"\n文件: {filename}")
            print("-" * 80)
            
            # 只显示前100行差异
            for line in diff[:100]:
                print(line)
            
            if len(diff) > 100:
                print(f"\n... (还有 {len(diff) - 100} 行差异未显示)")
    
    if missing_files:
        print("\n" + "=" * 80)
        print("缺失的文件:")
        print("=" * 80)
        for filename, where in missing_files:
            print(f"  - {filename} (在{where}中缺失)")
    
    # 检查配置文件
    print("\n" + "=" * 80)
    print("检查训练配置")
    print("=" * 80)
    
    config_backup = backup_root.parent.parent / "config" / "whole_config.yaml"
    config_current = current_root / "experiments" / "diffuser_objcentric_mini_pointnet2_moreepochs" / "config" / "whole_config.yaml"
    
    if config_backup.exists() and config_current.exists():
        backup_hash = compute_file_hash(str(config_backup))
        current_hash = compute_file_hash(str(config_current))
        
        if backup_hash == current_hash:
            print("  ✅ 配置文件完全相同")
        else:
            print("  ⚠️  配置文件存在差异")
            diff = compare_files(str(config_backup), str(config_current))
            if diff:
                for line in diff[:50]:
                    print(line)
    
    # 返回是否找到差异
    return len(differences_found) > 0

if __name__ == "__main__":
    has_diff = main()
    exit(0 if not has_diff else 1)

