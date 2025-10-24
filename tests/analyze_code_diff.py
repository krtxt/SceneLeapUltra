#!/usr/bin/env python3
"""分析两个版本代码库的核心差异"""

import difflib
import os
from pathlib import Path
from typing import List, Tuple

# 定义路径
V2_PATH = "/home/xiantuo/source/grasp/SceneLeapUltra/experiments/mini_obj_centric_origin_diffuser_pointnet2/backups/v2"
V1_PATH = "/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra/experiments/diffuser_objcentric_mini_pointnet2_moreepochs/backups/v1"

# 定义核心文件列表
CORE_FILES = [
    "train_lightning.py",
    "train_distributed.py",
    "test_lightning.py",
    "models/diffuser_lightning.py",
    "models/decoder/dit.py",
    "models/backbone/pointnet2.py",
    "models/utils/diffusion_core.py",
    "models/utils/diffusion_utils.py",
    "models/loss/grasp_loss_pose.py",
]

def read_file_lines(filepath: str) -> List[str]:
    """读取文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"读取文件失败 {filepath}: {e}")
        return []

def compare_files(file_path: str) -> Tuple[bool, List[str]]:
    """比较两个版本的文件"""
    v2_file = os.path.join(V2_PATH, file_path)
    v1_file = os.path.join(V1_PATH, file_path)
    
    # 检查文件是否存在
    v2_exists = os.path.exists(v2_file)
    v1_exists = os.path.exists(v1_file)
    
    if not v2_exists and not v1_exists:
        return False, [f"两个版本都不存在文件: {file_path}"]
    elif not v2_exists:
        return True, [f"原始版本(v2)中不存在此文件: {file_path}"]
    elif not v1_exists:
        return True, [f"修改版本(v1)中不存在此文件: {file_path}"]
    
    # 读取文件内容
    v2_lines = read_file_lines(v2_file)
    v1_lines = read_file_lines(v1_file)
    
    # 生成diff
    diff = list(difflib.unified_diff(
        v2_lines, v1_lines,
        fromfile=f'v2/{file_path}',
        tofile=f'v1/{file_path}',
        lineterm=''
    ))
    
    has_diff = len(diff) > 0
    return has_diff, diff

def analyze_diff_summary(diff_lines: List[str]) -> dict:
    """分析diff的摘要信息"""
    adds = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
    dels = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))
    
    # 提取关键修改（函数、类定义）
    key_changes = []
    for i, line in enumerate(diff_lines):
        if line.startswith('+') or line.startswith('-'):
            content = line[1:].strip()
            if any(keyword in content for keyword in ['def ', 'class ', 'import ', '__init__']):
                key_changes.append(line)
    
    return {
        'additions': adds,
        'deletions': dels,
        'key_changes': key_changes[:20]  # 只保留前20个关键变化
    }

def main():
    print("=" * 80)
    print("代码差异分析报告")
    print("=" * 80)
    print(f"\n原始版本 (v2): {V2_PATH}")
    print(f"修改版本 (v1): {V1_PATH}\n")
    
    # 检查新增和删除的文件
    print("\n" + "=" * 80)
    print("文件结构变化")
    print("=" * 80)
    
    # 检查v1中特有的文件
    v1_only_files = [
        "models/fm_lightning.py",
        "models/fm/paths.py",
        "models/fm/solvers.py",
        "models/fm/guidance.py",
        "models/decoder/dit_fm.py",
    ]
    
    print("\n修改版本(v1)中新增的文件:")
    for f in v1_only_files:
        full_path = os.path.join(V1_PATH, f)
        if os.path.exists(full_path):
            print(f"  ✓ {f}")
    
    # 比较核心文件
    print("\n" + "=" * 80)
    print("核心文件差异分析")
    print("=" * 80)
    
    total_changes = 0
    for file_path in CORE_FILES:
        print(f"\n{'=' * 80}")
        print(f"文件: {file_path}")
        print(f"{'=' * 80}")
        
        has_diff, diff = compare_files(file_path)
        
        if not has_diff:
            print("  ✓ 无差异")
            continue
        
        if isinstance(diff, list) and len(diff) == 1 and "不存在" in diff[0]:
            print(f"  ⚠ {diff[0]}")
            continue
        
        # 分析diff
        summary = analyze_diff_summary(diff)
        print(f"\n  总体变化:")
        print(f"    - 新增行数: {summary['additions']}")
        print(f"    - 删除行数: {summary['deletions']}")
        total_changes += summary['additions'] + summary['deletions']
        
        if summary['key_changes']:
            print(f"\n  关键代码变化 (前20条):")
            for change in summary['key_changes']:
                print(f"    {change[:100]}")
        
        # 保存完整diff到文件
        diff_output_dir = Path("tests/diff_output")
        diff_output_dir.mkdir(exist_ok=True)
        
        safe_filename = file_path.replace('/', '_').replace('.py', '_diff.txt')
        diff_file = diff_output_dir / safe_filename
        
        with open(diff_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(diff))
        
        print(f"\n  完整diff已保存到: {diff_file}")
    
    print("\n" + "=" * 80)
    print(f"总结: 总共有 {total_changes} 行代码变化")
    print("=" * 80)

if __name__ == "__main__":
    main()

