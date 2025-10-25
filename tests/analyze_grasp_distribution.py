#!/usr/bin/env python3
"""
分析测试数据集中所有场景的成功抓取数分布
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # 使用非交互式后端

def collect_grasp_statistics(dataset_path):
    """
    遍历数据集，收集所有物体的 num_collision_free 数据
    
    Args:
        dataset_path: 数据集根目录路径
        
    Returns:
        统计数据字典
    """
    dataset_path = Path(dataset_path)
    
    # 存储所有数据
    all_num_collision_free = []
    scene_statistics = []
    object_details = []
    
    # 遍历所有场景文件夹
    scene_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith('scene_')])
    
    print(f"找到 {len(scene_folders)} 个场景文件夹")
    
    for scene_folder in scene_folders:
        json_file = scene_folder / "collision_free_grasp_indices.json"
        
        if not json_file.exists():
            print(f"警告: {scene_folder.name} 中没有找到 collision_free_grasp_indices.json")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scene_name = scene_folder.name
            scene_grasp_counts = []
            
            for obj in data:
                num_cf = obj['num_collision_free']
                all_num_collision_free.append(num_cf)
                scene_grasp_counts.append(num_cf)
                
                object_details.append({
                    'scene': scene_name,
                    'object_index': obj['object_index'],
                    'object_name': obj.get('object_name', 'unknown'),
                    'uid': obj.get('uid', 'unknown'),
                    'num_collision_free': num_cf
                })
            
            scene_statistics.append({
                'scene': scene_name,
                'num_objects': len(data),
                'total_grasps': sum(scene_grasp_counts),
                'avg_grasps_per_object': np.mean(scene_grasp_counts) if scene_grasp_counts else 0,
                'min_grasps': min(scene_grasp_counts) if scene_grasp_counts else 0,
                'max_grasps': max(scene_grasp_counts) if scene_grasp_counts else 0
            })
            
        except Exception as e:
            print(f"处理 {scene_folder.name} 时出错: {e}")
            continue
    
    return {
        'all_num_collision_free': all_num_collision_free,
        'scene_statistics': scene_statistics,
        'object_details': object_details
    }


def analyze_distribution(data_dict):
    """
    对收集的数据进行详细的统计分析
    """
    all_counts = np.array(data_dict['all_num_collision_free'])
    scene_stats = data_dict['scene_statistics']
    
    print("\n" + "="*80)
    print("数据集整体统计")
    print("="*80)
    
    # 基本统计
    print(f"\n总场景数: {len(scene_stats)}")
    print(f"总物体数: {len(all_counts)}")
    print(f"总成功抓取数: {np.sum(all_counts)}")
    
    print(f"\n每场景平均物体数: {np.mean([s['num_objects'] for s in scene_stats]):.2f}")
    print(f"每场景物体数范围: [{min(s['num_objects'] for s in scene_stats)}, {max(s['num_objects'] for s in scene_stats)}]")
    
    print("\n" + "-"*80)
    print("成功抓取数 (num_collision_free) 分布统计")
    print("-"*80)
    
    # 描述性统计
    print(f"\n均值 (Mean): {np.mean(all_counts):.2f}")
    print(f"中位数 (Median): {np.median(all_counts):.2f}")
    print(f"标准差 (Std): {np.std(all_counts):.2f}")
    print(f"方差 (Variance): {np.var(all_counts):.2f}")
    
    print(f"\n最小值: {np.min(all_counts)}")
    print(f"最大值: {np.max(all_counts)}")
    print(f"范围 (Range): {np.max(all_counts) - np.min(all_counts)}")
    
    # 分位数
    print(f"\n分位数统计:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(all_counts, p)
        print(f"  {p}th percentile: {value:.2f}")
    
    # 分布区间统计
    print(f"\n分布区间统计:")
    ranges = [
        (0, 100, "0-100"),
        (101, 200, "101-200"),
        (201, 300, "201-300"),
        (301, 400, "301-400"),
        (401, 500, "401-500"),
        (501, 1000, "501-1000"),
        (1001, 2000, "1001-2000"),
        (2001, float('inf'), "2001+")
    ]
    
    for min_val, max_val, label in ranges:
        count = np.sum((all_counts >= min_val) & (all_counts <= max_val))
        percentage = (count / len(all_counts)) * 100
        print(f"  {label}: {count} 个物体 ({percentage:.2f}%)")
    
    # 零抓取统计
    zero_grasps = np.sum(all_counts == 0)
    if zero_grasps > 0:
        print(f"\n无成功抓取 (num_collision_free = 0) 的物体数: {zero_grasps} ({zero_grasps/len(all_counts)*100:.2f}%)")
    
    # 找出抓取数最多和最少的物体
    print("\n" + "-"*80)
    print("极端情况分析")
    print("-"*80)
    
    object_details = data_dict['object_details']
    sorted_objects = sorted(object_details, key=lambda x: x['num_collision_free'])
    
    print(f"\n成功抓取数最少的前10个物体:")
    for i, obj in enumerate(sorted_objects[:10], 1):
        print(f"  {i}. 场景: {obj['scene']}, 物体索引: {obj['object_index']}, "
              f"名称: {obj['object_name']}, 抓取数: {obj['num_collision_free']}")
    
    print(f"\n成功抓取数最多的前10个物体:")
    for i, obj in enumerate(sorted_objects[-10:][::-1], 1):
        print(f"  {i}. 场景: {obj['scene']}, 物体索引: {obj['object_index']}, "
              f"名称: {obj['object_name']}, 抓取数: {obj['num_collision_free']}")
    
    # 场景级别统计
    print("\n" + "-"*80)
    print("场景级别统计")
    print("-"*80)
    
    total_grasps_per_scene = [s['total_grasps'] for s in scene_stats]
    avg_grasps_per_scene = [s['avg_grasps_per_object'] for s in scene_stats]
    
    print(f"\n每场景总抓取数:")
    print(f"  均值: {np.mean(total_grasps_per_scene):.2f}")
    print(f"  中位数: {np.median(total_grasps_per_scene):.2f}")
    print(f"  标准差: {np.std(total_grasps_per_scene):.2f}")
    print(f"  范围: [{np.min(total_grasps_per_scene)}, {np.max(total_grasps_per_scene)}]")
    
    print(f"\n每场景平均抓取数 (per object):")
    print(f"  均值: {np.mean(avg_grasps_per_scene):.2f}")
    print(f"  中位数: {np.median(avg_grasps_per_scene):.2f}")
    print(f"  标准差: {np.std(avg_grasps_per_scene):.2f}")
    
    # 场景排名
    sorted_scenes = sorted(scene_stats, key=lambda x: x['total_grasps'], reverse=True)
    print(f"\n总抓取数最多的前10个场景:")
    for i, scene in enumerate(sorted_scenes[:10], 1):
        print(f"  {i}. {scene['scene']}: {scene['total_grasps']} 次抓取 "
              f"({scene['num_objects']} 个物体, 平均 {scene['avg_grasps_per_object']:.2f} 次/物体)")
    
    print(f"\n总抓取数最少的前10个场景:")
    for i, scene in enumerate(sorted_scenes[-10:][::-1], 1):
        print(f"  {i}. {scene['scene']}: {scene['total_grasps']} 次抓取 "
              f"({scene['num_objects']} 个物体, 平均 {scene['avg_grasps_per_object']:.2f} 次/物体)")
    
    return all_counts, scene_stats


def create_visualizations(all_counts, scene_stats, output_dir):
    """
    创建可视化图表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 直方图 - 成功抓取数分布
    plt.figure(figsize=(12, 6))
    plt.hist(all_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Collision-Free Grasps', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Collision-Free Grasps per Object', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'grasp_distribution_histogram.png', dpi=300)
    print(f"已保存: {output_dir / 'grasp_distribution_histogram.png'}")
    plt.close()
    
    # 2. 对数尺度直方图
    plt.figure(figsize=(12, 6))
    plt.hist(all_counts, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Collision-Free Grasps', fontsize=12)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title('Distribution of Collision-Free Grasps per Object (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'grasp_distribution_histogram_log.png', dpi=300)
    print(f"已保存: {output_dir / 'grasp_distribution_histogram_log.png'}")
    plt.close()
    
    # 3. 箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_counts, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Number of Collision-Free Grasps', fontsize=12)
    plt.title('Box Plot of Collision-Free Grasps Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'grasp_distribution_boxplot.png', dpi=300)
    print(f"已保存: {output_dir / 'grasp_distribution_boxplot.png'}")
    plt.close()
    
    # 4. CDF (累积分布函数)
    plt.figure(figsize=(12, 6))
    sorted_counts = np.sort(all_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    plt.plot(sorted_counts, cdf, linewidth=2)
    plt.xlabel('Number of Collision-Free Grasps', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'grasp_distribution_cdf.png', dpi=300)
    print(f"已保存: {output_dir / 'grasp_distribution_cdf.png'}")
    plt.close()
    
    # 5. 分区间柱状图
    ranges = [
        (0, 100), (101, 200), (201, 300), (301, 400),
        (401, 500), (501, 1000), (1001, 2000), (2001, 10000)
    ]
    range_labels = ['0-100', '101-200', '201-300', '301-400',
                   '401-500', '501-1000', '1001-2000', '2001+']
    range_counts = []
    
    for min_val, max_val in ranges:
        count = np.sum((all_counts >= min_val) & (all_counts <= max_val))
        range_counts.append(count)
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range_labels, range_counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Grasp Count Range', fontsize=12)
    plt.ylabel('Number of Objects', fontsize=12)
    plt.title('Distribution of Objects by Grasp Count Range', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    
    # 在柱子上添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'grasp_distribution_ranges.png', dpi=300)
    print(f"已保存: {output_dir / 'grasp_distribution_ranges.png'}")
    plt.close()
    
    # 6. 场景级别统计
    total_grasps_per_scene = [s['total_grasps'] for s in scene_stats]
    
    plt.figure(figsize=(12, 6))
    plt.hist(total_grasps_per_scene, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Total Grasps per Scene', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Total Grasps per Scene', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scene_total_grasps_distribution.png', dpi=300)
    print(f"已保存: {output_dir / 'scene_total_grasps_distribution.png'}")
    plt.close()


def save_detailed_report(data_dict, all_counts, scene_stats, output_dir):
    """
    保存详细的分析报告到文本文件
    """
    output_dir = Path(output_dir)
    report_file = output_dir / 'grasp_distribution_analysis_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("成功抓取数分布详细分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 基本统计
        f.write("数据集整体统计\n")
        f.write("-"*80 + "\n")
        f.write(f"总场景数: {len(scene_stats)}\n")
        f.write(f"总物体数: {len(all_counts)}\n")
        f.write(f"总成功抓取数: {np.sum(all_counts)}\n")
        f.write(f"每场景平均物体数: {np.mean([s['num_objects'] for s in scene_stats]):.2f}\n\n")
        
        # 分布统计
        f.write("成功抓取数分布统计\n")
        f.write("-"*80 + "\n")
        f.write(f"均值: {np.mean(all_counts):.2f}\n")
        f.write(f"中位数: {np.median(all_counts):.2f}\n")
        f.write(f"标准差: {np.std(all_counts):.2f}\n")
        f.write(f"最小值: {np.min(all_counts)}\n")
        f.write(f"最大值: {np.max(all_counts)}\n\n")
        
        # 分位数
        f.write("分位数统计:\n")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(all_counts, p)
            f.write(f"  {p}th percentile: {value:.2f}\n")
        f.write("\n")
        
        # 所有物体详情
        f.write("所有物体详细信息\n")
        f.write("-"*80 + "\n")
        object_details = sorted(data_dict['object_details'], 
                              key=lambda x: (x['scene'], x['object_index']))
        
        for obj in object_details:
            f.write(f"场景: {obj['scene']}, 物体索引: {obj['object_index']}, "
                   f"名称: {obj['object_name']}, 抓取数: {obj['num_collision_free']}\n")
    
    print(f"\n已保存详细报告: {report_file}")


def main():
    dataset_path = "/home/xiantuo/source/grasp/SceneLeapUltra/data/test_final_test_723_2_processed"
    output_dir = "/home/xiantuo/source/grasp/SceneLeapUltra/grasp_analysis_results"
    
    print("开始收集数据...")
    data_dict = collect_grasp_statistics(dataset_path)
    
    print(f"\n成功收集 {len(data_dict['all_num_collision_free'])} 个物体的数据")
    
    print("\n开始统计分析...")
    all_counts, scene_stats = analyze_distribution(data_dict)
    
    print("\n生成可视化图表...")
    create_visualizations(all_counts, scene_stats, output_dir)
    
    print("\n保存详细报告...")
    save_detailed_report(data_dict, all_counts, scene_stats, output_dir)
    
    print("\n" + "="*80)
    print("分析完成!")
    print(f"结果已保存到: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

