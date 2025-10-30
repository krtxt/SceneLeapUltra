"""
可视化时间门控曲线
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.decoder.time_gating import CosineSquaredGate, MLPGate


def visualize_cosine_squared_gate():
    """可视化余弦平方门控曲线"""
    print("\n=== 可视化余弦平方门控 ===")
    
    # 创建不同缩放的门控
    scales = [1.0, 0.8, 0.5]
    t_values = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 不同缩放因子
    plt.subplot(1, 2, 1)
    for scale in scales:
        gate = CosineSquaredGate(scale=scale)
        alpha_values = []
        for t in t_values:
            alpha = gate(torch.tensor([t]))
            alpha_values.append(alpha.item())
        
        plt.plot(t_values, alpha_values, label=f'scale={scale}', linewidth=2)
    
    plt.xlabel('时间 t (归一化)', fontsize=12)
    plt.ylabel('门控因子 α(t)', fontsize=12)
    plt.title('余弦平方门控 - 不同缩放因子', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    
    # 添加关键点标注
    key_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    gate_1 = CosineSquaredGate(scale=1.0)
    for t in key_points:
        alpha = gate_1(torch.tensor([t])).item()
        plt.plot(t, alpha, 'ro', markersize=8)
        plt.annotate(f't={t:.2f}\nα={alpha:.3f}', 
                    xy=(t, alpha), xytext=(10, 10),
                    textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 子图2: 对比不同阶段
    plt.subplot(1, 2, 2)
    gate = CosineSquaredGate(scale=1.0)
    alpha_values = []
    for t in t_values:
        alpha = gate(torch.tensor([t]))
        alpha_values.append(alpha.item())
    
    plt.plot(t_values, alpha_values, 'b-', linewidth=2, label='门控曲线')
    
    # 标注不同阶段
    plt.axvspan(0, 0.25, alpha=0.2, color='green', label='早期阶段 (强约束)')
    plt.axvspan(0.25, 0.75, alpha=0.2, color='yellow', label='中期阶段 (渐弱)')
    plt.axvspan(0.75, 1.0, alpha=0.2, color='red', label='后期阶段 (弱约束)')
    
    plt.xlabel('时间 t (归一化)', fontsize=12)
    plt.ylabel('门控因子 α(t)', fontsize=12)
    plt.title('余弦平方门控 - 扩散阶段划分', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    output_path = 'docs/cosine_squared_gate.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存至: {output_path}")
    plt.close()


def compare_gate_types():
    """对比不同门控类型"""
    print("\n=== 对比不同门控类型 ===")
    
    t_values = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(12, 5))
    
    # 余弦平方门控
    cos_gate = CosineSquaredGate(scale=1.0)
    cos_alpha = []
    for t in t_values:
        alpha = cos_gate(torch.tensor([t]))
        cos_alpha.append(alpha.item())
    
    # 线性衰减（用于对比）
    linear_alpha = 1.0 - t_values
    
    # 指数衰减（用于对比）
    exp_alpha = np.exp(-3 * t_values)  # e^(-3t)
    
    # 子图1: 对比曲线
    plt.subplot(1, 2, 1)
    plt.plot(t_values, cos_alpha, 'b-', linewidth=2, label='余弦平方 (cos²)')
    plt.plot(t_values, linear_alpha, 'g--', linewidth=2, label='线性衰减')
    plt.plot(t_values, exp_alpha, 'r:', linewidth=2, label='指数衰减')
    
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('门控因子 α(t)', fontsize=12)
    plt.title('不同门控函数对比', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    
    # 子图2: 梯度对比（变化率）
    plt.subplot(1, 2, 2)
    
    # 计算数值梯度
    cos_grad = np.gradient(cos_alpha, t_values)
    linear_grad = np.gradient(linear_alpha, t_values)
    exp_grad = np.gradient(exp_alpha, t_values)
    
    plt.plot(t_values, np.abs(cos_grad), 'b-', linewidth=2, label='余弦平方 |dα/dt|')
    plt.plot(t_values, np.abs(linear_grad), 'g--', linewidth=2, label='线性 |dα/dt|')
    plt.plot(t_values, np.abs(exp_grad), 'r:', linewidth=2, label='指数 |dα/dt|')
    
    plt.xlabel('时间 t', fontsize=12)
    plt.ylabel('门控变化率 |dα/dt|', fontsize=12)
    plt.title('门控函数的变化率', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    
    plt.tight_layout()
    output_path = 'docs/gate_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存至: {output_path}")
    plt.close()


def visualize_scene_text_gating():
    """可视化场景和文本的不同门控强度"""
    print("\n=== 可视化场景与文本门控 ===")
    
    t_values = np.linspace(0, 1, 100)
    
    # 场景条件：全强度
    scene_gate = CosineSquaredGate(scale=1.0)
    scene_alpha = []
    for t in t_values:
        alpha = scene_gate(torch.tensor([t]))
        scene_alpha.append(alpha.item())
    
    # 文本条件：较弱强度
    text_gate = CosineSquaredGate(scale=0.6)
    text_alpha = []
    for t in t_values:
        alpha = text_gate(torch.tensor([t]))
        text_alpha.append(alpha.item())
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(t_values, scene_alpha, 'b-', linewidth=3, label='场景条件 (scale=1.0)')
    plt.plot(t_values, text_alpha, 'r--', linewidth=3, label='文本条件 (scale=0.6)')
    
    # 填充区域表示差异
    plt.fill_between(t_values, scene_alpha, text_alpha, alpha=0.2, color='purple',
                     label='场景-文本 强度差')
    
    plt.xlabel('时间 t (归一化)', fontsize=12)
    plt.ylabel('门控因子 α(t)', fontsize=12)
    plt.title('场景条件 vs 文本条件 - 门控强度对比', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    
    # 添加阶段标注
    plt.axvline(x=0.3, color='gray', linestyle=':', alpha=0.5)
    plt.axvline(x=0.7, color='gray', linestyle=':', alpha=0.5)
    plt.text(0.15, 1.05, '早期\n(强约束)', ha='center', fontsize=10)
    plt.text(0.5, 1.05, '中期\n(渐弱)', ha='center', fontsize=10)
    plt.text(0.85, 1.05, '后期\n(弱约束)', ha='center', fontsize=10)
    
    plt.tight_layout()
    output_path = 'docs/scene_text_gating.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存至: {output_path}")
    plt.close()


def visualize_attention_output_effect():
    """可视化门控对attention输出的影响"""
    print("\n=== 可视化门控效果 ===")
    
    t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # 模拟attention输出强度（假设为常数1.0）
    base_strength = 1.0
    
    # 计算门控后的强度
    gate = CosineSquaredGate(scale=1.0)
    gated_strength = []
    for t in t_values:
        alpha = gate(torch.tensor([t]))
        gated_strength.append(base_strength * alpha.item())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图1: 柱状图对比
    x_pos = np.arange(len(t_values))
    width = 0.35
    
    ax1.bar(x_pos - width/2, [base_strength] * len(t_values), 
            width, label='原始输出', alpha=0.7, color='gray')
    ax1.bar(x_pos + width/2, gated_strength, 
            width, label='门控后输出', alpha=0.7, color='blue')
    
    ax1.set_xlabel('扩散步数 t', fontsize=12)
    ax1.set_ylabel('Attention 输出强度', fontsize=12)
    ax1.set_title('门控对 Cross-Attention 输出的影响', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f't={t:.2f}' for t in t_values])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.2)
    
    # 子图2: 抑制率
    suppression_rate = (1.0 - np.array(gated_strength)) * 100
    
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    bars = ax2.barh(range(len(t_values)), suppression_rate, color=colors, alpha=0.7)
    
    ax2.set_xlabel('抑制率 (%)', fontsize=12)
    ax2.set_ylabel('扩散步数 t', fontsize=12)
    ax2.set_title('条件约束的抑制率', fontsize=14)
    ax2.set_yticks(range(len(t_values)))
    ax2.set_yticklabels([f't={t:.2f}' for t in t_values])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 110)
    
    # 添加数值标注
    for i, (bar, val) in enumerate(zip(bars, suppression_rate)):
        ax2.text(val + 2, i, f'{val:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = 'docs/gating_effect.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("开始生成时间门控可视化")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs('docs', exist_ok=True)
    
    try:
        visualize_cosine_squared_gate()
        compare_gate_types()
        visualize_scene_text_gating()
        visualize_attention_output_effect()
        
        print("\n" + "=" * 60)
        print("✓ 所有可视化图表生成完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - docs/cosine_squared_gate.png")
        print("  - docs/gate_comparison.png")
        print("  - docs/scene_text_gating.png")
        print("  - docs/gating_effect.png")
        
    except Exception as e:
        print(f"\n✗ 生成可视化时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

