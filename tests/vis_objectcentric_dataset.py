"""
ObjectCentricGraspDataset 可视化脚本

使用plotly进行交互式3D可视化，展示：
- 物体表面采样点云（OMF坐标系）
- 手部抓取位置
- 物体mesh

运行方式:
cd /home/xiantuo/source/grasp/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python scripts/vis_objectcentric_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import plotly.graph_objects as go
import torch
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset


def visualize_object_grasp(batch, batch_idx=0, show_mesh=True, show_grasp_positions=True,
                          colormap='viridis', point_size=2, grasp_size=5, save_html=None):
    """
    可视化ObjectCentricGraspDataset的数据
    
    Args:
        batch: 数据batch
        batch_idx: batch中的索引
        show_mesh: 是否显示物体mesh
        show_grasp_positions: 是否显示抓取位置
        colormap: 点云着色方案 ('viridis', 'plasma', 'height', 'uniform')
        point_size: 点云点的大小
        grasp_size: 抓取点的大小
        save_html: 保存HTML文件路径（None则显示）
    """
    fig = go.Figure()
    
    # 获取数据
    scene_pc = batch['scene_pc'][batch_idx].cpu()  # (N, 3)
    hand_positions = batch['hand_model_pose'][batch_idx, :, :3].cpu()  # (G, 3)
    obj_verts = batch['obj_verts'][batch_idx].cpu()
    obj_faces = batch['obj_faces'][batch_idx].cpu()
    obj_code = batch['obj_code'][batch_idx]
    
    # 点云着色
    if colormap == 'height':
        # 根据Z轴高度着色
        z_values = scene_pc[:, 2]
        colors = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-8)
        colors = colors.numpy()
        colorscale = 'Viridis'
    elif colormap == 'uniform':
        # 统一颜色
        colors = 'lightblue'
        colorscale = None
    else:
        # 使用指定的colormap
        distances = torch.norm(scene_pc, dim=1)
        colors = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        colors = colors.numpy()
        colorscale = colormap.capitalize()
    
    # 添加点云
    fig.add_trace(go.Scatter3d(
        x=scene_pc[:, 0],
        y=scene_pc[:, 1],
        z=scene_pc[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors,
            colorscale=colorscale,
            opacity=0.8,
            showscale=True if colorscale else False
        ),
        name='物体表面点云',
        hovertemplate='<b>点云</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>'
    ))
    
    # 添加抓取位置
    if show_grasp_positions:
        fig.add_trace(go.Scatter3d(
            x=hand_positions[:, 0],
            y=hand_positions[:, 1],
            z=hand_positions[:, 2],
            mode='markers',
            marker=dict(
                size=grasp_size,
                color='red',
                opacity=1.0,
                symbol='diamond'
            ),
            name=f'抓取位置 (n={len(hand_positions)})',
            hovertemplate='<b>抓取 %{pointNumber}</b><br>x: %{x:.4f}<br>y: %{y:.4f}<br>z: %{z:.4f}<extra></extra>'
        ))
    
    # 添加物体mesh
    if show_mesh and obj_verts.shape[0] > 0 and obj_faces.shape[0] > 0:
        fig.add_trace(go.Mesh3d(
            x=obj_verts[:, 0],
            y=obj_verts[:, 1],
            z=obj_verts[:, 2],
            i=obj_faces[:, 0],
            j=obj_faces[:, 1],
            k=obj_faces[:, 2],
            color='lightgreen',
            opacity=0.3,
            name='物体Mesh',
            hovertemplate='<b>Mesh</b><extra></extra>'
        ))
    
    # 布局设置
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (OMF)', showgrid=True),
            yaxis=dict(title='Y (OMF)', showgrid=True),
            zaxis=dict(title='Z (OMF)', showgrid=True)
        ),
        width=1000,
        height=800,
        title=f'物体: {obj_code}<br>点云数: {len(scene_pc)}, 抓取数: {len(hand_positions)}',
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    if save_html:
        fig.write_html(save_html)
        print(f"已保存到: {save_html}")
    else:
        fig.show()


def visualize_multiple_samples(batch, num_samples=4, colormap='height'):
    """
    在一个图中可视化多个样本
    
    Args:
        batch: 数据batch
        num_samples: 要可视化的样本数
        colormap: 颜色映射
    """
    num_samples = min(num_samples, len(batch['obj_code']))
    
    fig = make_subplots(
        rows=1, cols=num_samples,
        specs=[[{'type': 'scatter3d'}] * num_samples],
        subplot_titles=[batch['obj_code'][i] for i in range(num_samples)],
        horizontal_spacing=0.05
    )
    
    for idx in range(num_samples):
        col = idx + 1
        
        scene_pc = batch['scene_pc'][idx].cpu()
        hand_positions = batch['hand_model_pose'][idx, :, :3].cpu()
        
        # 点云着色
        if colormap == 'height':
            z_values = scene_pc[:, 2]
            colors = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-8)
            colors = colors.numpy()
        else:
            colors = 'lightblue'
        
        # 添加点云
        fig.add_trace(
            go.Scatter3d(
                x=scene_pc[:, 0],
                y=scene_pc[:, 1],
                z=scene_pc[:, 2],
                mode='markers',
                marker=dict(size=2, color=colors, colorscale='Viridis', opacity=0.6),
                name=f'点云_{idx}',
                showlegend=False
            ),
            row=1, col=col
        )
        
        # 添加抓取位置
        fig.add_trace(
            go.Scatter3d(
                x=hand_positions[:, 0],
                y=hand_positions[:, 1],
                z=hand_positions[:, 2],
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.8),
                name=f'抓取_{idx}',
                showlegend=False
            ),
            row=1, col=col
        )
    
    fig.update_layout(
        height=600,
        width=400 * num_samples,
        title_text="ObjectCentric数据集 - 多样本对比",
    )
    
    # 统一坐标轴
    for i in range(1, num_samples + 1):
        fig.update_scenes(aspectmode='data', row=1, col=i)
    
    fig.show()


def print_statistics(batch, batch_idx=0):
    """打印数据统计信息"""
    scene_pc = batch['scene_pc'][batch_idx].cpu()
    hand_positions = batch['hand_model_pose'][batch_idx, :, :3].cpu()
    obj_verts = batch['obj_verts'][batch_idx].cpu()
    obj_code = batch['obj_code'][batch_idx]
    
    print("=" * 80)
    print(f"样本统计 - {obj_code}")
    print("=" * 80)
    
    print(f"\n点云统计:")
    print(f"  数量: {len(scene_pc)}")
    print(f"  X范围: [{scene_pc[:, 0].min():.4f}, {scene_pc[:, 0].max():.4f}]")
    print(f"  Y范围: [{scene_pc[:, 1].min():.4f}, {scene_pc[:, 1].max():.4f}]")
    print(f"  Z范围: [{scene_pc[:, 2].min():.4f}, {scene_pc[:, 2].max():.4f}]")
    print(f"  中心: [{scene_pc[:, 0].mean():.4f}, {scene_pc[:, 1].mean():.4f}, {scene_pc[:, 2].mean():.4f}]")
    
    print(f"\n抓取统计:")
    print(f"  数量: {len(hand_positions)}")
    print(f"  X范围: [{hand_positions[:, 0].min():.4f}, {hand_positions[:, 0].max():.4f}]")
    print(f"  Y范围: [{hand_positions[:, 1].min():.4f}, {hand_positions[:, 1].max():.4f}]")
    print(f"  Z范围: [{hand_positions[:, 2].min():.4f}, {hand_positions[:, 2].max():.4f}]")
    print(f"  中心: [{hand_positions[:, 0].mean():.4f}, {hand_positions[:, 1].mean():.4f}, {hand_positions[:, 2].mean():.4f}]")
    
    # 抓取间距离
    distances = torch.cdist(hand_positions, hand_positions)
    mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
    pairwise_distances = distances[mask]
    
    print(f"\n抓取间距离:")
    print(f"  最小: {pairwise_distances.min():.4f}")
    print(f"  最大: {pairwise_distances.max():.4f}")
    print(f"  平均: {pairwise_distances.mean():.4f}")
    print(f"  中位: {pairwise_distances.median():.4f}")
    
    print(f"\n顶点统计:")
    print(f"  数量: {len(obj_verts)}")
    print(f"  面片数: {len(batch['obj_faces'][batch_idx])}")
    
    # 坐标系验证
    pc_center = scene_pc.mean(dim=0)
    verts_center = obj_verts.mean(dim=0)
    grasp_center = hand_positions.mean(dim=0)
    
    print(f"\n坐标系验证 (OMF):")
    print(f"  点云-顶点中心距离: {torch.norm(pc_center - verts_center):.4f}")
    print(f"  点云-抓取中心距离: {torch.norm(pc_center - grasp_center):.4f}")
    print(f"  顶点-抓取中心距离: {torch.norm(verts_center - grasp_center):.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='可视化ObjectCentricGraspDataset')
    parser.add_argument('--config', type=str, default='config/data_cfg/objectcentric.yaml',
                       help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_grasps', type=int, default=8,
                       help='每个样本的抓取数量')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='要可视化的样本索引')
    parser.add_argument('--show_mesh', action='store_true', default=True,
                       help='是否显示mesh')
    parser.add_argument('--colormap', type=str, default='height',
                       choices=['height', 'viridis', 'plasma', 'uniform'],
                       help='点云颜色映射')
    parser.add_argument('--multi_sample', action='store_true',
                       help='可视化多个样本对比')
    parser.add_argument('--save_html', type=str, default=None,
                       help='保存HTML文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.create(cfg)
    
    # 创建数据集
    print(f"创建数据集...")
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir=cfg.train.succ_grasp_dir,
        obj_root_dir=cfg.train.obj_root_dir,
        num_grasps=args.num_grasps,
        max_points=cfg.train.max_points,
        max_grasps_per_object=50,  # 用较少数量加快加载
        mesh_scale=cfg.train.mesh_scale,
        grasp_sampling_strategy="farthest_point",
        use_exhaustive_sampling=False
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"物体数量: {len(dataset.hand_pose_data)}")
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ObjectCentricGraspDataset.collate_fn
    )
    
    # 获取第一个batch
    batch = next(iter(dataloader))
    print(f"Batch加载成功，大小: {len(batch['obj_code'])}")
    
    # 打印统计信息
    print_statistics(batch, args.sample_idx)
    
    # 可视化
    if args.multi_sample:
        print(f"\n可视化多个样本...")
        visualize_multiple_samples(batch, num_samples=min(4, len(batch['obj_code'])), colormap=args.colormap)
    else:
        print(f"\n可视化样本 {args.sample_idx}...")
        visualize_object_grasp(
            batch,
            batch_idx=args.sample_idx,
            show_mesh=args.show_mesh,
            show_grasp_positions=True,
            colormap=args.colormap,
            point_size=2,
            grasp_size=6,
            save_html=args.save_html
        )


if __name__ == "__main__":
    main()

