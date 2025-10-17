#!/usr/bin/env python3
"""
基于配置文件的预测抓取姿态与真实抓取姿态对比可视化脚本

使用方法:
1. 修改 config_visualization.yaml 中的模型路径和数据路径
2. 运行: python visualize_with_config.py --config config_visualization.yaml
3. 或直接运行: python visualize_with_config.py (使用默认配置)

主要功能：
- 支持配置文件驱动的可视化
- 加载预训练模型进行抓取预测
- 预测与真实抓取姿态的对比可视化
- 定量误差分析和统计
- 批量样本分析
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
import argparse
import json
from typing import Optional, Dict, List
from pathlib import Path
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
from models.diffuser_lightning import DDPMLightning
from utils.hand_model import HandModel, HandModelType
from utils.hand_helper import process_hand_pose_test
from visualize_prediction_vs_ground_truth import (
    load_pretrained_model, predict_grasps, predict_grasps_with_details, calculate_pose_errors,
    create_hand_meshes_comparison, create_object_mesh,
    create_coordinate_frame, create_highlighted_point_cloud, create_point_cloud_from_sample,
    print_forward_get_pose_matched_details, print_matched_preds_structure,
)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)

def validate_config(cfg: Dict) -> bool:
    """验证配置文件的有效性"""
    required_keys = [
        'dataset.root_dir',
        'dataset.succ_grasp_dir', 
        'dataset.obj_root_dir',
        'model.checkpoint_path'
    ]
    
    for key in required_keys:
        keys = key.split('.')
        current = cfg
        try:
            for k in keys:
                current = current[k]
        except KeyError:
            print(f"错误: 配置文件缺少必需的键: {key}")
            return False
    
    # 检查路径是否存在
    paths_to_check = [
        cfg['dataset']['root_dir'],
        cfg['dataset']['succ_grasp_dir'],
        cfg['dataset']['obj_root_dir']
    ]
    
    for path in paths_to_check:
        if not os.path.exists(path):
            print(f"警告: 路径不存在: {path}")
    
    if not os.path.exists(cfg['model']['checkpoint_path']):
        print(f"错误: checkpoint文件不存在: {cfg['model']['checkpoint_path']}")
        return False
    
    return True

def create_dataset_from_config(cfg: Dict) -> SceneLeapPlusDataset:
    """根据配置创建数据集"""
    dataset_cfg = cfg['dataset']
    
    dataset = SceneLeapPlusDataset(
        root_dir=dataset_cfg['root_dir'],
        succ_grasp_dir=dataset_cfg['succ_grasp_dir'],
        obj_root_dir=dataset_cfg['obj_root_dir'],
        num_grasps=dataset_cfg['num_grasps'],
        mode=dataset_cfg['mode'],
        max_grasps_per_object=dataset_cfg['max_grasps_per_object'],
        mesh_scale=dataset_cfg['mesh_scale'],
        num_neg_prompts=dataset_cfg['num_neg_prompts'],
        enable_cropping=dataset_cfg['enable_cropping'],
        max_points=dataset_cfg['max_points'],
        grasp_sampling_strategy=dataset_cfg['grasp_sampling_strategy']
    )
    
    return dataset

def create_hand_meshes_with_config(pred_poses: torch.Tensor, gt_poses: torch.Tensor,
                                 hand_model: HandModel, cfg: Dict) -> tuple:
    """使用配置文件中的颜色创建手部mesh"""
    pred_meshes = []
    gt_meshes = []

    print(f"输入pred_poses形状: {pred_poses.shape}")
    print(f"输入gt_poses形状: {gt_poses.shape}")

    # 处理pred_poses的形状 - 可能是[1, 1, 8, 25]需要重塑为[1, 8, 25]
    if pred_poses.dim() == 4:
        # 假设是[batch, 1, num_grasps, pose_dim]，重塑为[batch, num_grasps, pose_dim]
        pred_poses = pred_poses.squeeze(1)
        print(f"重塑后pred_poses形状: {pred_poses.shape}")
    elif pred_poses.dim() == 2:
        pred_poses = pred_poses.unsqueeze(0)

    # 处理gt_poses的形状
    if gt_poses.dim() == 2:
        gt_poses = gt_poses.unsqueeze(0)

    print(f"最终pred_poses形状: {pred_poses.shape}")
    print(f"最终gt_poses形状: {gt_poses.shape}")

    B, num_grasps_pred, _ = pred_poses.shape
    _, num_grasps_gt, _ = gt_poses.shape

    max_grasps = cfg['visualization']['max_grasps_to_show']
    display_grasps = min(min(num_grasps_pred, num_grasps_gt), max_grasps)
    
    pred_colors = cfg['visualization']['colors']['prediction']
    gt_colors = cfg['visualization']['colors']['ground_truth']
    
    for b in range(B):
        for g in range(display_grasps):
            try:
                # 创建预测姿态mesh
                print(f"设置预测姿态参数，形状: {pred_poses[b, g].shape}")
                hand_model.set_parameters(pred_poses[b, g])
                pred_trimesh = hand_model.get_trimesh_data(0)
                print(f"预测trimesh顶点形状: {pred_trimesh.vertices.shape}, 面形状: {pred_trimesh.faces.shape}")

                pred_mesh = o3d.geometry.TriangleMesh()
                pred_mesh.vertices = o3d.utility.Vector3dVector(pred_trimesh.vertices)
                pred_mesh.triangles = o3d.utility.Vector3iVector(pred_trimesh.faces)
                pred_mesh.paint_uniform_color(pred_colors[g % len(pred_colors)])
                pred_mesh.compute_vertex_normals()
                pred_meshes.append(pred_mesh)

                # 创建真实姿态mesh
                print(f"设置真实姿态参数，形状: {gt_poses[b, g].shape}")
                hand_model.set_parameters(gt_poses[b, g])
                gt_trimesh = hand_model.get_trimesh_data(0)
                print(f"真实trimesh顶点形状: {gt_trimesh.vertices.shape}, 面形状: {gt_trimesh.faces.shape}")

                gt_mesh = o3d.geometry.TriangleMesh()
                gt_mesh.vertices = o3d.utility.Vector3dVector(gt_trimesh.vertices)
                gt_mesh.triangles = o3d.utility.Vector3iVector(gt_trimesh.faces)
                gt_mesh.paint_uniform_color(gt_colors[g % len(gt_colors)])
                gt_mesh.compute_vertex_normals()
                gt_meshes.append(gt_mesh)

            except Exception as e:
                print(f"创建第{b}批次第{g}个抓取的mesh失败: {e}")
                continue
    
    return pred_meshes, gt_meshes

def visualize_with_config(dataset: SceneLeapPlusDataset, model: DDPMLightning, cfg: Dict):
    """使用配置文件进行可视化"""
    sample_idx = cfg['visualization']['sample_idx']
    device = cfg['model']['device']
    
    print(f"正在可视化样本 {sample_idx}...")
    
    # 获取样本数据
    try:
        sample = dataset[sample_idx]
    except Exception as e:
        print(f"获取样本失败: {e}")
        return
    
    # 打印样本信息
    print(f"样本信息:")
    print(f"  - 场景ID: {sample['scene_id']}")
    print(f"  - 物体代码: {sample['obj_code']}")
    print(f"  - 点云形状: {sample['scene_pc'].shape}")
    print(f"  - 真实抓取形状: {sample['hand_model_pose'].shape}")
    
    # 检查文本提示是否存在
    if 'positive_prompt' in sample:
        print(f"  - 正向提示: '{sample['positive_prompt']}'")
    else:
        print(f"  - ⚠️  警告: 样本中缺少 positive_prompt")
    
    if 'negative_prompts' in sample:
        print(f"  - 负向提示: {sample['negative_prompts']}")
    else:
        print(f"  - ⚠️  警告: 样本中缺少 negative_prompts")
    
    # 准备批次数据进行预测
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = [value]
    
    # 验证批次数据中的文本提示
    print(f"\n批次数据验证:")
    print(f"  - 批次键: {list(batch.keys())}")
    if 'positive_prompt' in batch:
        print(f"  - 批次正向提示: {batch['positive_prompt']}")
    if 'negative_prompts' in batch:
        print(f"  - 批次负向提示: {batch['negative_prompts']}")
    
    # 进行抓取预测
    print("正在进行抓取预测...")
    try:
        # 从样本中获取抓取数量
        num_grasps = sample['hand_model_pose'].shape[0] if 'hand_model_pose' in sample else 8
        batch_size = 1  # 当前是单样本预测
        
        # 使用详细预测函数
        prediction_result = predict_grasps_with_details(model, batch, device, num_grasps=num_grasps)
        pred_poses = prediction_result['pred_poses']
        outputs = prediction_result['outputs']
        targets = prediction_result['targets']
        
        print(f"✓ 预测完成，预测姿态形状: {pred_poses.shape}")
        
        # 打印详细信息
        print_forward_get_pose_matched_details(outputs, targets, batch_size, num_grasps)
        
    except Exception as e:
        print(f"预测失败: {e}")
        return
    
    # 获取真实抓取姿态并进行与训练一致的 hand_helper 处理
    print(f"原始sample['hand_model_pose']形状: {sample['hand_model_pose'].shape}")
    if 'se3' in sample:
        print(f"原始sample['se3']形状: {sample['se3'].shape}")
        # 添加batch维度以匹配process_hand_pose_test的期望格式
        gt_dict = {
            'hand_model_pose': sample['hand_model_pose'].clone().unsqueeze(0),  # [8, 23] -> [1, 8, 23]
            'se3': sample['se3'].clone().unsqueeze(0),  # [8, 4, 4] -> [1, 8, 4, 4]
        }
        processed = process_hand_pose_test(gt_dict, rot_type=model.rot_type, mode=getattr(dataset, 'mode', 'camera_centric_scene_mean_normalized'))
        print(f"处理后hand_model_pose形状: {processed['hand_model_pose'].shape}")
        gt_poses = processed['hand_model_pose']  # 已经有batch维度了，不需要再添加
        print(f"最终gt_poses形状: {gt_poses.shape}")
    else:
        # 回退: 没有 se3 信息时直接使用原始数据
        gt_poses = sample['hand_model_pose'].unsqueeze(0)
        print(f"回退gt_poses形状: {gt_poses.shape}")
    
    # 计算误差指标
    if cfg['logging']['show_detailed_errors']:
        print("正在计算误差指标...")
        errors = calculate_pose_errors(pred_poses, gt_poses, rot_type=model.rot_type)
        
        print(f"\n误差统计结果:")
        print(f"位置误差 (米): {errors['translation_mean']:.4f} ± {errors['translation_std']:.4f}")
        print(f"关节角度误差: {errors['qpos_mean']:.4f} ± {errors['qpos_std']:.4f}")
        print(f"旋转误差: {errors['rotation_mean']:.4f} ± {errors['rotation_std']:.4f}")
    
    # 创建可视化对象
    vis_objects = []
    
    # 坐标轴
    frame_size = cfg['visualization']['coordinate_frame_size']
    coordinate_frame = create_coordinate_frame(size=frame_size)
    vis_objects.append(coordinate_frame)
    
    # 点云
    scene_pc = sample['scene_pc']
    object_mask = sample['object_mask']
    
    if len(object_mask) == len(scene_pc):
        background_pcd, object_pcd = create_highlighted_point_cloud(scene_pc, object_mask)
        if background_pcd is not None:
            vis_objects.append(background_pcd)
        if object_pcd is not None:
            vis_objects.append(object_pcd)
    else:
        pcd = create_point_cloud_from_sample(scene_pc)
        vis_objects.append(pcd)
    
    # 目标物体mesh
    try:
        obj_verts = sample['obj_verts']
        obj_faces = sample['obj_faces']
        object_color = cfg['visualization']['colors']['object_mesh']
        object_mesh = create_object_mesh(obj_verts, obj_faces, color=tuple(object_color))
        if object_mesh is not None:
            vis_objects.append(object_mesh)
    except Exception as e:
        print(f"目标物体mesh创建失败: {e}")
    
    # 手部mesh
    try:
        print("正在创建手部mesh...")
        print(f"模型旋转类型: {model.rot_type}")
        
        # 首先尝试从outputs和targets创建mesh
        from visualize_prediction_vs_ground_truth import create_hand_meshes_from_outputs
        pred_meshes, gt_meshes = create_hand_meshes_from_outputs(
            outputs, targets, batch_size, num_grasps, cfg['visualization']['max_grasps_to_show']
        )
        
        if pred_meshes and gt_meshes:
            vis_objects.extend(pred_meshes)
            vis_objects.extend(gt_meshes)
            print(f"✓ 从模型输出创建了 {len(pred_meshes)} 个预测mesh和 {len(gt_meshes)} 个真实mesh")
        else:
            # 回退到使用HandModel创建mesh
            print("回退到使用HandModel创建mesh...")
            hand_model = HandModel(hand_model_type=HandModelType.LEAP, device='cpu', rot_type=model.rot_type)
            
            pred_meshes, gt_meshes = create_hand_meshes_with_config(
                pred_poses.cpu(), gt_poses.cpu(), hand_model, cfg
            )
            
            vis_objects.extend(pred_meshes)
            vis_objects.extend(gt_meshes)
            
            print(f"✓ 创建了 {len(pred_meshes)} 个预测mesh和 {len(gt_meshes)} 个真实mesh")
        
    except Exception as e:
        print(f"手部模型创建失败: {e}")
    
    # 启动可视化
    print("\n正在启动可视化...")
    print("可视化说明:")
    print("  - 蓝色系mesh: 预测的抓取姿态")
    print("  - 红色系mesh: 真实的抓取姿态")
    print("  - 绿色mesh: 目标物体")
    
    window_cfg = cfg['visualization']
    o3d.visualization.draw_geometries(
        vis_objects,
        window_name=f"预测 vs 真实 - {sample['obj_code']}",
        width=window_cfg['window_width'],
        height=window_cfg['window_height'],
        left=50,
        top=50
    )

def batch_analyze_with_config(dataset: SceneLeapPlusDataset, model: DDPMLightning, cfg: Dict):
    """使用配置文件进行批量分析"""
    if not cfg['batch_analysis']['enabled']:
        print("批量分析已禁用")
        return

    num_samples = cfg['batch_analysis']['num_samples']
    device = cfg['model']['device']

    print(f"正在进行批量分析 ({num_samples} 个样本)...")

    all_errors = []

    for i in range(min(num_samples, len(dataset))):
        try:
            if cfg['logging']['show_progress']:
                print(f"处理样本 {i+1}/{num_samples}...")

            sample = dataset[i]

            # 准备批次数据
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = [value]

            # 预测
            num_grasps = sample['hand_model_pose'].shape[0] if 'hand_model_pose' in sample else 8
            prediction_result = predict_grasps_with_details(model, batch, device, num_grasps=num_grasps)
            pred_poses = prediction_result['pred_poses']

            # 获取真实抓取姿态并进行与训练一致的 hand_helper 处理
            if 'se3' in sample:
                gt_dict = {
                    'hand_model_pose': sample['hand_model_pose'].clone(),
                    'se3': sample['se3'].clone(),
                }
                processed = process_hand_pose_test(gt_dict, rot_type=model.rot_type, mode=getattr(dataset, 'mode', 'camera_centric_scene_mean_normalized'))
                gt_poses = processed['hand_model_pose'].unsqueeze(0)
            else:
                # 回退: 没有 se3 信息时直接使用原始数据
                gt_poses = sample['hand_model_pose'].unsqueeze(0)

            # 计算误差
            errors = calculate_pose_errors(pred_poses, gt_poses, rot_type=model.rot_type)
            errors['sample_idx'] = i
            errors['obj_code'] = sample['obj_code']
            errors['scene_id'] = sample['scene_id']

            all_errors.append(errors)

        except Exception as e:
            print(f"处理样本 {i} 失败: {e}")
            continue

    if not all_errors:
        print("没有成功处理的样本")
        return

    # 计算聚合统计
    translation_errors = [e['translation_mean'] for e in all_errors]
    qpos_errors = [e['qpos_mean'] for e in all_errors]
    rotation_errors = [e['rotation_mean'] for e in all_errors]

    results = {
        'num_samples': len(all_errors),
        'translation_stats': {
            'mean': float(np.mean(translation_errors)),
            'std': float(np.std(translation_errors)),
            'min': float(np.min(translation_errors)),
            'max': float(np.max(translation_errors))
        },
        'qpos_stats': {
            'mean': float(np.mean(qpos_errors)),
            'std': float(np.std(qpos_errors)),
            'min': float(np.min(qpos_errors)),
            'max': float(np.max(qpos_errors))
        },
        'rotation_stats': {
            'mean': float(np.mean(rotation_errors)),
            'std': float(np.std(rotation_errors)),
            'min': float(np.min(rotation_errors)),
            'max': float(np.max(rotation_errors))
        },
        'detailed_results': all_errors
    }

    # 打印结果
    print(f"\n批量分析结果 (基于 {len(all_errors)} 个样本):")
    print(f"位置误差: {results['translation_stats']['mean']:.4f} ± {results['translation_stats']['std']:.4f}")
    print(f"关节误差: {results['qpos_stats']['mean']:.4f} ± {results['qpos_stats']['std']:.4f}")
    print(f"旋转误差: {results['rotation_stats']['mean']:.4f} ± {results['rotation_stats']['std']:.4f}")

    # 保存结果
    if cfg['batch_analysis']['save_results']:
        results_path = cfg['batch_analysis']['results_path']
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ 结果已保存到: {results_path}")

    return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预测与真实抓取对比可视化")
    parser.add_argument('--config', type=str, default='config_visualization.yaml',
                       help='配置文件路径')
    parser.add_argument('--sample_idx', type=int, default=None,
                       help='要可视化的样本索引 (覆盖配置文件)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型checkpoint路径 (覆盖配置文件)')
    parser.add_argument('--batch_only', action='store_true',
                       help='只进行批量分析，不显示可视化')

    args = parser.parse_args()

    print("=" * 80)
    print("预测抓取姿态与真实抓取姿态对比可视化")
    print("=" * 80)

    try:
        # 加载配置
        print(f"正在加载配置文件: {args.config}")
        cfg = load_config(args.config)

        # 命令行参数覆盖
        if args.sample_idx is not None:
            cfg['visualization']['sample_idx'] = args.sample_idx
        if args.checkpoint is not None:
            cfg['model']['checkpoint_path'] = args.checkpoint

        # 验证配置
        if not validate_config(cfg):
            print("配置验证失败，请检查配置文件")
            return

        # 创建数据集
        print("正在初始化数据集...")
        dataset = create_dataset_from_config(cfg)
        print(f"✓ 数据集初始化成功，包含 {len(dataset)} 个样本")

        if len(dataset) == 0:
            print("数据集为空，请检查数据路径")
            return

        # 加载模型
        print("正在加载预训练模型...")
        model = load_pretrained_model(
            cfg['model']['checkpoint_path'],
            cfg['model'].get('config_path')
        )
        model = model.to(cfg['model']['device'])

        # 检查文本条件配置
        print("\n=== 模型配置分析 ===")
        
        # 检查模型是否有文本编码器
        has_text_encoder = hasattr(model.eps_model, 'text_encoder') if hasattr(model, 'eps_model') else False
        
        # 检查配置中的文本条件设置
        use_text_condition = False
        if hasattr(model.eps_model, 'use_text_condition'):
            use_text_condition = model.eps_model.use_text_condition
        elif hasattr(model, 'hparams') and 'decoder' in model.hparams:
            decoder_cfg = model.hparams['decoder']
            use_text_condition = decoder_cfg.get('use_text_condition', False)
        
        print(f"文本条件功能状态:")
        print(f"  - 配置中启用文本条件: {'✓' if use_text_condition else '✗'}")
        print(f"  - 模型包含文本编码器: {'✓' if has_text_encoder else '✗'}")
        
        if use_text_condition and not has_text_encoder:
            print(f"  ⚠️  警告: 配置启用了文本条件但模型中缺少文本编码器组件")
            print(f"      这可能是因为checkpoint与当前配置不匹配")
        elif not use_text_condition:
            print(f"  ℹ️  信息: 当前模型仅使用场景点云进行抓取预测")
            print(f"      如需使用文本指定目标物体，请在配置中设置 use_text_condition: true")
        else:
            print(f"  ✓ 模型支持基于文本的目标物体指定功能")
        
        print("=== 配置分析完成 ===\n")

        # 执行分析
        if not args.batch_only:
            print("\n" + "="*60)
            print("单样本可视化")
            print("="*60)
            visualize_with_config(dataset, model, cfg)

        print("\n" + "="*60)
        print("批量误差分析")
        print("="*60)
        batch_analyze_with_config(dataset, model, cfg)

        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
