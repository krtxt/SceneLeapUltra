#!/usr/bin/env python3
"""
预测抓取姿态与真实抓取姿态对比可视化脚本

主要功能：
1. 加载预训练的DDPMLightning模型
2. 对数据集中的场景进行抓取预测
3. 将预测的抓取姿态与真实标注的抓取姿态进行匹配
4. 在同一个3D场景中同时显示预测和真实的抓取姿态
5. 计算并输出定量差距指标（位置误差、旋转误差等）
6. 提供统计分析结果

可视化组件：
- 红色点云：目标物体点云
- 灰色点云：背景点云  
- 绿色mesh：目标物体mesh
- 蓝色系mesh：预测的抓取姿态
- 红色系mesh：真实的抓取姿态
- RGB坐标轴：世界坐标系参考
"""

import os
import sys
import torch
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging
from omegaconf import OmegaConf

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
from models.diffuser_lightning import DDPMLightning
from utils.hand_model import HandModel, HandModelType
from utils.hand_helper import process_hand_pose_test
# from models.backbone.pointnet2 import farthest_point_sample

def load_pretrained_model(checkpoint_path: str, config_path: Optional[str] = None) -> DDPMLightning:
    """
    加载预训练的DDPMLightning模型（修复版本，正确处理文本编码器）

    Args:
        checkpoint_path: 模型checkpoint文件路径
        config_path: 配置文件路径，如果为None则尝试从checkpoint目录推断

    Returns:
        DDPMLightning: 加载的模型实例
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")

    # 尝试加载配置文件
    if config_path is None:
        # 从checkpoint路径推断配置文件位置
        exp_dir = checkpoint_path.parent.parent
        config_path = exp_dir / "config" / "whole_config.yaml"

        if not config_path.exists():
            # 尝试其他可能的配置文件位置
            config_path = exp_dir / ".hydra" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"无法找到配置文件，请手动指定config_path")

    # 加载配置
    cfg = OmegaConf.load(config_path)

    # 检查配置结构，确保包含model部分
    if 'model' not in cfg:
        raise ValueError(f"配置文件缺少 'model' 部分: {config_path}")

    # 提取模型配置
    model_cfg = cfg.model

    # 创建模型实例
    model = DDPMLightning(model_cfg)
    
    # 【关键修复】：在加载checkpoint前强制初始化text_encoder
    text_encoder_initialized = False
    if hasattr(model.eps_model, '_ensure_text_encoder'):
        try:
            print("🔧 正在修复文本编码器初始化...")
            model.eps_model._ensure_text_encoder()
            text_encoder_initialized = True
            print("✅ 文本编码器初始化完成")
            
            # 验证初始化结果
            if model.eps_model.text_encoder is not None:
                print(f"  - Text encoder类型: {type(model.eps_model.text_encoder).__name__}")
            else:
                print("❌ 文本编码器初始化失败")
                text_encoder_initialized = False
        except Exception as e:
            print(f"❌ 文本编码器初始化失败: {e}")
            text_encoder_initialized = False

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取模型和checkpoint的state_dict
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint['state_dict']

    print(f"\n=== 权重加载分析 ===")
    print(f"当前模型权重数量: {len(model_state_dict)}")
    print(f"Checkpoint权重数量: {len(checkpoint_state_dict)}")

    # 分析当前模型的权重结构
    model_modules = {}
    for key in model_state_dict.keys():
        module_name = key.split('.')[0] if '.' in key else 'root'
        if module_name not in model_modules:
            model_modules[module_name] = 0
        model_modules[module_name] += 1

    print(f"当前模型模块结构:")
    for module, count in sorted(model_modules.items()):
        print(f"  - {module}: {count} 个权重")

    # 分析checkpoint的权重结构
    checkpoint_modules = {}
    for key in checkpoint_state_dict.keys():
        module_name = key.split('.')[0] if '.' in key else 'root'
        if module_name not in checkpoint_modules:
            checkpoint_modules[module_name] = 0
        checkpoint_modules[module_name] += 1

    print(f"Checkpoint模块结构:")
    for module, count in sorted(checkpoint_modules.items()):
        print(f"  - {module}: {count} 个权重")
    
    # 【新增】文本编码器权重详细分析
    print(f"\n=== 文本编码器权重分析 ===")
    text_encoder_keys_model = [k for k in model_state_dict.keys() if 'text_encoder' in k]
    text_encoder_keys_checkpoint = [k for k in checkpoint_state_dict.keys() if 'text_encoder' in k]
    
    print(f"模型中文本编码器权重: {len(text_encoder_keys_model)}")
    print(f"Checkpoint中文本编码器权重: {len(text_encoder_keys_checkpoint)}")
    
    if text_encoder_initialized:
        if len(text_encoder_keys_model) > 0 and len(text_encoder_keys_checkpoint) > 0:
            print("✅ 文本编码器权重匹配检查:")
            matched_text_weights = 0
            for key in text_encoder_keys_model[:5]:  # 检查前5个权重
                if key in checkpoint_state_dict:
                    model_shape = model_state_dict[key].shape
                    checkpoint_shape = checkpoint_state_dict[key].shape
                    if model_shape == checkpoint_shape:
                        print(f"  ✅ {key}: {model_shape}")
                        matched_text_weights += 1
                    else:
                        print(f"  ❌ {key}: 形状不匹配 {model_shape} vs {checkpoint_shape}")
                else:
                    print(f"  ❌ {key}: checkpoint中缺失")
            
            if matched_text_weights == len(text_encoder_keys_model[:5]):
                print(f"✅ 文本编码器权重形状匹配正常")
            else:
                print(f"⚠️  部分文本编码器权重形状不匹配")
        else:
            if len(text_encoder_keys_model) == 0:
                print("❌ 模型中没有文本编码器权重（可能初始化失败）")
            if len(text_encoder_keys_checkpoint) == 0:
                print("❌ Checkpoint中没有文本编码器权重")
    else:
        print("⚠️  文本编码器未初始化，跳过权重分析")

    # 只加载匹配的权重
    matched_state_dict = {}
    unmatched_keys = []

    for key in model_state_dict.keys():
        if key in checkpoint_state_dict:
            if model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                matched_state_dict[key] = checkpoint_state_dict[key]
            else:
                print(f"警告: 权重形状不匹配 {key}: 模型 {model_state_dict[key].shape} vs checkpoint {checkpoint_state_dict[key].shape}")
                unmatched_keys.append(key)
        else:
            print(f"警告: checkpoint中缺少权重 {key}")
            unmatched_keys.append(key)

    # 检查checkpoint中多余的权重
    extra_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())
    if extra_keys:
        print(f"信息: checkpoint包含 {len(extra_keys)} 个额外权重（可能来自更复杂的模型版本）")

        # 详细分析额外权重
        print("\n=== 额外权重详细分析 ===")

        # 按模块分组分析额外权重
        extra_by_module = {}
        for key in extra_keys:
            module_name = key.split('.')[0] if '.' in key else 'root'
            if module_name not in extra_by_module:
                extra_by_module[module_name] = []
            extra_by_module[module_name].append(key)

        for module, keys in extra_by_module.items():
            print(f"模块 '{module}': {len(keys)} 个额外权重")
            for key in sorted(keys)[:5]:  # 只显示前5个
                if key in checkpoint_state_dict:
                    shape = checkpoint_state_dict[key].shape
                    print(f"  - {key}: {shape}")
            if len(keys) > 5:
                print(f"  ... 还有 {len(keys) - 5} 个权重")

        # 分析可能的原因
        print("\n=== 可能的原因分析 ===")
        score_related = [k for k in extra_keys if 'score' in k.lower()]
        text_related = [k for k in extra_keys if any(word in k.lower() for word in ['text', 'clip', 'bert', 'transformer'])]
        attention_related = [k for k in extra_keys if any(word in k.lower() for word in ['attention', 'attn', 'self_attn'])]

        if score_related:
            print(f"- 包含 {len(score_related)} 个评分相关权重（可能是评分模块）")
        if text_related:
            print(f"- 包含 {len(text_related)} 个文本相关权重（可能是文本编码器）")
        if attention_related:
            print(f"- 包含 {len(attention_related)} 个注意力相关权重（可能是更复杂的注意力机制）")

        print("=== 分析完成 ===\n")

    # 尝试严格加载权重
    loading_success = False
    try:
        if text_encoder_initialized and len(text_encoder_keys_model) > 0 and len(text_encoder_keys_checkpoint) > 0:
            # 如果文本编码器正确初始化且权重匹配，尝试严格加载
            model.load_state_dict(checkpoint_state_dict, strict=True)
            print("✅ 权重严格加载成功 (strict=True)")
            loading_success = True
        else:
            # 否则使用匹配加载
            model.load_state_dict(matched_state_dict, strict=False)
            print(f"⚠️  权重非严格加载 (加载了 {len(matched_state_dict)}/{len(model_state_dict)} 个权重)")
    except Exception as e:
        print(f"❌ 严格加载失败: {e}")
        # 回退到匹配加载
        model.load_state_dict(matched_state_dict, strict=False)
        print(f"⚠️  回退到非严格加载 (加载了 {len(matched_state_dict)}/{len(model_state_dict)} 个权重)")
    
    model.eval()

    print(f"✓ 成功加载预训练模型: {checkpoint_path}")
    print(f"✓ 使用配置文件: {config_path}")
    print(f"✓ 模型配置: {model_cfg.get('name', 'DDPMLightning')}")
    
    # 【新增】文本编码器状态验证
    print(f"\n=== 文本编码器状态验证 ===")
    if text_encoder_initialized and hasattr(model.eps_model, 'text_encoder') and model.eps_model.text_encoder is not None:
        text_encoder = model.eps_model.text_encoder
        print(f"✅ 文本编码器状态:")
        print(f"  - 设备: {text_encoder.device}")
        print(f"  - 训练模式: {text_encoder.training}")
        
        # 检查关键权重是否已加载
        if hasattr(text_encoder, 'text_encoder') and hasattr(text_encoder.text_encoder, 'clip_model'):
            clip_model = text_encoder.text_encoder.clip_model
            if hasattr(clip_model, 'text_projection'):
                proj_weight = clip_model.text_projection
                weight_std = proj_weight.std().item()
                weight_mean = proj_weight.mean().item()
                print(f"  - 文本投影权重统计: 均值={weight_mean:.4f}, 标准差={weight_std:.4f}")
                
                # 判断权重是否合理（训练过的权重通常有特定的分布特征）
                if 0.01 < weight_std < 1.0 and abs(weight_mean) < 0.5:
                    print(f"  ✅ 文本编码器权重看起来已正确加载")
                else:
                    print(f"  ⚠️  文本编码器权重可能未正确加载（统计异常）")
            else:
                print(f"  ❌ 无法访问文本投影权重")
        else:
            print(f"  ❌ 无法访问CLIP模型组件")
    else:
        print(f"❌ 文本编码器未正确初始化或加载")
    
    if unmatched_keys:
        print(f"\n⚠️ {len(unmatched_keys)} 个权重未能加载，模型可能需要重新训练")
    
    if loading_success:
        print(f"\n🎉 模型加载完全成功！文本编码器应该可以正常工作。")
    else:
        print(f"\n⚠️  模型加载部分成功，建议验证文本功能是否正常。")

    return model

def predict_grasps(model: DDPMLightning, batch: Dict, device: str = 'cuda', num_grasps: int = 8) -> torch.Tensor:
    """
    使用模型预测抓取姿态
    
    Args:
        model: 预训练的DDPMLightning模型
        batch: 数据批次
        device: 计算设备
        num_grasps: 预测的抓取数量
    
    Returns:
        torch.Tensor: 预测的抓取姿态 [B, num_grasps, pose_dim]
    """
    model = model.to(device)
    
    # 将batch数据移动到设备
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    with torch.no_grad():
        # 使用forward_get_pose_matched方法获取预测姿态
        matched_preds, matched_targets, outputs, targets = model.forward_get_pose_matched(batch, k=num_grasps)
        
        # 打印结构信息用于调试
        print_matched_preds_structure(matched_preds, matched_targets)
        
        # 检查matched_preds的类型
        if isinstance(matched_preds, dict):
            # 如果matched_preds是字典，尝试提取手部姿态
            if 'hand_model_pose' in matched_preds:
                pred_poses = matched_preds['hand_model_pose']
            elif 'pred_pose_norm' in matched_preds:
                pred_poses = matched_preds['pred_pose_norm']
            else:
                # 尝试其他可能的键
                pose_keys = [k for k in matched_preds.keys() if 'pose' in k.lower() or 'hand' in k.lower()]
                if pose_keys:
                    pred_poses = matched_preds[pose_keys[0]]
                else:
                    raise ValueError(f"无法从matched_preds字典中找到姿态数据。可用的键: {list(matched_preds.keys())}")
        else:
            # 如果matched_preds是张量，直接使用
            pred_poses = matched_preds
        
        # 确保pred_poses是正确的形状
        if pred_poses.dim() == 2:
            # 如果是 [B*num_grasps, pose_dim]，重塑为 [B, num_grasps, pose_dim]
            B = batch['scene_pc'].shape[0]
            pred_poses = pred_poses.view(B, num_grasps, -1)
        elif pred_poses.dim() == 3:
            # 如果已经是 [B, num_grasps, pose_dim]，直接使用
            pass
        else:
            raise ValueError(f"预测姿态的形状不正确: {pred_poses.shape}")
        
    return pred_poses

def predict_grasps_with_details(model: DDPMLightning, batch: Dict, device: str = 'cuda', num_grasps: int = 8) -> Dict:
    """
    使用模型预测抓取姿态，并返回详细信息
    
    Args:
        model: 预训练的DDPMLightning模型
        batch: 数据批次
        device: 计算设备
        num_grasps: 预测的抓取数量
    
    Returns:
        Dict: 包含预测结果的详细信息
    """
    model = model.to(device)
    
    # 将batch数据移动到设备
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    with torch.no_grad():
        # 使用forward_get_pose_matched方法获取预测姿态
        matched_preds, matched_targets, outputs, targets = model.forward_get_pose_matched(batch, k=num_grasps)
        
        # 打印结构信息用于调试
        print_matched_preds_structure(matched_preds, matched_targets)
        
        # 检查matched_preds的类型并提取手部姿态
        if isinstance(matched_preds, dict):
            # 如果matched_preds是字典，尝试提取手部姿态
            if 'hand_model_pose' in matched_preds:
                pred_poses = matched_preds['hand_model_pose']
            elif 'pred_pose_norm' in matched_preds:
                pred_poses = matched_preds['pred_pose_norm']
            else:
                # 尝试其他可能的键
                pose_keys = [k for k in matched_preds.keys() if 'pose' in k.lower() or 'hand' in k.lower()]
                if pose_keys:
                    pred_poses = matched_preds[pose_keys[0]]
                else:
                    raise ValueError(f"无法从matched_preds字典中找到姿态数据。可用的键: {list(matched_preds.keys())}")
        else:
            # 如果matched_preds是张量，直接使用
            pred_poses = matched_preds
        
        # 确保pred_poses是正确的形状
        if pred_poses.dim() == 2:
            # 如果是 [B*num_grasps, pose_dim]，重塑为 [B, num_grasps, pose_dim]
            B = batch['scene_pc'].shape[0]
            pred_poses = pred_poses.view(B, num_grasps, -1)
        elif pred_poses.dim() == 3:
            # 如果已经是 [B, num_grasps, pose_dim]，直接使用
            pass
        else:
            raise ValueError(f"预测姿态的形状不正确: {pred_poses.shape}")
        
        # 返回详细信息
        result = {
            'pred_poses': pred_poses,
            'matched_preds': matched_preds,
            'matched_targets': matched_targets,
            'outputs': outputs,
            'targets': targets
        }
        
    return result

def print_forward_get_pose_matched_details(outputs: Dict, targets: Dict, batch_size: int, num_grasps: int):
    """
    打印forward_get_pose_matched输出的详细信息
    
    Args:
        outputs: 模型输出字典
        targets: 目标字典
        batch_size: 批次大小
        num_grasps: 抓取数量
    """
    print(f"\n=== forward_get_pose_matched 输出详细信息 ===")
    print(f"批次大小: {batch_size}, 抓取数量: {num_grasps}")
    print(f"总样本数: {batch_size * num_grasps}")
    
    if 'hand' in outputs:
        print(f"\noutputs['hand'] 包含以下数据项:")
        for key, value in outputs['hand'].items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: 形状 {value.shape}, 设备 {value.device}")
            else:
                print(f"  - {key}: {type(value)}")
    
    if 'hand' in targets:
        print(f"\ntargets['hand'] 包含以下数据项:")
        for key, value in targets['hand'].items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: 形状 {value.shape}, 设备 {value.device}")
            else:
                print(f"  - {key}: {type(value)}")
    
    # 检查其他可能的键
    other_output_keys = [k for k in outputs.keys() if k != 'hand']
    if other_output_keys:
        print(f"\noutputs 中的其他键: {other_output_keys}")
        for key in other_output_keys:
            if isinstance(outputs[key], torch.Tensor):
                print(f"  - {key}: 形状 {outputs[key].shape}, 设备 {outputs[key].device}")
            else:
                print(f"  - {key}: {type(outputs[key])}")
    
    other_target_keys = [k for k in targets.keys() if k != 'hand']
    if other_target_keys:
        print(f"\ntargets 中的其他键: {other_target_keys}")
        for key in other_target_keys:
            if isinstance(targets[key], torch.Tensor):
                print(f"  - {key}: 形状 {targets[key].shape}, 设备 {targets[key].device}")
            else:
                print(f"  - {key}: {type(targets[key])}")
    
    print("=" * 50)

def print_matched_preds_structure(matched_preds, matched_targets):
    """
    打印matched_preds和matched_targets的结构信息
    
    Args:
        matched_preds: 预测结果
        matched_targets: 目标结果
    """
    print(f"\n=== matched_preds 和 matched_targets 结构分析 ===")
    
    print(f"matched_preds 类型: {type(matched_preds)}")
    if isinstance(matched_preds, dict):
        print(f"matched_preds 键: {list(matched_preds.keys())}")
        for key, value in matched_preds.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: 形状 {value.shape}, 设备 {value.device}")
            else:
                print(f"  - {key}: {type(value)}")
    elif isinstance(matched_preds, torch.Tensor):
        print(f"matched_preds 形状: {matched_preds.shape}, 设备: {matched_preds.device}")
    else:
        print(f"matched_preds: {matched_preds}")
    
    print(f"\nmatched_targets 类型: {type(matched_targets)}")
    if isinstance(matched_targets, dict):
        print(f"matched_targets 键: {list(matched_targets.keys())}")
        for key, value in matched_targets.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: 形状 {value.shape}, 设备 {value.device}")
            else:
                print(f"  - {key}: {type(value)}")
    elif isinstance(matched_targets, torch.Tensor):
        print(f"matched_targets 形状: {matched_targets.shape}, 设备: {matched_targets.device}")
    else:
        print(f"matched_targets: {matched_targets}")
    
    print("=" * 50)

def create_hand_meshes_from_outputs(outputs: Dict, targets: Dict, batch_size: int, num_grasps: int,
                                  max_grasps: int = 3) -> Tuple[List, List]:
    """
    从forward_get_pose_matched的输出创建手部mesh
    
    Args:
        outputs: 模型输出字典
        targets: 目标字典
        batch_size: 批次大小
        num_grasps: 抓取数量
        max_grasps: 最大显示抓取数量
    
    Returns:
        Tuple[List, List]: (预测mesh列表, 真实mesh列表)
    """
    pred_meshes = []
    gt_meshes = []
    
    if 'hand' not in outputs or 'hand' not in targets:
        print("警告: outputs或targets中缺少'hand'键")
        return pred_meshes, gt_meshes
    
    outputs_hand = outputs['hand']
    targets_hand = targets['hand']
    
    # 检查是否有vertices和faces数据
    if 'vertices' not in outputs_hand or 'faces' not in outputs_hand:
        print("警告: outputs['hand']中缺少vertices或faces数据")
        return pred_meshes, gt_meshes
    
    if 'vertices' not in targets_hand or 'faces' not in targets_hand:
        print("警告: targets['hand']中缺少vertices或faces数据")
        return pred_meshes, gt_meshes
    
    # 获取vertices和faces数据
    pred_vertices = outputs_hand['vertices']  # [B*num_grasps, num_vertices, 3]
    pred_faces = outputs_hand['faces']        # [num_faces, 3]
    gt_vertices = targets_hand['vertices']    # [B*num_grasps, num_vertices, 3]
    gt_faces = targets_hand['faces']          # [num_faces, 3]
    
    print(f"预测vertices形状: {pred_vertices.shape}")
    print(f"预测faces形状: {pred_faces.shape}")
    print(f"真实vertices形状: {gt_vertices.shape}")
    print(f"真实faces形状: {gt_faces.shape}")
    
    # 预测姿态颜色 (蓝色系)
    pred_colors = [
        [0.0, 0.0, 1.0],  # 蓝色
        [0.0, 0.5, 1.0],  # 浅蓝色
        [0.0, 1.0, 1.0],  # 青色
        [0.5, 0.0, 1.0],  # 紫蓝色
        [0.0, 0.0, 0.5],  # 深蓝色
    ]
    
    # 真实姿态颜色 (红色系)
    gt_colors = [
        [1.0, 0.0, 0.0],  # 红色
        [1.0, 0.5, 0.0],  # 橙色
        [1.0, 0.0, 0.5],  # 粉红色
        [0.8, 0.0, 0.0],  # 深红色
        [1.0, 0.2, 0.2],  # 浅红色
    ]
    
    display_grasps = min(num_grasps, max_grasps)
    
    for b in range(batch_size):
        for g in range(display_grasps):
            try:
                # 计算在展平数组中的索引
                idx = b * num_grasps + g
                
                # 创建预测mesh
                pred_verts = pred_vertices[idx].detach().cpu().numpy()
                pred_faces_np = pred_faces.detach().cpu().numpy()
                
                pred_mesh = o3d.geometry.TriangleMesh()
                pred_mesh.vertices = o3d.utility.Vector3dVector(pred_verts)
                pred_mesh.triangles = o3d.utility.Vector3iVector(pred_faces_np.astype(np.int32))
                pred_mesh.paint_uniform_color(pred_colors[g % len(pred_colors)])
                pred_mesh.compute_vertex_normals()
                pred_meshes.append(pred_mesh)
                
                # 创建真实mesh
                gt_verts = gt_vertices[idx].detach().cpu().numpy()
                gt_faces_np = gt_faces.detach().cpu().numpy()
                
                gt_mesh = o3d.geometry.TriangleMesh()
                gt_mesh.vertices = o3d.utility.Vector3dVector(gt_verts)
                gt_mesh.triangles = o3d.utility.Vector3iVector(gt_faces_np.astype(np.int32))
                gt_mesh.paint_uniform_color(gt_colors[g % len(gt_colors)])
                gt_mesh.compute_vertex_normals()
                gt_meshes.append(gt_mesh)
                
            except Exception as e:
                print(f"创建第{b}批次第{g}个抓取的mesh失败: {e}")
                continue
    
    return pred_meshes, gt_meshes

def calculate_pose_errors(pred_poses: torch.Tensor, gt_poses: torch.Tensor, 
                         rot_type: str = 'r6d') -> Dict[str, float]:
    """
    计算预测姿态与真实姿态之间的误差
    
    Args:
        pred_poses: 预测姿态 [B, num_grasps, 23] 或 [B, num_grasps, 25]
        gt_poses: 真实姿态 [B, num_grasps, 23] 
        rot_type: 旋转表示类型
    
    Returns:
        Dict[str, float]: 误差指标字典
    """
    # 确保两个张量在同一设备上
    device = pred_poses.device
    gt_poses = gt_poses.to(device)

    # 为处理不同长度的姿态，我们只比较共同的部分
    common_dim = min(pred_poses.shape[-1], gt_poses.shape[-1])
    
    # 提取位置、关节角度
    pred_trans = pred_poses[..., :3]
    pred_qpos = pred_poses[..., 3:19]
    
    gt_trans = gt_poses[..., :3]
    gt_qpos = gt_poses[..., 3:19] 
    
    # 计算位置误差 (欧几里得距离)
    trans_error = torch.norm(pred_trans - gt_trans, dim=-1)
    
    # 计算关节角度误差 (MSE)
    qpos_error = torch.mean((pred_qpos - gt_qpos) ** 2, dim=-1)
    
    # 计算旋转误差 (仅当维度匹配时)
    rot_error = torch.tensor(0.0, device=pred_poses.device) # 默认值
    if pred_poses.shape[-1] == gt_poses.shape[-1]:
        pred_rot = pred_poses[..., 19:]
        gt_rot = gt_poses[..., 19:]
        if rot_type == 'r6d':
            # 对于6D旋转表示，使用MSE
            rot_error = torch.mean((pred_rot - gt_rot) ** 2, dim=-1)
        elif rot_type == 'quat':
            # 对于四元数，使用角度误差
            dot_product = torch.sum(pred_rot * gt_rot, dim=-1)
            rot_error = 1.0 - torch.abs(dot_product)
        else:
            rot_error = torch.mean((pred_rot - gt_rot) ** 2, dim=-1)
    else:
        print(f"警告: 预测姿态和真实姿态的旋转维度不匹配 ({pred_poses.shape[-1]} vs {gt_poses.shape[-1]})，跳过旋转误差计算。")

    # 计算统计指标
    errors = {
        'translation_mean': float(torch.mean(trans_error)),
        'translation_std': float(torch.std(trans_error)),
        'translation_max': float(torch.max(trans_error)),
        'translation_min': float(torch.min(trans_error)),
        
        'qpos_mean': float(torch.mean(qpos_error)),
        'qpos_std': float(torch.std(qpos_error)),
        'qpos_max': float(torch.max(qpos_error)),
        'qpos_min': float(torch.min(qpos_error)),
        
        'rotation_mean': float(torch.mean(rot_error)),
        'rotation_std': float(torch.std(rot_error)),
        'rotation_max': float(torch.max(rot_error)),
        'rotation_min': float(torch.min(rot_error)),
    }
    
    return errors

def create_hand_meshes_comparison(pred_poses: torch.Tensor, gt_poses: torch.Tensor,
                                hand_model: HandModel, max_grasps: int = 3) -> Tuple[List, List]:
    """
    创建预测和真实抓取姿态的手部mesh用于对比
    
    Args:
        pred_poses: 预测姿态 [B, num_grasps, 23]
        gt_poses: 真实姿态 [B, num_grasps, 23]
        hand_model: 手部模型实例
        max_grasps: 最大显示抓取数量
    
    Returns:
        Tuple[List, List]: (预测mesh列表, 真实mesh列表)
    """
    pred_meshes = []
    gt_meshes = []
    
    # 确保输入是3D张量
    if pred_poses.dim() == 2:
        pred_poses = pred_poses.unsqueeze(0)
    if gt_poses.dim() == 2:
        gt_poses = gt_poses.unsqueeze(0)
    
    B, num_grasps, _ = pred_poses.shape
    display_grasps = min(num_grasps, max_grasps)
    
    # 预测姿态颜色 (蓝色系)
    pred_colors = [
        [0.0, 0.0, 1.0],  # 蓝色
        [0.0, 0.5, 1.0],  # 浅蓝色
        [0.0, 1.0, 1.0],  # 青色
        [0.5, 0.0, 1.0],  # 紫蓝色
        [0.0, 0.0, 0.5],  # 深蓝色
    ]
    
    # 真实姿态颜色 (红色系)
    gt_colors = [
        [1.0, 0.0, 0.0],  # 红色
        [1.0, 0.5, 0.0],  # 橙色
        [1.0, 0.0, 0.5],  # 粉红色
        [0.8, 0.0, 0.0],  # 深红色
        [1.0, 0.2, 0.2],  # 浅红色
    ]
    
    for b in range(B):
        for g in range(display_grasps):
            try:
                # 创建预测姿态mesh
                hand_model.set_parameters(pred_poses[b, g])
                pred_trimesh = hand_model.get_trimesh_data(0)
                
                pred_mesh = o3d.geometry.TriangleMesh()
                pred_mesh.vertices = o3d.utility.Vector3dVector(pred_trimesh.vertices)
                pred_mesh.triangles = o3d.utility.Vector3iVector(pred_trimesh.faces)
                pred_mesh.paint_uniform_color(pred_colors[g % len(pred_colors)])
                pred_mesh.compute_vertex_normals()
                pred_meshes.append(pred_mesh)
                
                # 创建真实姿态mesh
                hand_model.set_parameters(gt_poses[b, g])
                gt_trimesh = hand_model.get_trimesh_data(0)
                
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

def create_point_cloud_from_sample(scene_pc: torch.Tensor, object_mask: Optional[torch.Tensor] = None) -> o3d.geometry.PointCloud:
    """从样本数据创建Open3D点云"""
    if isinstance(scene_pc, torch.Tensor):
        scene_pc_np = scene_pc.detach().cpu().numpy()
    else:
        scene_pc_np = scene_pc
    
    assert scene_pc_np.shape[1] == 6, f"期望6维点云 (xyz+rgb)，实际得到 {scene_pc_np.shape[1]} 维"
    
    xyz = scene_pc_np[:, :3]
    rgb = scene_pc_np[:, 3:6]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def create_highlighted_point_cloud(scene_pc: torch.Tensor, object_mask: torch.Tensor) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """创建高亮显示目标物体的点云"""
    scene_pc_np = scene_pc.detach().cpu().numpy()
    object_mask_np = object_mask.detach().cpu().numpy()
    
    if len(object_mask_np) != len(scene_pc_np):
        print(f"警告: object_mask大小 ({len(object_mask_np)}) 与点云大小 ({len(scene_pc_np)}) 不匹配")
        return create_point_cloud_from_sample(scene_pc), None
    
    background_points = scene_pc_np[~object_mask_np]
    object_points = scene_pc_np[object_mask_np]
    
    # 创建背景点云
    background_pcd = o3d.geometry.PointCloud()
    if len(background_points) > 0:
        background_pcd.points = o3d.utility.Vector3dVector(background_points[:, :3])
        background_colors = background_points[:, 3:6] * 0.5
        background_pcd.colors = o3d.utility.Vector3dVector(background_colors)
    
    # 创建目标物体点云
    object_pcd = o3d.geometry.PointCloud()
    if len(object_points) > 0:
        object_pcd.points = o3d.utility.Vector3dVector(object_points[:, :3])
        object_colors = np.ones((len(object_points), 3)) * [1.0, 0.0, 0.0]
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
    
    return background_pcd, object_pcd

def create_object_mesh(obj_verts: torch.Tensor, obj_faces: torch.Tensor,
                      color: Tuple[float, float, float] = (0.0, 1.0, 0.0)) -> Optional[o3d.geometry.TriangleMesh]:
    """从顶点和面数据创建目标物体mesh"""
    try:
        if obj_verts.numel() == 0 or obj_faces.numel() == 0:
            return None

        vertices_np = obj_verts.detach().cpu().numpy()
        faces_np = obj_faces.detach().cpu().numpy()

        object_mesh = o3d.geometry.TriangleMesh()
        object_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        object_mesh.triangles = o3d.utility.Vector3iVector(faces_np.astype(np.int32))
        object_mesh.paint_uniform_color(color)
        object_mesh.compute_vertex_normals()

        return object_mesh

    except Exception as e:
        print(f"创建物体mesh失败: {e}")
        return None

def create_coordinate_frame(size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """创建坐标轴参考框架"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def visualize_prediction_vs_ground_truth(dataset: SceneLeapPlusDataset, model: DDPMLightning,
                                       sample_idx: int = 0, max_grasps: int = 3,
                                       device: str = 'cuda'):
    """
    可视化预测抓取姿态与真实抓取姿态的对比

    Args:
        dataset: SceneLeapPlusDataset实例
        model: 预训练的DDPMLightning模型
        sample_idx: 样本索引
        max_grasps: 最大显示抓取数量
        device: 计算设备
    """
    print(f"正在可视化样本 {sample_idx} 的预测与真实抓取对比...")

    # 获取样本数据
    try:
        sample = dataset[sample_idx]
    except Exception as e:
        print(f"获取样本失败: {e}")
        return

    # 打印样本信息
    print(f"样本信息:")
    print(f"  - 场景ID: {sample['scene_id']}")
    print(f"  - 视角索引: {sample['depth_view_index']}")
    print(f"  - 物体代码: {sample['obj_code']}")
    print(f"  - 正面提示词: '{sample['positive_prompt']}'")
    print(f"  - 点云形状: {sample['scene_pc'].shape}")
    print(f"  - 真实抓取形状: {sample['hand_model_pose'].shape}")

    # 准备批次数据进行预测
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)  # 添加批次维度
        else:
            batch[key] = [value]

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

    # 降维并移动到设备
    if pred_poses.dim() == 4:
        pred_poses = pred_poses.squeeze(1) # 从 [B, 1, G, D] -> [B, G, D]

    # 获取真实抓取姿态并移动到设备
    gt_poses = sample['hand_model_pose'].unsqueeze(0).to(device)  # 添加批次维度并移动到设备

    # 计算误差指标
    print("正在计算误差指标...")
    errors = calculate_pose_errors(pred_poses, gt_poses, rot_type=model.rot_type)

    # 打印误差统计
    print(f"\n误差统计结果:")
    print(f"位置误差 (米):")
    print(f"  - 平均: {errors['translation_mean']:.4f}")
    print(f"  - 标准差: {errors['translation_std']:.4f}")
    print(f"  - 最大: {errors['translation_max']:.4f}")
    print(f"  - 最小: {errors['translation_min']:.4f}")

    print(f"关节角度误差 (MSE):")
    print(f"  - 平均: {errors['qpos_mean']:.4f}")
    print(f"  - 标准差: {errors['qpos_std']:.4f}")
    print(f"  - 最大: {errors['qpos_max']:.4f}")
    print(f"  - 最小: {errors['qpos_min']:.4f}")

    print(f"旋转误差:")
    print(f"  - 平均: {errors['rotation_mean']:.4f}")
    print(f"  - 标准差: {errors['rotation_std']:.4f}")
    print(f"  - 最大: {errors['rotation_max']:.4f}")
    print(f"  - 最小: {errors['rotation_min']:.4f}")

    # 创建可视化对象列表
    vis_objects = []

    # 添加坐标轴
    coordinate_frame = create_coordinate_frame(size=0.1)
    vis_objects.append(coordinate_frame)

    # 创建点云
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

    # 创建目标物体mesh
    try:
        obj_verts = sample['obj_verts']
        obj_faces = sample['obj_faces']
        object_mesh = create_object_mesh(obj_verts, obj_faces, color=(0.0, 0.8, 0.0))
        if object_mesh is not None:
            vis_objects.append(object_mesh)
            print("✓ 目标物体mesh创建成功")
    except Exception as e:
        print(f"目标物体mesh创建失败: {e}")

    # 创建手部mesh - 尝试从outputs和targets创建
    try:
        print("正在从模型输出创建手部mesh...")
        
        # 首先尝试从outputs和targets创建mesh
        pred_meshes, gt_meshes = create_hand_meshes_from_outputs(
            outputs, targets, batch_size, num_grasps, max_grasps
        )
        
        if pred_meshes and gt_meshes:
            vis_objects.extend(pred_meshes)
            vis_objects.extend(gt_meshes)
            print(f"✓ 从模型输出创建了 {len(pred_meshes)} 个预测mesh和 {len(gt_meshes)} 个真实mesh")
        else:
            # 回退到使用HandModel创建mesh
            print("回退到使用HandModel创建mesh...")
            hand_model = HandModel(hand_model_type=HandModelType.LEAP, device='cpu')

            # 创建预测和真实的手部mesh
            pred_meshes, gt_meshes = create_hand_meshes_comparison(
                pred_poses.cpu(), gt_poses.cpu(), hand_model, max_grasps
            )

            if pred_meshes:
                vis_objects.extend(pred_meshes)
                print(f"✓ {len(pred_meshes)} 个预测手部mesh创建成功")

            if gt_meshes:
                vis_objects.extend(gt_meshes)
                print(f"✓ {len(gt_meshes)} 个真实手部mesh创建成功")

    except Exception as e:
        print(f"手部mesh创建失败: {e}")

    # 打印可视化组件总结
    print(f"\n可视化组件总结:")
    print(f"  - 坐标轴: ✓")
    print(f"  - 点云: ✓ ({len(scene_pc)} 个点)")
    print(f"  - 目标物体mesh: ✓")
    print(f"  - 预测手部mesh: ✓ ({len(pred_meshes) if 'pred_meshes' in locals() else 0} 个)")
    print(f"  - 真实手部mesh: ✓ ({len(gt_meshes) if 'gt_meshes' in locals() else 0} 个)")

    # 创建可视化窗口
    print("\n正在启动Open3D可视化...")
    print("可视化说明:")
    print("  - 红色点: 目标物体点云")
    print("  - 灰色点: 背景点云")
    print("  - 绿色mesh: 目标物体mesh")
    print("  - 蓝色系mesh: 预测的抓取姿态")
    print("  - 红色系mesh: 真实的抓取姿态")
    print("  - RGB坐标轴: 世界坐标系")
    print("\n操作提示:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角")
    print("  - 鼠标滚轮: 缩放")

    o3d.visualization.draw_geometries(
        vis_objects,
        window_name=f"预测 vs 真实抓取对比 - Sample {sample_idx} - {sample['obj_code']}",
        width=1400,
        height=900,
        left=50,
        top=50
    )

def batch_analyze_predictions(dataset: SceneLeapPlusDataset, model: DDPMLightning,
                            num_samples: int = 10, device: str = 'cuda') -> Dict[str, float]:
    """
    批量分析多个样本的预测误差

    Args:
        dataset: SceneLeapPlusDataset实例
        model: 预训练的DDPMLightning模型
        num_samples: 分析的样本数量
        device: 计算设备

    Returns:
        Dict[str, float]: 聚合的误差统计
    """
    print(f"正在批量分析 {num_samples} 个样本的预测误差...")

    all_errors = {
        'translation_errors': [],
        'qpos_errors': [],
        'rotation_errors': []
    }

    for i in range(min(num_samples, len(dataset))):
        try:
            print(f"处理样本 {i+1}/{num_samples}...")

            # 获取样本数据
            sample = dataset[i]

            # 准备批次数据
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = [value]

            # 进行预测
            num_grasps = sample['hand_model_pose'].shape[0] if 'hand_model_pose' in sample else 8
            prediction_result = predict_grasps_with_details(model, batch, device, num_grasps=num_grasps)
            pred_poses = prediction_result['pred_poses']
            # 注意：forward_get_pose_matched返回的形状已经是 [B, num_grasps, pose_dim]，不需要额外的squeeze操作

            gt_poses = sample['hand_model_pose'].unsqueeze(0).to(device)

            # 计算误差
            errors = calculate_pose_errors(pred_poses, gt_poses, rot_type=model.rot_type)

            # 收集误差数据
            all_errors['translation_errors'].append(errors['translation_mean'])
            all_errors['qpos_errors'].append(errors['qpos_mean'])
            all_errors['rotation_errors'].append(errors['rotation_mean'])

        except Exception as e:
            print(f"处理样本 {i} 失败: {e}")
            continue

    # 计算聚合统计
    aggregated_stats = {}
    for error_type, error_list in all_errors.items():
        if error_list:
            aggregated_stats[f'{error_type}_mean'] = float(np.mean(error_list))
            aggregated_stats[f'{error_type}_std'] = float(np.std(error_list))
            aggregated_stats[f'{error_type}_max'] = float(np.max(error_list))
            aggregated_stats[f'{error_type}_min'] = float(np.min(error_list))

    # 打印聚合结果
    print(f"\n批量分析结果 (基于 {len(all_errors['translation_errors'])} 个有效样本):")
    print(f"位置误差统计:")
    print(f"  - 平均: {aggregated_stats.get('translation_errors_mean', 0):.4f} ± {aggregated_stats.get('translation_errors_std', 0):.4f}")
    print(f"  - 范围: [{aggregated_stats.get('translation_errors_min', 0):.4f}, {aggregated_stats.get('translation_errors_max', 0):.4f}]")

    print(f"关节角度误差统计:")
    print(f"  - 平均: {aggregated_stats.get('qpos_errors_mean', 0):.4f} ± {aggregated_stats.get('qpos_errors_std', 0):.4f}")
    print(f"  - 范围: [{aggregated_stats.get('qpos_errors_min', 0):.4f}, {aggregated_stats.get('qpos_errors_max', 0):.4f}]")

    print(f"旋转误差统计:")
    print(f"  - 平均: {aggregated_stats.get('rotation_errors_mean', 0):.4f} ± {aggregated_stats.get('rotation_errors_std', 0):.4f}")
    print(f"  - 范围: [{aggregated_stats.get('rotation_errors_min', 0):.4f}, {aggregated_stats.get('rotation_errors_max', 0):.4f}]")

    return aggregated_stats

def main():
    """主函数"""
    print("=" * 80)
    print("预测抓取姿态与真实抓取姿态对比可视化")
    print("=" * 80)

    # 数据路径配置 (使用真实路径)
    root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/520_0_sub_3"
    succ_grasp_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect"
    obj_root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models"

    # 模型配置
    checkpoint_path = "path/to/your/checkpoint.ckpt"  # 需要用户指定实际的checkpoint路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 可视化参数
    sample_idx = 0
    max_grasps_to_show = 3
    mode = "camera_centric_scene_mean_normalized"

    try:
        # 检查checkpoint路径
        if not os.path.exists(checkpoint_path):
            print(f"错误: checkpoint文件不存在: {checkpoint_path}")
            print("请修改脚本中的checkpoint_path变量为实际的模型文件路径")
            return

        # 初始化数据集
        print(f"正在初始化数据集 (模式: {mode})...")
        dataset = SceneLeapPlusDataset(
            root_dir=root_dir,
            succ_grasp_dir=succ_grasp_dir,
            obj_root_dir=obj_root_dir,
            num_grasps=8,
            mode=mode,
            max_grasps_per_object=2,  # 使用较小的值加快测试
            mesh_scale=0.1,
            num_neg_prompts=4,
            enable_cropping=True,
            max_points=20000,
            grasp_sampling_strategy="random"
        )

        print(f"✓ 数据集初始化成功，包含 {len(dataset)} 个样本")

        if len(dataset) == 0:
            print("数据集为空，请检查数据路径")
            return

        # 加载预训练模型
        print(f"正在加载预训练模型...")
        model = load_pretrained_model(checkpoint_path)
        model = model.to(device)

        # 单样本可视化
        print(f"\n{'='*60}")
        print(f"单样本可视化分析")
        print(f"{'='*60}")

        visualize_prediction_vs_ground_truth(
            dataset, model, sample_idx, max_grasps_to_show, device
        )

        # 批量分析 (可选)
        print(f"\n{'='*60}")
        print(f"批量误差分析")
        print(f"{'='*60}")

        batch_stats = batch_analyze_predictions(dataset, model, num_samples=5, device=device)

        print(f"\n{'='*60}")
        print(f"分析完成！")
        print(f"{'='*60}")

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
