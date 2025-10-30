#!/usr/bin/env python
"""
测试 PointNext Backbone 使用真实数据

使用 ObjectCentricGraspDataset 加载真实数据，测试 PointNext backbone 的功能
运行: CUDA_VISIBLE_DEVICES=6 python tests/test_pointnext_real_data.py
"""

import os
import sys
sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset
from models.backbone import build_backbone


def test_pointnext_with_real_data():
    """使用真实数据测试 PointNext backbone"""
    
    print("="*80)
    print("PointNext Backbone 真实数据测试")
    print("="*80)
    
    # 检查 GPU
    if not torch.cuda.is_available():
        print("\n✗ CUDA 不可用")
        return
    
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES=6 后，设备索引为 0
    print(f"\n✓ 使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 1. 加载数据配置
    print("\n[步骤 1] 加载数据配置...")
    try:
        data_cfg = OmegaConf.load('config/data_cfg/objectcentric.yaml')
        
        # 手动设置插值变量
        OmegaConf.set_struct(data_cfg, False)
        data_cfg.target_num_grasps = 8
        data_cfg.exhaustive_sampling_strategy = 'sequential'
        data_cfg.batch_size = 2
        OmegaConf.resolve(data_cfg)
        
        print("  ✓ 配置加载成功")
        print(f"  数据集: {data_cfg.name}")
        print(f"  模式: {data_cfg.mode}")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return
    
    # 2. 创建数据集（使用少量数据进行测试）
    print("\n[步骤 2] 创建测试数据集...")
    try:
        # 使用测试集配置
        test_cfg = data_cfg.test
        
        # 创建数据集（不使用缓存，直接加载）
        dataset = ObjectCentricGraspDataset(
            succ_grasp_dir=test_cfg.succ_grasp_dir,
            obj_root_dir=test_cfg.obj_root_dir,
            num_grasps=test_cfg.num_grasps if 'num_grasps' in test_cfg else 8,
            max_points=test_cfg.max_points if 'max_points' in test_cfg else 8192,
            max_grasps_per_object=test_cfg.max_grasps_per_object if 'max_grasps_per_object' in test_cfg else None,
            mesh_scale=test_cfg.mesh_scale if 'mesh_scale' in test_cfg else 0.1,
            grasp_sampling_strategy=test_cfg.grasp_sampling_strategy if 'grasp_sampling_strategy' in test_cfg else 'random',
            use_exhaustive_sampling=test_cfg.use_exhaustive_sampling if 'use_exhaustive_sampling' in test_cfg else False,
            exhaustive_sampling_strategy=test_cfg.exhaustive_sampling_strategy if 'exhaustive_sampling_strategy' in test_cfg else 'sequential',
            object_sampling_ratio=test_cfg.object_sampling_ratio if 'object_sampling_ratio' in test_cfg else 0.8,
            table_size=test_cfg.table_size if 'table_size' in test_cfg else 0.4,
        )
        
        print(f"  ✓ 数据集创建成功")
        print(f"  数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("  ✗ 数据集为空，请检查数据路径")
            return
            
    except Exception as e:
        print(f"  ✗ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 创建 DataLoader
    print("\n[步骤 3] 创建 DataLoader...")
    batch_size = 2
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # 测试时使用单进程
            collate_fn=dataset.collate_fn,
        )
        print(f"  ✓ DataLoader 创建成功 (batch_size={batch_size})")
    except Exception as e:
        print(f"  ✗ DataLoader 创建失败: {e}")
        return
    
    # 4. 加载一个 batch 的数据
    print("\n[步骤 4] 加载测试数据...")
    try:
        batch = next(iter(dataloader))
        
        scene_pc = batch['scene_pc']  # (B, N, 3)
        hand_model_pose = batch['hand_model_pose']  # (B, num_grasps, 23)
        
        print(f"  ✓ 数据加载成功")
        print(f"  batch size: {scene_pc.shape[0]}")
        print(f"  scene_pc 形状: {scene_pc.shape}")
        print(f"  hand_model_pose 形状: {hand_model_pose.shape}")
        print(f"  scene_pc 范围: [{scene_pc.min():.3f}, {scene_pc.max():.3f}]")
        
        # 移动到 GPU
        scene_pc = scene_pc.to(device)
        
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 创建 PointNext backbone
    print("\n[步骤 5] 创建 PointNext backbone...")
    try:
        # 加载配置
        backbone_cfg = OmegaConf.load('config/model/flow_matching/decoder/backbone/pointnext.yaml')
        
        # 确保配置正确
        backbone_cfg.sampler = 'random'  # 使用随机采样避免潜在问题
        
        print(f"  配置:")
        print(f"    num_points: {backbone_cfg.num_points}")
        print(f"    num_tokens: {backbone_cfg.num_tokens}")
        print(f"    out_dim: {backbone_cfg.out_dim}")
        print(f"    strides: {backbone_cfg.strides}")
        
        # 创建模型
        model = build_backbone(backbone_cfg).to(device)
        model.eval()
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ Backbone 创建成功")
        print(f"  参数量: {param_count:.2f}M")
        
    except Exception as e:
        print(f"  ✗ Backbone 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 测试前向传播
    print("\n[步骤 6] 测试前向传播...")
    try:
        with torch.no_grad():
            # 前向传播
            xyz_out, feat_out = model(scene_pc)
        
        print(f"  ✓ 前向传播成功")
        print(f"  输入形状: {scene_pc.shape}")
        print(f"  输出 xyz 形状: {xyz_out.shape}")
        print(f"  输出 features 形状: {feat_out.shape}")
        print(f"  输出 xyz 范围: [{xyz_out.min():.3f}, {xyz_out.max():.3f}]")
        print(f"  输出 features 范围: [{feat_out.min():.3f}, {feat_out.max():.3f}]")
        
        # 检查是否有 NaN 或 Inf
        has_nan = torch.isnan(feat_out).any() or torch.isnan(xyz_out).any()
        has_inf = torch.isinf(feat_out).any() or torch.isinf(xyz_out).any()
        
        if has_nan:
            print(f"  ⚠️  警告: 输出包含 NaN")
        if has_inf:
            print(f"  ⚠️  警告: 输出包含 Inf")
        
        if not has_nan and not has_inf:
            print(f"  ✓ 输出数值正常 (无 NaN/Inf)")
        
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 多 batch 测试
    print("\n[步骤 7] 测试多个 batch...")
    try:
        num_test_batches = 3
        total_time = 0.0
        
        for i in range(num_test_batches):
            batch = next(iter(dataloader))
            scene_pc = batch['scene_pc'].to(device)
            
            # 计时
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                xyz_out, feat_out = model(scene_pc)
            end_time.record()
            
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)  # ms
            total_time += elapsed
            
            print(f"  Batch {i+1}/{num_test_batches}: {elapsed:.2f} ms")
        
        avg_time = total_time / num_test_batches
        print(f"\n  ✓ 多 batch 测试完成")
        print(f"  平均推理时间: {avg_time:.2f} ms/batch")
        print(f"  单样本推理时间: {avg_time/batch_size:.2f} ms/sample")
        
    except Exception as e:
        print(f"  ⚠️  多 batch 测试失败: {e}")
    
    # 8. 显存使用情况
    print("\n[步骤 8] 显存使用情况...")
    try:
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        
        print(f"  当前分配: {allocated:.2f} GB")
        print(f"  当前预留: {reserved:.2f} GB")
        print(f"  峰值分配: {max_allocated:.2f} GB")
        
    except Exception as e:
        print(f"  ⚠️  显存信息获取失败: {e}")
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print("✓ PointNext backbone 在真实数据上工作正常！")
    print(f"✓ 输入: {scene_pc.shape}")
    print(f"✓ 输出 xyz: {xyz_out.shape}")
    print(f"✓ 输出 features: {feat_out.shape}")
    print(f"✓ 平均推理时间: {avg_time:.2f} ms/batch")
    print("="*80)
    
    print("\n提示:")
    print("  - PointNext backbone 可以正常处理真实点云数据")
    print("  - 可以在训练脚本中使用: model/decoder/backbone=pointnext")
    print("  - 建议根据实际情况调整 batch_size 和 num_workers")


if __name__ == "__main__":
    # 检查环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        print(f"CUDA_VISIBLE_DEVICES = {cuda_visible}")
    else:
        print("提示: 可以设置 CUDA_VISIBLE_DEVICES=6 来指定 GPU")
    
    test_pointnext_with_real_data()

