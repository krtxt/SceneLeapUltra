#!/usr/bin/env python3
"""
基准测试：PointNet2 vs PointNeXt vs PTv3 vs PTv3Sparse vs PointTransformer（真实点云数据，CUDA）

- 数据来源：datasets/objectcentric_grasp_dataset.py + config/data_cfg/objectcentric.yaml
- 场景输入：仅 xyz 坐标（不使用 RGB / object_mask）
- 点数：8192
- 目标特征维度：512
"""

import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset
from models.backbone import build_backbone


def _resolve_objectcentric_placeholders(cfg):
    """解析 objectcentric.yaml 中的占位符，提供安全默认值。"""
    # 允许写入
    OmegaConf.set_struct(cfg, False)

    # 提供默认占位符
    try:
        if "${target_num_grasps}" in str(cfg):
            cfg.target_num_grasps = 8
        if "${batch_size}" in str(cfg):
            cfg.batch_size = 2
        if "${exhaustive_sampling_strategy}" in str(cfg):
            cfg.exhaustive_sampling_strategy = "sequential"
    except Exception:
        pass

    # 解析
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def load_objectcentric_dataloader(cfg_path: str, limit_samples: int = 8, batch_size: int = 2) -> DataLoader:
    cfg = OmegaConf.load(cfg_path)
    cfg = _resolve_objectcentric_placeholders(cfg)

    test_cfg = cfg.test
    dataset = ObjectCentricGraspDataset(
        succ_grasp_dir=test_cfg.succ_grasp_dir,
        obj_root_dir=test_cfg.obj_root_dir,
        num_grasps=test_cfg.num_grasps,
        max_points=test_cfg.max_points,
        max_grasps_per_object=test_cfg.get('max_grasps_per_object', None),
        mesh_scale=test_cfg.mesh_scale,
        grasp_sampling_strategy=test_cfg.grasp_sampling_strategy,
        use_exhaustive_sampling=test_cfg.use_exhaustive_sampling,
        exhaustive_sampling_strategy=test_cfg.exhaustive_sampling_strategy,
        object_sampling_ratio=test_cfg.object_sampling_ratio,
        table_size=test_cfg.table_size,
    )

    subset_indices = list(range(min(limit_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 避免多进程带来的不确定性
        collate_fn=ObjectCentricGraspDataset.collate_fn,
    )
    return loader


def build_pointnet2_for_xyz_only(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    # xyz-only：第一层特征通道设为 0（无附加特征），xyz 由 use_xyz=True 处理
    try:
        mlp_list = list(cfg.layer1.mlp_list)
        mlp_list[0] = 0
        cfg.layer1.mlp_list = mlp_list
    except Exception:
        pass
    cfg.use_xyz = True
    cfg.normalize_xyz = True
    OmegaConf.set_struct(cfg, True)
    model = build_backbone(cfg)
    return model


def build_pointnext_for_xyz_only(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    # xyz-only 且 use_xyz=True 时，PointNeXt 的 input_feature_dim 必须等于 3
    cfg.input_feature_dim = 3
    cfg.use_xyz = True
    cfg.normalize_xyz = True
    # 统一输出维度
    cfg.out_dim = 512
    OmegaConf.set_struct(cfg, True)
    model = build_backbone(cfg)
    return model


def build_ptv3_for_xyz_only(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    # xyz-only：设置为仅使用xyz坐标
    cfg.input_feature_dim = 1  # 使用1作为占位符，backbone内部会处理
    cfg.use_xyz = True
    cfg.normalize_xyz = True
    # 确保输出维度为512
    cfg.out_dim = 512
    OmegaConf.set_struct(cfg, True)
    model = build_backbone(cfg)
    return model


def build_ptv3_sparse_for_xyz_only(cfg_path: str, token_strategy: str = 'fps'):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    # xyz-only：设置为仅使用xyz坐标
    cfg.input_feature_dim = 1  # 使用1作为占位符
    cfg.use_xyz = True
    cfg.normalize_xyz = True
    # 确保输出维度为256（与配置文件一致）
    cfg.out_dim = 256
    # 设置目标token数量
    cfg.target_num_tokens = 128
    # 设置token策略
    cfg.token_strategy = token_strategy
    OmegaConf.set_struct(cfg, True)
    model = build_backbone(cfg)
    return model


def build_ptv3_sparse_fps(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'fps')


def build_ptv3_sparse_last_layer(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'last_layer')


def build_ptv3_sparse_grid(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'grid')


def build_ptv3_sparse_learned(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'learned')


def build_ptv3_sparse_multiscale(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'multiscale')


def build_ptv3_sparse_surface_aware(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'surface_aware')


def build_ptv3_sparse_hybrid(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'hybrid')


def build_ptv3_sparse_surface_hybrid(cfg_path: str):
    return build_ptv3_sparse_for_xyz_only(cfg_path, 'surface_hybrid')


def build_point_transformer_for_xyz_only(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    # xyz-only：设置输入通道为3（仅xyz）
    cfg.c = 3
    cfg.use_xyz = True
    cfg.normalize_xyz = True
    # 确保输出维度为512
    cfg.out_dim = 512
    OmegaConf.set_struct(cfg, True)
    model = build_backbone(cfg)
    return model


@torch.no_grad()
def benchmark_model(model, dataloader: DataLoader, device: torch.device, max_batches: int = 5, warmup: int = 2):
    model.to(device)
    model.eval()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 预热
    seen = 0
    for batch in dataloader:
        points = batch["scene_pc"]  # (B, N, 6) or (B, N, 3)
        points = points[..., :3].contiguous()  # 仅 xyz
        points = points.to(device)
        try:
            _ = model(points)
        except Exception:
            traceback.print_exc()
            raise
        seen += 1
        if seen >= warmup:
            break

    # 正式计时
    times_ms = []
    n_batches = 0
    first_shapes = None

    for batch in dataloader:
        points = batch["scene_pc"]
        points = points[..., :3].contiguous()
        points = points.to(device)

        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        xyz_out, feat_out = model(points)
        end.record()
        torch.cuda.synchronize(device)
        elapsed = start.elapsed_time(end)  # ms
        times_ms.append(elapsed)

        if first_shapes is None:
            first_shapes = (tuple(points.shape), tuple(xyz_out.shape), tuple(feat_out.shape))

        n_batches += 1
        if n_batches >= max_batches:
            break

    avg_ms = sum(times_ms) / max(1, len(times_ms))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_time_ms_per_batch": avg_ms,
        "num_batches": len(times_ms),
        "first_shapes": first_shapes,
        "times_ms": times_ms,
    }


def humanize(n):
    return f"{n:,}"


def main():
    print("=" * 80)
    print("PointNet2 vs PointNeXt vs PTv3 vs PTv3Sparse vs PointTransformer 基准测试（仅 xyz，真实数据，CUDA）")
    print("=" * 80)

    # 设备
    if not torch.cuda.is_available():
        print("✗ 未检测到 CUDA，请在有 GPU 的环境下运行。")
        return
    device = torch.device("cuda:0")
    print(f"✓ 使用设备: {torch.cuda.get_device_name(device)}")

    # 数据
    data_cfg_path = os.path.join(PROJECT_ROOT, "config", "data_cfg", "objectcentric.yaml")
    print(f"\n加载数据配置: {data_cfg_path}")
    try:
        dataloader = load_objectcentric_dataloader(data_cfg_path, limit_samples=10, batch_size=2)
        print(f"✓ DataLoader 就绪: {len(dataloader.dataset)} 样本 (batch_size=2)")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        traceback.print_exc()
        return

    # 构建与测试模型
    results = {}

    # PointNet2 (xyz-only)
    pn2_cfg_path = os.path.join(PROJECT_ROOT, "config", "model", "diffuser", "decoder", "backbone", "pointnet2.yaml")
    print(f"\n--- 测试 PointNet2 ---\n配置: {pn2_cfg_path}")
    try:
        pn2 = build_pointnet2_for_xyz_only(pn2_cfg_path)
        res_pn2 = benchmark_model(pn2, dataloader, device, max_batches=5, warmup=2)
        results["pointnet2"] = res_pn2
        print(f"  参数量: {humanize(res_pn2['total_params'])} (可训练: {humanize(res_pn2['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_pn2['first_shapes']}")
        print(f"  平均推理: {res_pn2['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_pn2['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PointNet2 测试失败: {e}")
        traceback.print_exc()

    # PointNeXt (xyz-only)
    pnx_cfg_path = os.path.join(PROJECT_ROOT, "config", "model", "diffuser", "decoder", "backbone", "pointnext.yaml")
    print(f"\n--- 测试 PointNeXt ---\n配置: {pnx_cfg_path}")
    try:
        pnx = build_pointnext_for_xyz_only(pnx_cfg_path)
        res_pnx = benchmark_model(pnx, dataloader, device, max_batches=5, warmup=2)
        results["pointnext"] = res_pnx
        print(f"  参数量: {humanize(res_pnx['total_params'])} (可训练: {humanize(res_pnx['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_pnx['first_shapes']}")
        print(f"  平均推理: {res_pnx['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_pnx['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PointNeXt 测试失败: {e}")
        print("  可能需要先安装 OpenPoints 依赖：cd third_party/openpoints && pip install -e .")
        traceback.print_exc()

    # PTv3 (xyz-only)
    ptv3_cfg_path = os.path.join(PROJECT_ROOT, "config", "model", "diffuser", "decoder", "backbone", "ptv3_light.yaml")
    print(f"\n--- 测试 PTv3 ---\n配置: {ptv3_cfg_path}")
    try:
        ptv3 = build_ptv3_for_xyz_only(ptv3_cfg_path)
        res_ptv3 = benchmark_model(ptv3, dataloader, device, max_batches=5, warmup=2)
        results["ptv3"] = res_ptv3
        print(f"  参数量: {humanize(res_ptv3['total_params'])} (可训练: {humanize(res_ptv3['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3['first_shapes']}")
        print(f"  平均推理: {res_ptv3['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse (xyz-only) - 五种策略测试
    ptv3s_cfg_path = os.path.join(PROJECT_ROOT, "config", "model", "diffuser", "decoder", "backbone", "ptv3_sparse.yaml")
    
    # PTv3Sparse - FPS策略
    print(f"\n--- 测试 PTv3Sparse (FPS策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_fps = build_ptv3_sparse_fps(ptv3s_cfg_path)
        res_ptv3s_fps = benchmark_model(ptv3s_fps, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_fps"] = res_ptv3s_fps
        print(f"  参数量: {humanize(res_ptv3s_fps['total_params'])} (可训练: {humanize(res_ptv3s_fps['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_fps['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_fps['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_fps['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (FPS) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Last Layer策略
    print(f"\n--- 测试 PTv3Sparse (Last Layer策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_last = build_ptv3_sparse_last_layer(ptv3s_cfg_path)
        res_ptv3s_last = benchmark_model(ptv3s_last, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_last_layer"] = res_ptv3s_last
        print(f"  参数量: {humanize(res_ptv3s_last['total_params'])} (可训练: {humanize(res_ptv3s_last['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_last['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_last['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_last['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Last Layer) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Grid策略
    print(f"\n--- 测试 PTv3Sparse (Grid策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_grid = build_ptv3_sparse_grid(ptv3s_cfg_path)
        res_ptv3s_grid = benchmark_model(ptv3s_grid, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_grid"] = res_ptv3s_grid
        print(f"  参数量: {humanize(res_ptv3s_grid['total_params'])} (可训练: {humanize(res_ptv3s_grid['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_grid['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_grid['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_grid['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Grid) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Learned策略
    print(f"\n--- 测试 PTv3Sparse (Learned策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_learned = build_ptv3_sparse_learned(ptv3s_cfg_path)
        res_ptv3s_learned = benchmark_model(ptv3s_learned, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_learned"] = res_ptv3s_learned
        print(f"  参数量: {humanize(res_ptv3s_learned['total_params'])} (可训练: {humanize(res_ptv3s_learned['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_learned['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_learned['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_learned['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Learned) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Multiscale策略
    print(f"\n--- 测试 PTv3Sparse (Multiscale策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_multiscale = build_ptv3_sparse_multiscale(ptv3s_cfg_path)
        res_ptv3s_multiscale = benchmark_model(ptv3s_multiscale, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_multiscale"] = res_ptv3s_multiscale
        print(f"  参数量: {humanize(res_ptv3s_multiscale['total_params'])} (可训练: {humanize(res_ptv3s_multiscale['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_multiscale['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_multiscale['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_multiscale['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Multiscale) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Surface-aware策略
    print(f"\n--- 测试 PTv3Sparse (Surface-Aware策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_surface = build_ptv3_sparse_surface_aware(ptv3s_cfg_path)
        res_ptv3s_surface = benchmark_model(ptv3s_surface, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_surface_aware"] = res_ptv3s_surface
        print(f"  参数量: {humanize(res_ptv3s_surface['total_params'])} (可训练: {humanize(res_ptv3s_surface['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_surface['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_surface['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_surface['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Surface-Aware) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Hybrid策略
    print(f"\n--- 测试 PTv3Sparse (Hybrid策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_hybrid = build_ptv3_sparse_hybrid(ptv3s_cfg_path)
        res_ptv3s_hybrid = benchmark_model(ptv3s_hybrid, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_hybrid"] = res_ptv3s_hybrid
        print(f"  参数量: {humanize(res_ptv3s_hybrid['total_params'])} (可训练: {humanize(res_ptv3s_hybrid['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_hybrid['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_hybrid['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_hybrid['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Hybrid) 测试失败: {e}")
        traceback.print_exc()

    # PTv3Sparse - Surface-Hybrid策略
    print(f"\n--- 测试 PTv3Sparse (Surface-Hybrid策略) ---\n配置: {ptv3s_cfg_path}")
    try:
        ptv3s_surface_hybrid = build_ptv3_sparse_surface_hybrid(ptv3s_cfg_path)
        res_ptv3s_surface_hybrid = benchmark_model(ptv3s_surface_hybrid, dataloader, device, max_batches=5, warmup=2)
        results["ptv3sparse_surface_hybrid"] = res_ptv3s_surface_hybrid
        print(f"  参数量: {humanize(res_ptv3s_surface_hybrid['total_params'])} (可训练: {humanize(res_ptv3s_surface_hybrid['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_ptv3s_surface_hybrid['first_shapes']}")
        print(f"  平均推理: {res_ptv3s_surface_hybrid['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_ptv3s_surface_hybrid['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PTv3Sparse (Surface-Hybrid) 测试失败: {e}")
        traceback.print_exc()

    # PointTransformer (xyz-only)
    pt_cfg_path = os.path.join(PROJECT_ROOT, "config", "model", "diffuser", "decoder", "backbone", "point_transformer.yaml")
    print(f"\n--- 测试 PointTransformer ---\n配置: {pt_cfg_path}")
    try:
        pt = build_point_transformer_for_xyz_only(pt_cfg_path)
        res_pt = benchmark_model(pt, dataloader, device, max_batches=5, warmup=2)
        results["pointtransformer"] = res_pt
        print(f"  参数量: {humanize(res_pt['total_params'])} (可训练: {humanize(res_pt['trainable_params'])})")
        print(f"  形状(首批): 输入/xyz/features = {res_pt['first_shapes']}")
        print(f"  平均推理: {res_pt['avg_time_ms_per_batch']:.2f} ms/批 (共 {res_pt['num_batches']} 批)")
    except Exception as e:
        print(f"  ✗ PointTransformer 测试失败: {e}")
        traceback.print_exc()

    # 总结
    print("\n" + "=" * 80)
    print("结果总结：平均推理时间 (ms/批)")
    for name in (
        "pointnet2",
        "pointnext",
        "ptv3",
        "ptv3sparse_fps",
        "ptv3sparse_last_layer",
        "ptv3sparse_grid",
        "ptv3sparse_learned",
        "ptv3sparse_multiscale",
        "ptv3sparse_surface_aware",
        "ptv3sparse_hybrid",
        "ptv3sparse_surface_hybrid",
        "pointtransformer",
    ):
        if name in results:
            print(f"  {name:25s}: {results[name]['avg_time_ms_per_batch']:.2f} ms/批")
    print("=" * 80)


if __name__ == "__main__":
    main()
