"""
PTv3 五种 Token 策略在两种 grid_size 下的全面测试（随机点云）

验证内容：
- 所有策略是否能正常前向，无报错
- 输出形状是否符合期望：xyz [B, K, 3]，feat [B, K, D]（启用 tokens_last）
- 统计稀疏前的有效点数（K_full，来自 encoder 稀疏输出 densify 后非零计数）
- 对比 grid_size=0.02 与 0.003 下的 K_full（0.003 应明显更大）

使用方法：
python tests/test_ptv3_token_strategies_gridsize_compare.py --device cuda --batch 2 --points 8192 --tokens 128
"""

import argparse
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple

import torch
from omegaconf import OmegaConf

# 将项目根目录加入 sys.path 以便导入 models/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.backbone.ptv3_sparse_encoder import PTv3SparseEncoder


def build_cfg(strategy: str, grid_size: float, target_tokens: int) -> OmegaConf:
    cfg = OmegaConf.create({
        'grid_size': grid_size,
        'use_flash_attention': True,
        'encoder_channels': [32, 64, 128, 256],
        'encoder_depths': [1, 1, 2, 2],
        'encoder_num_head': [2, 4, 8, 16],
        'enc_patch_size': [1024, 1024, 1024, 1024],
        'stride': [2, 2, 2],
        'out_dim': 256,
        'input_feature_dim': 1,
        'mlp_ratio': 2,
        'target_num_tokens': target_tokens,
        'token_strategy': strategy,
        'grid_resolution': [5, 5, 5] if strategy == 'grid' else [8, 8, 8],
        'tokens_last': True,
    })
    return cfg


@torch.inference_mode()
def run_once(
    strategy: str,
    grid_size: float,
    device: torch.device,
    batch_size: int,
    num_points: int,
    target_tokens: int,
) -> Dict:
    cfg = build_cfg(strategy, grid_size, target_tokens)
    model = PTv3SparseEncoder(cfg, target_num_tokens=target_tokens, token_strategy=strategy).to(device)
    model.eval()

    # 构造随机点云（均匀分布在单位立方体）
    pos = torch.rand(batch_size, num_points, 3, device=device)

    # 前向
    t0 = time.time()
    xyz, feat = model(pos, return_full_res=True)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    elapsed_ms = (time.time() - t0) * 1000

    # 形状检查（tokens_last=True 期望 feat [B, K, D]）
    ok_shape = (
        xyz.shape[0] == batch_size
        and xyz.shape[1] == target_tokens
        and xyz.shape[2] == 3
        and feat.shape[0] == batch_size
        and feat.shape[1] == target_tokens
        and feat.shape[2] == 256
    )

    # 稀疏前的非零计数（仅对非 multiscale 策略可靠）
    k_full_list: List[int] = []
    if hasattr(model, 'debug_last') and isinstance(model.debug_last, dict):
        if 'xyz_sparse_full' in model.debug_last:
            xyz_sparse_full = model.debug_last['xyz_sparse_full']  # [B, Kmax, 3]
            # 有效点掩码：排除 padding 的 0 向量
            valid_mask = (xyz_sparse_full.abs().sum(dim=-1) > 0)
            for b in range(batch_size):
                k_full_list.append(int(valid_mask[b].sum().item()))

    return {
        'strategy': strategy,
        'grid_size': grid_size,
        'elapsed_ms': elapsed_ms,
        'xyz_shape': tuple(xyz.shape),
        'feat_shape': tuple(feat.shape),
        'ok_shape': ok_shape,
        'k_full_per_batch': k_full_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--points', type=int, default=8192)
    parser.add_argument('--tokens', type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Batch={args.batch}, Points={args.points}, Tokens={args.tokens}")

    strategies = ['last_layer', 'fps', 'grid', 'learned', 'multiscale']
    grid_sizes = [0.02, 0.003]

    results: List[Dict] = []

    for gs in grid_sizes:
        print("\n" + "=" * 80)
        print(f"Testing grid_size={gs}")
        print("=" * 80)
        for st in strategies:
            print(f"\n- Strategy: {st}")
            try:
                res = run_once(
                    strategy=st,
                    grid_size=gs,
                    device=device,
                    batch_size=args.batch,
                    num_points=args.points,
                    target_tokens=args.tokens,
                )
                results.append(res)
                print(f"  xyz: {res['xyz_shape']} | feat: {res['feat_shape']} | shape_ok: {res['ok_shape']}")
                if res['k_full_per_batch']:
                    print(f"  K_full per batch: {res['k_full_per_batch']}")
                print(f"  elapsed: {res['elapsed_ms']:.2f} ms")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # 汇总对比 grid_size 影响（仅统计有 K_full 的策略）
    print("\n" + "=" * 80)
    print("Summary (K_full by grid_size)")
    print("=" * 80)
    for st in strategies:
        r_small = [r for r in results if r['strategy'] == st and abs(r['grid_size'] - 0.003) < 1e-9]
        r_large = [r for r in results if r['strategy'] == st and abs(r['grid_size'] - 0.02) < 1e-9]
        if r_small and r_large and r_small[0]['k_full_per_batch'] and r_large[0]['k_full_per_batch']:
            small_mean = sum(r_small[0]['k_full_per_batch']) / len(r_small[0]['k_full_per_batch'])
            large_mean = sum(r_large[0]['k_full_per_batch']) / len(r_large[0]['k_full_per_batch'])
            print(f"{st:<12} | K_full@0.003 ~ {small_mean:.1f} | K_full@0.02 ~ {large_mean:.1f}")
        else:
            print(f"{st:<12} | (no K_full stats, likely multiscale or missing debug)")


if __name__ == "__main__":
    main()


