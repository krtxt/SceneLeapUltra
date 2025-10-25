"""
CLIP 文本-图像注意力测试实验

测试 CLIP 模型能否通过文本描述准确识别和关注场景中的目标物体。
使用图像块相似度热力图方法进行可视化。
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import open_clip
import torch.nn.functional as F
from PIL import Image

from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
from datasets.utils.io_utils import load_scene_images
from datasets.utils.mask_utils import extract_object_mask
from models.utils.text_encoder import TextEncoder


class CLIPAttentionVisualizer:
    """CLIP 注意力可视化器"""
    
    def __init__(self, device='cuda', patch_size=224, stride=112):
        """
        初始化 CLIP 注意力可视化器
        
        Args:
            device: 计算设备
            patch_size: 图像块大小（CLIP 标准输入尺寸为 224x224）
            stride: 滑动窗口步长（stride < patch_size 会产生重叠，获得更密集的热力图）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        self.stride = stride
        
        print(f"初始化 CLIP 模型（设备: {self.device}）...")
        
        # 加载与项目一致的 CLIP 模型
        self.text_encoder = TextEncoder(
            device=self.device,
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )
        
        # 获取 CLIP 图像编码器和预处理
        self.clip_model = self.text_encoder.clip_model
        _, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device='cpu'
        )
        
        self.clip_model.eval()
        print("CLIP 模型加载完成！")
    
    def extract_image_patches(self, image: np.ndarray) -> Tuple[List[Image.Image], List[Tuple[int, int]]]:
        """
        从图像中提取图像块
        
        Args:
            image: BGR 格式的 OpenCV 图像 (H, W, 3)
            
        Returns:
            patches: PIL Image 图像块列表
            positions: 图像块中心位置列表 [(y, x), ...]
        """
        # 转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        patches = []
        positions = []
        
        # 使用滑动窗口提取图像块
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image_rgb[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(Image.fromarray(patch))
                # 记录图像块的中心位置
                center_y = y + self.patch_size // 2
                center_x = x + self.patch_size // 2
                positions.append((center_y, center_x))
        
        return patches, positions
    
    def compute_clip_similarity(self, text: str, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        计算文本与图像块的 CLIP 相似度
        
        Args:
            text: 文本描述
            image: BGR 格式的 OpenCV 图像
            
        Returns:
            heatmap: 相似度热力图 (H, W)
            max_similarity: 最大相似度分数
            mean_similarity: 平均相似度分数
        """
        h, w = image.shape[:2]
        
        # 编码文本
        with torch.no_grad():
            text_features = self.text_encoder([text])  # (1, 512)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 提取图像块
        patches, positions = self.extract_image_patches(image)
        
        if len(patches) == 0:
            return np.zeros((h, w)), 0.0, 0.0
        
        # 批量编码图像块
        similarities = []
        batch_size = 32
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]
            
            # 预处理并转换为张量
            batch_tensor = torch.stack([self.preprocess_val(p) for p in batch_patches]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(batch_tensor)  # (batch, 512)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 计算余弦相似度
                batch_similarities = (image_features @ text_features.T).squeeze(-1)  # (batch,)
                similarities.extend(batch_similarities.cpu().numpy().tolist())
        
        similarities = np.array(similarities)
        
        # 创建热力图
        heatmap = self._create_heatmap(similarities, positions, (h, w))
        
        max_similarity = float(similarities.max())
        mean_similarity = float(similarities.mean())
        
        return heatmap, max_similarity, mean_similarity
    
    def _create_heatmap(self, similarities: np.ndarray, positions: List[Tuple[int, int]], 
                       image_shape: Tuple[int, int]) -> np.ndarray:
        """
        从相似度分数和位置创建热力图
        
        Args:
            similarities: 相似度分数数组
            positions: 图像块中心位置列表
            image_shape: 目标图像形状 (H, W)
            
        Returns:
            heatmap: 插值后的热力图
        """
        h, w = image_shape
        
        # 创建稀疏热力图
        sparse_heatmap = np.zeros((h, w))
        weights = np.zeros((h, w))
        
        for (cy, cx), sim in zip(positions, similarities):
            # 使用高斯核进行软分配
            sigma = self.stride
            y_min = max(0, cy - self.patch_size // 2)
            y_max = min(h, cy + self.patch_size // 2)
            x_min = max(0, cx - self.patch_size // 2)
            x_max = min(w, cx + self.patch_size // 2)
            
            # 简化版本：直接在图像块区域内填充
            sparse_heatmap[y_min:y_max, x_min:x_max] += sim
            weights[y_min:y_max, x_min:x_max] += 1
        
        # 归一化
        weights[weights == 0] = 1  # 避免除以零
        heatmap = sparse_heatmap / weights
        
        # 应用高斯模糊平滑
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        return heatmap
    
    def visualize_attention(self, image: np.ndarray, heatmap: np.ndarray, 
                           object_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        可视化 CLIP 注意力
        
        Args:
            image: 原始 BGR 图像
            heatmap: 相似度热力图
            object_mask: 目标物体掩码（可选）
            
        Returns:
            original: 原始图像
            overlay: 热力图叠加图像
            mask_vis: 掩码可视化图像
        """
        # 原始图像（转换为 RGB）
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化热力图到 [0, 255]
        heatmap_norm = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
        
        # 应用颜色映射（使用 JET 色图，红色表示高相似度）
        heatmap_colored_bgr = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)
        
        # 叠加热力图到原图
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        # 可视化掩码
        if object_mask is not None and np.any(object_mask):
            mask_vis = original.copy()
            mask_binary = (object_mask > 0).astype(np.uint8)
            
            # 对背景轻微降亮，让目标区域更突出
            mask_vis = (mask_vis * 0.6).astype(np.uint8)
            
            # 创建显眼的绿色叠加层
            highlight_overlay = np.zeros_like(mask_vis)
            highlight_overlay[mask_binary > 0] = [64, 255, 112]
            mask_vis = cv2.addWeighted(mask_vis, 0.3, highlight_overlay, 0.7, 0)
            
            # 添加白色描边轮廓
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_vis, contours, -1, (255, 255, 255), 2)
        else:
            mask_vis = original.copy()
        
        return original, overlay, mask_vis, heatmap_colored
    
    def create_group_figure(self, samples_info: List[Dict]) -> plt.Figure:
        """
        将多个样本的可视化拼接在同一张图上。
        
        Args:
            samples_info: 每个元素包含 original/overlay/mask_overlay 等键的字典
        
        Returns:
            matplotlib Figure
        """
        num_samples = len(samples_info)
        fig_height = max(4.5 * num_samples, 4.5)
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, fig_height))
        
        if num_samples == 1:
            axes = np.expand_dims(axes, axis=0)
        
        for row_idx, sample in enumerate(samples_info):
            row_axes = axes[row_idx]
            scene_id = sample.get('scene_id', 'unknown')
            target_text = sample.get('positive_prompt', '')
            max_sim = sample.get('max_sim', 0.0)
            mean_sim = sample.get('mean_sim', 0.0)
            top_hit = sample.get('top_hit')
            sample_idx = sample.get('sample_idx')
            obj_code = sample.get('obj_code', '')
            
            row_axes[0].imshow(sample['original'])
            title_left = 'Original RGB'
            if sample_idx is not None:
                title_left = f"Sample {sample_idx} - {title_left}"
            row_axes[0].set_title(title_left, fontsize=13, family='sans-serif')
            row_axes[0].axis('off')
            row_axes[0].text(0.01, -0.08,
                             f"Scene: {scene_id}\nTarget: {target_text}",
                             ha='left', va='top', transform=row_axes[0].transAxes,
                             fontsize=11, family='sans-serif')
            
            row_axes[1].imshow(sample['overlay'])
            row_axes[1].set_title(f"CLIP Heatmap\nMax: {max_sim:.3f} | Mean: {mean_sim:.3f}",
                                  fontsize=13, family='sans-serif')
            row_axes[1].axis('off')
            
            if top_hit is None:
                hit_text = "Top-1 Hit: NA"
                hit_color = 'gray'
            elif top_hit:
                hit_text = "Top-1 Hit: YES"
                hit_color = 'limegreen'
            else:
                hit_text = "Top-1 Hit: NO"
                hit_color = 'crimson'
            row_axes[1].text(0.5, -0.10, hit_text, color=hit_color, fontsize=12,
                             ha='center', va='top', transform=row_axes[1].transAxes,
                             fontfamily='sans-serif', fontweight='bold')
            
            row_axes[2].imshow(sample['mask_overlay'])
            mask_title = "Ground Truth Mask"
            if obj_code:
                mask_title += f"\n{obj_code}"
            row_axes[2].set_title(mask_title, fontsize=13, family='sans-serif')
            row_axes[2].axis('off')
        
        fig.suptitle("CLIP Attention Results", fontsize=18, fontweight='bold', family='sans-serif')
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        return fig


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """将热力图归一化到 [0, 1]，供评估使用。"""
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_val) / (max_val - min_val)).astype(np.float32)


def evaluate_heatmap_with_mask(
    heatmap: np.ndarray,
    mask_2d: np.ndarray,
    thresholds: List[float]
) -> Dict[str, object]:
    """
    基于二维掩码评估热力图质量。
    
    Args:
        heatmap: CLIP 相似度热力图
        mask_2d: 目标物体二维布尔掩码
        thresholds: 归一化热力图的阈值列表（0~1）
        
    Returns:
        dict，包含 mask 像素数量、top-1 命中情况，以及各阈值下的 TP/FP/FN
    """
    if mask_2d is None:
        return {
            'mask_pixel_count': 0,
            'top_hit': None,
            'threshold_stats': {},
            'max_heatmap_value': 0.0
        }
    
    if heatmap.shape != mask_2d.shape:
        raise ValueError(f"Heatmap shape {heatmap.shape} 与掩码 {mask_2d.shape} 不一致，无法评估。")
    
    mask_bool = mask_2d.astype(bool)
    mask_pixel_count = int(mask_bool.sum())
    
    normalized_heatmap = _normalize_heatmap(heatmap)
    max_heatmap_value = float(normalized_heatmap.max())
    
    top_hit = None
    threshold_stats = {}
    
    if mask_pixel_count > 0:
        top_coords = np.unravel_index(np.argmax(normalized_heatmap), normalized_heatmap.shape)
        top_hit = bool(mask_bool[top_coords])
        
        for thr in thresholds:
            pred_mask = normalized_heatmap >= thr
            tp = int(np.logical_and(pred_mask, mask_bool).sum())
            fp = int(pred_mask.sum() - tp)
            fn = int(mask_pixel_count - tp)
            threshold_stats[thr] = {'tp': tp, 'fp': fp, 'fn': fn}
    
    return {
        'mask_pixel_count': mask_pixel_count,
        'top_hit': top_hit,
        'threshold_stats': threshold_stats,
        'max_heatmap_value': max_heatmap_value
    }


def load_dataset_sample(dataset: SceneLeapPlusDataset, idx: int, root_dir: str) -> Dict:
    """
    加载数据集样本并获取原始 RGB 图像
    
    Args:
        dataset: SceneLeapPlusDataset 实例
        idx: 样本索引
        root_dir: 数据集根目录
        
    Returns:
        sample_data: 包含所有需要的数据
    """
    # 获取数据集处理后的样本
    item = dataset[idx]
    
    # 获取原始 RGB 图像
    scene_id = item['scene_id']
    depth_view_idx = item['depth_view_index']
    scene_dir = os.path.join(root_dir, scene_id)
    
    # 加载原始图像及实例掩码
    _, rgb_image, instance_mask = load_scene_images(scene_dir, depth_view_idx)
    
    # 从 scene_pc 提取对应的 RGB（验证用）
    scene_pc_with_rgb = item['scene_pc'].numpy()  # (N, 6) - [x, y, z, r, g, b]
    
    # 获取物体掩码（映射回 2D）
    object_mask = item['object_mask'].numpy()
    
    # 计算实例图中的目标物体像素区域
    if instance_mask is not None:
        category_id = int(item.get('category_id_from_object_index', -1))
        view_attrs = dataset.instance_maps.get(scene_id, {}).get(str(depth_view_idx), [])
        mask_flat = extract_object_mask(instance_mask, category_id, view_attrs)
        mask_2d = mask_flat.reshape(instance_mask.shape)
    else:
        mask_2d = np.zeros(rgb_image.shape[:2], dtype=bool)
    
    return {
        'rgb_image': rgb_image,
        'scene_pc': scene_pc_with_rgb,
        'object_mask': object_mask,
        'object_mask_2d': mask_2d,
        'positive_prompt': item['positive_prompt'],
        'scene_id': scene_id,
        'depth_view_index': depth_view_idx,
        'obj_code': item['obj_code']
    }


def run_clip_attention_experiment(num_samples=10, random_seed=42):
    """
    运行 CLIP 注意力实验
    
    Args:
        num_samples: 测试样本数量
        random_seed: 随机种子
    """
    print("="*80)
    print("CLIP 文本-图像注意力测试实验")
    print("="*80)
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 创建输出目录
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "bin" / "clip_attention_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 评估参数
    thresholds = [round(v, 2) for v in np.linspace(0.2, 0.9, 8)]
    threshold_totals = {thr: {'tp': 0, 'fp': 0, 'fn': 0} for thr in thresholds}
    top_hit_success = 0
    top_hit_total = 0
    group_samples: List[Dict] = []
    group_index = 1
    
    # 初始化 CLIP 可视化器
    visualizer = CLIPAttentionVisualizer(device='cuda', patch_size=224, stride=112)
    
    # 加载数据集（train 分割）
    print("\n加载 SceneLeapPlusDataset (train)...")
    train_root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/test_final_test_723_2_processed"
    succ_grasp_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/succ_collect"
    obj_root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/object/objaverse_v1/flat_models"
    
    dataset = SceneLeapPlusDataset(
        root_dir=train_root_dir,
        succ_grasp_dir=succ_grasp_dir,
        obj_root_dir=obj_root_dir,
        num_grasps=8,
        mode="camera_centric",
        max_grasps_per_object=200,
        mesh_scale=0.1,
        num_neg_prompts=4,
        enable_cropping=True,
        max_points=10000,
        grasp_sampling_strategy="random",
        use_exhaustive_sampling=False
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 随机选择样本
    num_samples = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f"\n随机选择 {num_samples} 个样本进行测试...")
    
    # 存储统计信息
    statistics = []
    
    # 处理每个样本
    for i, idx in enumerate(tqdm(sample_indices, desc="处理样本")):
        try:
            # 加载样本
            sample = load_dataset_sample(dataset, idx, train_root_dir)
            
            rgb_image = sample['rgb_image']
            positive_prompt = sample['positive_prompt']
            object_mask = sample['object_mask']
            object_mask_2d = sample.get('object_mask_2d')
            scene_id = sample['scene_id']
            obj_code = sample['obj_code']
            
            print(f"\n[{i+1}/{num_samples}] 场景: {scene_id}, 物体: {positive_prompt}")
            
            # 计算 CLIP 相似度热力图
            heatmap, max_sim, mean_sim = visualizer.compute_clip_similarity(
                positive_prompt, rgb_image
            )
            
            print(f"  - 最大相似度: {max_sim:.4f}")
            print(f"  - 平均相似度: {mean_sim:.4f}")
            
            # 评估热力图与真实掩码
            metrics_info = evaluate_heatmap_with_mask(heatmap, object_mask_2d, thresholds)
            mask_pixel_count = metrics_info['mask_pixel_count']
            top_hit = metrics_info['top_hit']
            
            if mask_pixel_count > 0 and top_hit is not None:
                top_hit_total += 1
                if top_hit:
                    top_hit_success += 1
                for thr, counts in metrics_info['threshold_stats'].items():
                    threshold_totals[thr]['tp'] += counts['tp']
                    threshold_totals[thr]['fp'] += counts['fp']
                    threshold_totals[thr]['fn'] += counts['fn']
            
            # 可视化
            original, overlay, mask_vis, heatmap_rgb = visualizer.visualize_attention(
                rgb_image, heatmap, object_mask_2d
            )
            
            # 记录统计信息
            statistics.append({
                'sample_idx': i + 1,
                'dataset_idx': idx,
                'scene_id': scene_id,
                'obj_code': obj_code,
                'positive_prompt': positive_prompt,
                'max_similarity': float(max_sim),
                'mean_similarity': float(mean_sim),
                'mask_pixel_count': int(mask_pixel_count),
                'heatmap_top_hit': top_hit,
                'heatmap_max_normalized': float(metrics_info['max_heatmap_value'])
            })
            
            # 收集可视化结果，5 个样本拼接一张图
            mask_heatmap_overlay = cv2.addWeighted(
                mask_vis.astype(np.uint8), 0.5,
                heatmap_rgb.astype(np.uint8), 0.5, 0
            )
            group_samples.append({
                'original': original,
                'overlay': overlay,
                'mask_overlay': mask_heatmap_overlay,
                'scene_id': scene_id,
                'positive_prompt': positive_prompt,
                'max_sim': max_sim,
                'mean_sim': mean_sim,
                'top_hit': top_hit,
                'sample_idx': i + 1,
                'obj_code': obj_code
            })
            
            if len(group_samples) == 5:
                fig = visualizer.create_group_figure(group_samples)
                save_path = output_dir / f"group_{group_index:03d}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  - 已保存组合图: {save_path.name}")
                group_index += 1
                group_samples = []
            
        except Exception as e:
            print(f"  - 错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存统计报告
    report_path = output_dir / "summary_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        threshold_metrics_summary = []
        if top_hit_total > 0:
            for thr in thresholds:
                counts = threshold_totals[thr]
                tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
                precision = tp / (tp + fp) if (tp + fp) > 0 else None
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
                denom = tp + fp + fn
                iou = tp / denom if denom > 0 else None
                threshold_metrics_summary.append({
                    'threshold': thr,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'iou': iou
                })
        
        evaluation_summary = {
            'num_valid_samples': top_hit_total,
            'num_skipped_samples': len(statistics) - top_hit_total,
            'top1_hit_success': top_hit_success,
            'top1_hit_rate': (top_hit_success / top_hit_total) if top_hit_total > 0 else None,
            'threshold_metrics': threshold_metrics_summary
        }
        
        json.dump({
            'total_samples': len(statistics),
            'statistics': statistics,
            'summary': {
                'avg_max_similarity': np.mean([s['max_similarity'] for s in statistics]),
                'avg_mean_similarity': np.mean([s['mean_similarity'] for s in statistics]),
                'std_max_similarity': np.std([s['max_similarity'] for s in statistics]),
                'std_mean_similarity': np.std([s['mean_similarity'] for s in statistics])
            },
            'evaluation_metrics': evaluation_summary
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"总样本数: {len(statistics)}")
    print(f"平均最大相似度: {np.mean([s['max_similarity'] for s in statistics]):.4f} ± {np.std([s['max_similarity'] for s in statistics]):.4f}")
    print(f"平均均值相似度: {np.mean([s['mean_similarity'] for s in statistics]):.4f} ± {np.std([s['mean_similarity'] for s in statistics]):.4f}")
    
    # 保存剩余未满组的可视化
    if group_samples:
        fig = visualizer.create_group_figure(group_samples)
        save_path = output_dir / f"group_{group_index:03d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  - 已保存组合图: {save_path.name}")
    
    
    if top_hit_total > 0:
        top_hit_rate = top_hit_success / top_hit_total
        print(f"Top-1 命中率: {top_hit_success}/{top_hit_total} = {top_hit_rate:.3f}")
        print("\n阈值评估（基于归一化热力图）：")
        for metrics_item in threshold_metrics_summary:
            thr = metrics_item['threshold']
            precision = metrics_item['precision']
            recall = metrics_item['recall']
            iou = metrics_item['iou']
            precision_str = f"{precision:.3f}" if precision is not None else "NA"
            recall_str = f"{recall:.3f}" if recall is not None else "NA"
            iou_str = f"{iou:.3f}" if iou is not None else "NA"
            print(f"  阈值 {thr:.2f} -> Precision {precision_str}, Recall {recall_str}, IoU {iou_str}")
    else:
        print("未找到有效掩码样本，跳过阈值评估。")
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # 运行实验，测试 15 个样本
    run_clip_attention_experiment(num_samples=500, random_seed=42)
