"""
CLIP 文本-图像注意力测试实验

测试 CLIP 模型能否通过文本描述准确识别和关注场景中的目标物体。
使用图像块相似度热力图方法进行可视化。
"""

import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict
import random
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
from datasets.utils.io_utils import load_scene_images
from models.utils.text_encoder import TextEncoder
import open_clip
from PIL import Image
import torch.nn.functional as F


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
                           object_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加热力图到原图
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        # 可视化掩码
        if object_mask is not None:
            mask_vis = original.copy()
            # 创建绿色叠加层
            green_overlay = np.zeros_like(mask_vis)
            green_overlay[object_mask > 0] = [0, 255, 0]
            mask_vis = cv2.addWeighted(mask_vis, 0.7, green_overlay, 0.3, 0)
            
            # 添加轮廓
            contours, _ = cv2.findContours(object_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_vis, contours, -1, (0, 255, 0), 2)
        else:
            mask_vis = original.copy()
        
        return original, overlay, mask_vis
    
    def create_comparison_figure(self, original: np.ndarray, overlay: np.ndarray, 
                                mask_vis: np.ndarray, text: str, scene_id: str,
                                max_sim: float, mean_sim: float) -> plt.Figure:
        """
        创建对比可视化图
        
        Args:
            original: 原始图像
            overlay: 热力图叠加
            mask_vis: 掩码可视化
            text: 文本描述
            scene_id: 场景 ID
            max_sim: 最大相似度
            mean_sim: 平均相似度
            
        Returns:
            fig: matplotlib Figure 对象
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 原始图像
        axes[0].imshow(original)
        axes[0].set_title('Original RGB Image', fontsize=14, family='sans-serif')
        axes[0].axis('off')
        
        # CLIP 注意力热力图
        axes[1].imshow(overlay)
        axes[1].set_title(f'CLIP Attention Heatmap\nMax Similarity: {max_sim:.3f} | Mean: {mean_sim:.3f}', 
                         fontsize=14, family='sans-serif')
        axes[1].axis('off')
        
        # 真实目标掩码
        axes[2].imshow(mask_vis)
        axes[2].set_title('Ground Truth Object Mask', fontsize=14, family='sans-serif')
        axes[2].axis('off')
        
        # 添加总标题
        fig.suptitle(f'Scene: {scene_id} | Target Object: {text}', 
                    fontsize=16, fontweight='bold', family='sans-serif')
        
        plt.tight_layout()
        
        return fig


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
    
    # 加载原始图像
    _, rgb_image, _ = load_scene_images(scene_dir, depth_view_idx)
    
    # 从 scene_pc 提取对应的 RGB（验证用）
    scene_pc_with_rgb = item['scene_pc'].numpy()  # (N, 6) - [x, y, z, r, g, b]
    
    # 获取物体掩码（映射回 2D）
    object_mask = item['object_mask'].numpy()
    
    return {
        'rgb_image': rgb_image,
        'scene_pc': scene_pc_with_rgb,
        'object_mask': object_mask,
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
    output_dir = Path(__file__).parent / "clip_attention_results"
    output_dir.mkdir(exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # 初始化 CLIP 可视化器
    visualizer = CLIPAttentionVisualizer(device='cuda', patch_size=224, stride=112)
    
    # 加载数据集（train 分割）
    print("\n加载 SceneLeapPlusDataset (train)...")
    train_root_dir = "/home/xiantuo/source/grasp/SceneLeapPro/data/test_final_test_520_1_processed"
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
            scene_id = sample['scene_id']
            obj_code = sample['obj_code']
            
            print(f"\n[{i+1}/{num_samples}] 场景: {scene_id}, 物体: {positive_prompt}")
            
            # 计算 CLIP 相似度热力图
            heatmap, max_sim, mean_sim = visualizer.compute_clip_similarity(
                positive_prompt, rgb_image
            )
            
            print(f"  - 最大相似度: {max_sim:.4f}")
            print(f"  - 平均相似度: {mean_sim:.4f}")
            
            # 创建掩码可视化（需要将点云掩码映射回图像空间）
            # 由于直接映射比较复杂，我们使用占位符
            mask_2d = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
            
            # 可视化
            original, overlay, mask_vis = visualizer.visualize_attention(
                rgb_image, heatmap, mask_2d
            )
            
            # 创建对比图
            fig = visualizer.create_comparison_figure(
                original, overlay, mask_vis,
                positive_prompt, scene_id, max_sim, mean_sim
            )
            
            # 保存图像
            save_path = output_dir / f"sample_{i+1:03d}_{scene_id}_{obj_code}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  - 已保存: {save_path.name}")
            
            # 记录统计信息
            statistics.append({
                'sample_idx': i + 1,
                'dataset_idx': idx,
                'scene_id': scene_id,
                'obj_code': obj_code,
                'positive_prompt': positive_prompt,
                'max_similarity': float(max_sim),
                'mean_similarity': float(mean_sim)
            })
            
        except Exception as e:
            print(f"  - 错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存统计报告
    report_path = output_dir / "summary_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_samples': len(statistics),
            'statistics': statistics,
            'summary': {
                'avg_max_similarity': np.mean([s['max_similarity'] for s in statistics]),
                'avg_mean_similarity': np.mean([s['mean_similarity'] for s in statistics]),
                'std_max_similarity': np.std([s['max_similarity'] for s in statistics]),
                'std_mean_similarity': np.std([s['mean_similarity'] for s in statistics])
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"总样本数: {len(statistics)}")
    print(f"平均最大相似度: {np.mean([s['max_similarity'] for s in statistics]):.4f} ± {np.std([s['max_similarity'] for s in statistics]):.4f}")
    print(f"平均均值相似度: {np.mean([s['mean_similarity'] for s in statistics]):.4f} ± {np.std([s['mean_similarity'] for s in statistics]):.4f}")
    print(f"\n所有结果已保存到: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    # 运行实验，测试 15 个样本
    run_clip_attention_experiment(num_samples=50, random_seed=42)

