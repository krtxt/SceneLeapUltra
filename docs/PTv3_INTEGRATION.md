# PTv3 Backbone Integration

本文档说明如何在 SceneLeapUltra 中使用 Point Transformer V3 (PTv3) 作为点云编码器。

## 概述

PTv3 已成功集成为与 PointNet2 并行的点云编码器选项，提供三种配置：
- **ptv3_light**: 轻量级版本，默认输出维度 512（与配置一致），适合资源受限场景
- **ptv3**: 标准版本，输出维度 512，启用 Flash Attention
- **ptv3_no_flash**: 标准版本但禁用 Flash Attention，适合兼容性需求

## 接口兼容性

PTv3Backbone 与 PointNet2Backbone 完全兼容：
- **输入**: `(B, N, C)` 其中 C = xyz(3) + 可选特征
- **输出**: `(xyz, features)` 其中 xyz 为 `(B, K, 3)`, features 为 `(B, out_dim, K)`
- **动态通道适配**: 根据 `use_rgb` 和 `use_object_mask` 自动处理输入通道

## 使用方法

### 1. 在 DiT 中使用

修改配置文件 `config/model/diffuser/decoder/dit.yaml`:

```yaml
defaults:
  - backbone: ptv3_light  # 或 ptv3, ptv3_no_flash

name: dit
# ... 其他配置 ...
```

或通过命令行覆盖：

```bash
python train_lightning.py model/diffuser/decoder=dit model/diffuser/decoder/backbone=ptv3_light
```

### 2. 在 UNet 中使用

修改配置文件 `config/model/diffuser/decoder/unet.yaml`:

```yaml
defaults:
  - backbone: ptv3_light  # 或 ptv3, ptv3_no_flash

name: unet
# ... 其他配置 ...
```

或通过命令行覆盖：

```bash
python train_lightning.py model/diffuser/decoder=unet model/diffuser/decoder/backbone=ptv3_light
```

## 配置参数

### ptv3_light.yaml
```yaml
name: ptv3
variant: light
use_flash_attention: true
grid_size: 0.03
encoder_channels: [32, 64, 128, 256, 512]
encoder_depths: [1, 1, 2, 2, 1]
encoder_num_head: [2, 4, 8, 16, 16]
enc_patch_size: [1024, 1024, 1024, 1024, 1024]
decoder_channels: [512, 256, 128, 64]
decoder_depths: [1, 2, 1, 1]
dec_patch_size: [1024, 1024, 1024, 1024]
mlp_ratio: 2
out_dim: 512  # 输出特征维度
```

### ptv3.yaml
```yaml
name: ptv3
variant: base
use_flash_attention: true
grid_size: 0.03
encoder_channels: [32, 64, 128, 256, 512]
encoder_depths: [2, 2, 2, 2, 2]
encoder_num_head: [2, 4, 8, 16, 16]
enc_patch_size: [1024, 1024, 1024, 1024, 1024]
decoder_channels: [512, 256, 128, 64]
decoder_depths: [2, 2, 2, 2]
dec_patch_size: [1024, 1024, 1024, 1024]
mlp_ratio: 4
out_dim: 512  # 输出特征维度
```

### ptv3_no_flash.yaml
```yaml
name: ptv3
variant: no_flash
use_flash_attention: false  # 禁用 Flash Attention
grid_size: 0.03
encoder_channels: [32, 64, 128, 256, 512]
encoder_depths: [2, 2, 2, 2, 2]
encoder_num_head: [2, 4, 8, 16, 16]
enc_patch_size: [1024, 1024, 1024, 1024, 1024]
decoder_channels: [512, 256, 128, 64]
decoder_depths: [2, 2, 2, 2]
dec_patch_size: [1024, 1024, 1024, 1024]
mlp_ratio: 4
out_dim: 512
```

## 特征通道处理

PTv3Backbone 自动处理不同的输入通道组合：

| use_rgb | use_object_mask | 输入维度 | 说明 |
|---------|----------------|---------|------|
| False   | False          | (B,N,3) | 仅 xyz |
| True    | False          | (B,N,6) | xyz + rgb |
| False   | True           | (B,N,4) | xyz + mask |
| True    | True           | (B,N,7) | xyz + rgb + mask |

内部实现：
- xyz 前 3 维作为坐标 (`coord`)
- 后续维度作为特征 (`feat`)
- 若无额外特征，使用 `ones(B,N,1)` 作为占位

## 显存与性能

### 参数量对比
- PointNet2: ~560K (2.2 MB FP32)
- PTv3 Light: ~8-12M
- PTv3 Standard: ~46M

### 显存占用（FP32，batch=2，points=1024）
- PointNet2: ~100-200 MB (激活)
- PTv3 Light: ~300-500 MB (激活)
- PTv3 Standard: ~500-800 MB (激活)

### 推理速度（估计）
- PointNet2: ~17ms
- PTv3 Light: ~30-40ms
- PTv3 Standard: ~60ms (Flash) / ~85ms (No Flash)

## 测试

运行冒烟测试验证集成：

```bash
# 测试 PTv3 backbone 基础功能
python tests/test_ptv3_backbone_basic.py

# 测试 DiT + PTv3
python tests/test_smoke_dit_ptv3.py

# 测试 UNet + PTv3
python tests/test_smoke_unet_ptv3.py
```

## 依赖

确保安装以下依赖：
```bash
pip install spconv-cu118  # 或对应的 CUDA 版本
pip install torch-scatter
pip install addict
pip install flash-attn  # 可选，用于 ptv3 和 ptv3_light
```

如无 Flash Attention，使用 `ptv3_no_flash` 配置。

## 故障排除

### ImportError: No module named 'flash_attn'
**解决方案**: 使用 `ptv3_no_flash` 或安装 flash-attn:
```bash
pip install flash-attn
```

### CUDA out of memory
**解决方案**:
1. 使用 `ptv3_light` 而非 `ptv3`
2. 减小 batch size
3. 减少点云数量 (`max_points`)
4. 启用混合精度训练

### 稀疏张量相关错误
**解决方案**: 确保安装了正确版本的 spconv:
```bash
pip install spconv-cu118  # CUDA 11.8
pip install spconv-cu117  # CUDA 11.7
```

## 内部实现细节

### 数据流

1. **输入预处理** (`PTV3Backbone.forward`):
   - 分离 xyz 和 feat
   - 转换为 PTv3 稀疏格式 (coord, feat, offset, grid_size)

2. **PTv3 前向**:
   - Serialization (z-order)
   - Sparsification (voxelization)
   - Encoder (5层下采样 + attention)
   - Decoder (4层上采样 + attention)

3. **输出聚合** (`densify_ptv3_output`):
   - 从稀疏格式转回致密格式
   - 按 batch 组织
   - 输出 (B, K, 3) xyz 和 (B, C, K) features

### 自动投影

解码器会自动处理 backbone 输出维度不匹配：
- DiT: `scene_projection` 固定为 `Linear(backbone_out_dim, d_model)`
- UNet: 若 `backbone_out_dim != d_model` 则添加投影层，否则用 `Identity`

## 引用

如果使用 PTv3，请引用：
```bibtex
@article{wu2023point,
  title={Point transformer v3: Simpler, faster, stronger},
  author={Wu, Xiaoyang and others},
  journal={arXiv preprint arXiv:2312.10035},
  year={2023}
}
```

