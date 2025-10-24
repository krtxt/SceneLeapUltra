# Backbone配置文件说明

本目录包含了独立的backbone配置文件，支持6维点云数据（XYZ + RGB）的处理。

## 📁 文件结构

```
backbone/
├── pointnet2.yaml          # PointNet2 backbone配置
├── ptv3.yaml              # PTv3 backbone配置 (标准版，启用Flash Attention)
├── ptv3_light.yaml        # PTv3 backbone配置 (轻量版，参数量减少~75%)
├── ptv3_no_flash.yaml     # PTv3 backbone配置 (禁用Flash Attention)
└── README.md              # 本说明文件
```

## 🔧 配置文件详情

### PointNet2配置 (`pointnet2.yaml`)

**特点:**
- 基于层次化采样的点云处理
- 参数量: ~560K
- 推理速度快
- 适合实时应用

**配置参数:**
- 输入: `[B, N, 6]` (xyz + rgb)
- 输出: `xyz [B, 128, 3]`, `features [B, 512, 128]`
- 采样层次: N → 2048 → 1024 → 512 → 128
- 特征维度: 3 → 128 → 256 → 256 → 512

### PTv3配置 (`ptv3.yaml`)

**特点:**
- 基于稀疏Transformer的点云处理
- 参数量: ~46M
- 特征表达能力强
- 支持大规模点云
- **启用Flash Attention** (推荐)

**配置参数:**
- 输入: `[B, N, 6]` (xyz + rgb)
- 输出: `xyz [B, K, 3]`, `features [B, 512, K]` (K为稀疏化后点数)
- 编码器: 5层，通道数 32→64→128→256→512
- 解码器: 4层，恢复空间分辨率
- Flash Attention: 启用，提升性能

### PTv3轻量版配置 (`ptv3_light.yaml`)

**特点:**
- 基于稀疏Transformer的轻量化点云处理
- 参数量: ~8-12M (减少约75%)
- 推理速度快，资源占用低
- 适合资源受限环境
- **启用Flash Attention**

**配置参数:**
- 输入: `[B, N, 6]` (xyz + rgb)
- 输出: `xyz [B, K, 3]`, `features [B, 256, K]` (最终特征256维)
- 编码器: 5层，通道数 16→32→64→128→256
- 编码器深度: [1, 1, 2, 2, 1] (减少约45%)
- 解码器: 4层，深度 [1, 1, 1, 2]
- MLP比例: 2 (减少50%)
- 网格大小: 0.03 (稍大，减少计算量)

### PTv3无Flash Attention配置 (`ptv3_no_flash.yaml`)

**特点:**
- 与ptv3.yaml相同的网络结构
- **禁用Flash Attention**
- 适合兼容性要求高的环境
- 适合较老的GPU硬件

## 🚀 使用方法

### 方法1: 使用defaults引用 (推荐)

```yaml
# 在unet.yaml中使用PointNet2
defaults:
  - backbone: pointnet2

# 切换到PTv3标准版
defaults:
  - backbone: ptv3

# 使用PTv3轻量版 (推荐用于资源受限场景)
defaults:
  - backbone: ptv3_light

# 使用无Flash Attention的PTv3
defaults:
  - backbone: ptv3_no_flash
```

### 方法2: 直接引用配置文件

```yaml
backbone: ${oc.create:config/model/diffuser/decoder/backbone/pointnet2.yaml}
```

### 方法3: 覆盖特定参数

```yaml
defaults:
  - backbone: pointnet2

backbone:
  use_pooling: true  # 覆盖默认设置
```

## 📊 性能对比

| Backbone | Flash Attn | 参数量 | 推理时间* | 内存占用 | 适用场景 |
|----------|------------|--------|-----------|----------|----------|
| PointNet2 | N/A | 560K | ~17ms | 低 | 实时应用、资源受限 |
| PTv3 Light | ✅ | 8-12M | ~30-40ms | 低-中等 | 平衡性能与精度 |
| PTv3 | ✅ | 46M | ~60ms | 中等 | 高精度、现代GPU |
| PTv3 | ❌ | 46M | ~85ms | 中等 | 兼容性、老GPU |

*测试条件: batch=2, points=1024, GPU

## ⚡ Flash Attention说明

### 什么是Flash Attention？
Flash Attention是一种内存高效的注意力计算方法，可以显著提升Transformer的训练和推理速度。

### 优势：
- **速度提升**: 推理速度提升约30%
- **内存节省**: 显著降低GPU内存占用
- **精度保持**: 数值精度与标准注意力相同

### 使用建议：
- **推荐使用**: `ptv3.yaml` (启用Flash Attention)
- **兼容性需求**: `ptv3_no_flash.yaml` (禁用Flash Attention)

### 硬件要求：
- GPU: 支持CUDA的现代GPU (推荐RTX 20系列及以上)
- 驱动: 较新的CUDA驱动版本
- 软件: 安装了flash-attn包

## 🔍 配置验证

运行测试脚本验证配置是否正确：

```bash
# 测试backbone配置
python tests/test_backbone_configs.py

# 测试defaults风格配置
python tests/test_defaults_config.py
```

## ⚙️ 自定义配置

### 切换backbone

```yaml
# 原配置
defaults:
  - backbone: pointnet2

# 切换到PTv3
defaults:
  - backbone: ptv3
```

### 调整Flash Attention

```yaml
defaults:
  - backbone: ptv3

backbone:
  enable_flash_attn: false  # 临时禁用Flash Attention
```

### 覆盖其他参数

```yaml
defaults:
  - backbone: pointnet2

backbone:
  use_pooling: true
  layer1:
    npoint: 4096  # 增加采样点数
```

## 📝 注意事项

1. **输入格式**: 所有配置都假设输入为6维点云 (xyz + rgb)
2. **Flash Attention**: PTv3默认启用，如遇问题可切换到ptv3_no_flash
3. **硬件兼容**: 较老GPU建议使用pointnet2或ptv3_no_flash
4. **内存管理**: PTv3使用稀疏化，内存效率较高
5. **配置优先级**: backbone字段中的设置会覆盖defaults引用的配置

## 🐛 故障排除

### Flash Attention相关问题

```bash
# 如果遇到Flash Attention错误
ImportError: No module named 'flash_attn'

# 解决方案1: 安装flash-attn
pip install flash-attn

# 解决方案2: 使用无Flash Attention版本
defaults:
  - backbone: ptv3_no_flash
```

### 配置加载问题

```python
# 检查配置是否正确加载
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/model/diffuser/decoder/unet.yaml')
print(cfg.defaults)  # 应该显示backbone引用
```

## 📚 相关文档

- [PointNet2论文](https://arxiv.org/abs/1706.02413)
- [Point Transformer V3论文](https://arxiv.org/abs/2312.10035)
- [Flash Attention论文](https://arxiv.org/abs/2205.14135)
- [Hydra配置系统](https://hydra.cc/docs/intro/)
