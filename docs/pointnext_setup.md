# PointNext Backbone 安装和使用指南

## 概述

PointNext 是一个高效的点云编码器，基于改进的 PointNet++ 架构。本文档说明如何在 SceneLeapUltra 项目中使用 PointNext backbone。

## 架构特点

- **输入**: (B, N, 3) 点云，N=8192
- **输出**: (B, K, d_model) tokens，K=128，d_model=512
- **参数量**: ~3M (标准配置)
- **速度**: ~60ms/sample (单 GPU)
- **性能**: 介于 PointNet2 和 PTv3 之间

## 安装依赖

### 方法1: 安装 multimethod (最简单)

PointNext 依赖于 OpenPoints 库，而 OpenPoints 需要 `multimethod` 包：

```bash
conda activate DexGrasp
pip install multimethod
```

### 方法2: 安装完整的 OpenPoints (推荐)

如果需要使用 OpenPoints 的其他功能，可以完整安装：

```bash
conda activate DexGrasp
cd third_party/openpoints
pip install -e .
```

注意：OpenPoints 的 C++ 扩展（如 pointops）是可选的。如果编译失败，不影响 PointNext 的使用。

## 使用方法

### 1. 通过配置文件使用

```yaml
# 在你的配置文件中指定 backbone
model:
  decoder:
    backbone:
      name: pointnext
      num_points: 8192
      num_tokens: 128
      out_dim: 512
      # ... 其他参数见 config/model/flow_matching/decoder/backbone/pointnext.yaml
```

### 2. 通过代码使用

```python
from omegaconf import OmegaConf
from models.backbone import build_backbone

# 加载配置
cfg = OmegaConf.load('config/model/flow_matching/decoder/backbone/pointnext.yaml')

# 创建 backbone
backbone = build_backbone(cfg).cuda()

# 使用
import torch
pointcloud = torch.randn(2, 8192, 3).cuda()  # (B, N, 3)
xyz, features = backbone(pointcloud)
# xyz: (B, 128, 3), features: (B, 512, 128)
```

### 3. 切换不同的 backbone

项目支持多种 backbone，可以通过修改配置轻松切换：

```bash
# 使用 PointNext
python train_lightning.py model.decoder.backbone=pointnext

# 使用 PointNet2
python train_lightning.py model.decoder.backbone=pointnet2

# 使用 PTv3
python train_lightning.py model.decoder.backbone=ptv3

# 使用 PTv3 Sparse
python train_lightning.py model.decoder.backbone=ptv3_sparse
```

## 配置参数说明

主要配置参数（位于 `config/model/flow_matching/decoder/backbone/pointnext.yaml`）：

### 基本参数

- `num_points` (默认: 8192): 输入点数
- `num_tokens` (默认: 128): 输出 token 数量
- `out_dim` (默认: 512): 输出特征维度

### 架构参数

- `width` (默认: 32): 基础通道宽度
  - 32: 轻量级 (~2M 参数)
  - 64: 标准 (~8M 参数)
  - 128: 重量级 (~25M 参数)

- `blocks` (默认: [1,1,1,1,1]): 每个阶段的块数
  - 增加块数可以提升表达能力，但会增加计算量

- `strides` (默认: [1,4,4,4,4]): 每个阶段的下采样步长
  - stride=1: 不下采样
  - stride=4: 下采样 4x
  - 总下采样率: 1×4×4×4×4 = 256x

### 局部聚合参数

- `radius` (默认: 0.1): Ball query 半径（米）
  - 越大 → 更大感受野
  - 越小 → 更精细的局部特征

- `nsample` (默认: 32): 每个查询点的邻居数
  - 越多 → 更稳定，但更慢
  - 越少 → 更快，但可能不稳定

### 输入/输出配置

- `input_feature_dim` (默认: 3): 输入特征维度
  - 3: 仅 xyz
  - 6: xyz + rgb
  
- `use_xyz` (默认: true): 是否使用 xyz 作为特征
- `normalize_xyz` (默认: true): 是否归一化坐标
- `use_fps` (默认: true): 是否使用 FPS 采样提取 tokens

## 预设配置

### 轻量级（快速原型）
```yaml
width: 32
blocks: [1, 1, 1, 1, 1]
num_tokens: 128
out_dim: 256
# 参数量: ~2M, 速度: ~50ms/sample
```

### 标准（平衡）- 默认配置
```yaml
width: 32
blocks: [1, 1, 1, 1, 1]
num_tokens: 128
out_dim: 512
# 参数量: ~3M, 速度: ~60ms/sample
```

### 增强（更强表达）
```yaml
width: 64
blocks: [1, 2, 2, 2, 1]
num_tokens: 256
out_dim: 512
# 参数量: ~8M, 速度: ~150ms/sample
```

### 重量级（最佳性能）
```yaml
width: 128
blocks: [2, 2, 3, 3, 2]
num_tokens: 512
out_dim: 768
# 参数量: ~25M, 速度: ~500ms/sample
```

## Backbone 对比

| Backbone | 参数量 | 速度 | 性能 | 特点 |
|----------|--------|------|------|------|
| PointNet2 | ~2M | ~30ms | 基线 | 简单高效 |
| **PointNext** | **~3M** | **~60ms** | **中等** | **平衡性能和效率** |
| PTv3 | ~10M | ~200ms | 最佳 | Transformer，全局建模强 |
| PTv3 Sparse | ~10M | ~150ms | 最佳 | 稀疏 tokens，DiT 友好 |

## 测试

运行测试以验证安装：

```bash
conda activate DexGrasp
cd tests
python test_pointnext_backbone.py
```

测试将验证：
1. 基本功能和形状
2. 不同输入格式
3. 配置文件加载
4. 与其他 backbone 的兼容性
5. 性能基准

## 常见问题

### Q1: ImportError: No module named 'multimethod'

**解决方案**: 安装 multimethod
```bash
pip install multimethod
```

### Q2: PointNext 创建失败

**解决方案**: 检查依赖是否安装正确
```bash
python -c "from openpoints.models.backbone.pointnext import PointNextEncoder; print('OK')"
```

### Q3: 输出形状不符合预期

**检查**:
- 确认 `num_tokens` 配置正确
- 确认 `out_dim` 与 decoder 输入维度匹配
- 检查 strides 配置（影响下采样率）

### Q4: 显存不足

**解决方案**:
1. 减小 `width` (如 32 → 16)
2. 减小 `num_points` (如 8192 → 4096)
3. 减小 `blocks` (如 [1,1,1,1,1] → [1,1,1,1])
4. 使用梯度检查点（需要修改代码）

### Q5: 推理速度慢

**解决方案**:
1. 减小 `width`
2. 减小 `blocks`
3. 增大 `strides` (更强下采样)
4. 减小 `nsample` (邻居数)
5. 使用 `use_fps=false` (更快但不均匀的采样)

## 调试信息

如果遇到问题，可以启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

这将输出：
- Encoder 构建信息
- 输入/输出形状
- 性能统计

## 贡献

如果发现问题或有改进建议，请：
1. 在 tests/ 目录下创建测试用例
2. 提交 issue 或 pull request
3. 更新文档

## 参考资料

- PointNeXt 论文: https://arxiv.org/abs/2206.04670
- OpenPoints 仓库: https://github.com/guochengqian/openpoints
- PointNet++ 论文: https://arxiv.org/abs/1706.02413

