# Point Transformer 集成文档

## 概述

Point Transformer 已成功集成到 SceneLeapUltra 项目中，可与 PointNet2 和 PTv3 并列使用。

## 依赖说明

Point Transformer 依赖于一个编译的 CUDA 扩展 `pointops_cuda`，这个模块需要在使用前编译。

### 编译 pointops_cuda

`pointops_cuda` 是 Point Transformer 的核心 CUDA 算子库，用于高效的点云操作（最远点采样、KNN查询、分组等）。

**注意**: 目前项目中的 `models/backbone/pointops.py` 需要编译的 C++/CUDA 扩展。如果您还没有编译过这个扩展，使用 Point Transformer 时会遇到 `ModuleNotFoundError: No module named 'pointops_cuda'` 错误。

### 解决方案选项

有以下几种方案：

#### 选项 1: 编译 pointops CUDA 扩展

如果您有 Point Transformer 的原始代码仓库，通常会包含一个 `lib/pointops` 目录with `setup.py`：

```bash
cd /path/to/pointtransformer/lib/pointops
python setup.py install
```

#### 选项 2: 使用预编译的 Point Transformer

如果您已经安装了 Point Transformer 包，确保 `pointops_cuda` 在 Python 路径中可访问。

#### 选项 3: 临时禁用 Point Transformer（开发阶段）

如果暂时不需要使用 Point Transformer，可以只使用 PointNet2 或 PTv3：

```bash
# 使用 PointNet2
python train_lightning.py model.decoder.backbone=pointnet2

# 使用 PTv3
python train_lightning.py model.decoder.backbone=ptv3
```

## 使用方法

### 配置文件

Point Transformer 的配置文件位于：
```
config/model/diffuser/decoder/backbone/point_transformer.yaml
```

### 训练命令

编译 `pointops_cuda` 后，可以通过以下命令使用 Point Transformer：

```bash
# 使用 Point Transformer backbone
python train_lightning.py model.decoder.backbone=point_transformer

# 自定义参数
python train_lightning.py \
    model.decoder.backbone=point_transformer \
    model.decoder.backbone.c=6 \
    model.decoder.backbone.num_points=32768
```

### 配置示例

在 YAML 配置文件中：

```yaml
model:
  decoder:
    backbone:
      name: point_transformer
      c: 6  # xyz + rgb
      num_points: 32768
      out_dim: 512
```

## 接口说明

### 输入格式

- `(B, N, C)` tensor
  - `C=3`: xyz
  - `C=6`: xyz + rgb
  - `C=7`: xyz + rgb + mask

### 输出格式

- `xyz`: `(B, K, 3)` - 下采样后的点坐标
- `features`: `(B, 512, K)` - 点特征

### 属性

- `output_dim`: 512 (与 PointNet2 保持一致)

## 架构详情

Point Transformer 使用以下架构：
- 5 个编码器层，通道数为 [32, 64, 128, 256, 512]
- 每层包含 [2, 3, 4, 6, 3] 个 Transformer 块
- 使用最远点采样 (FPS) 进行下采样
- 每个 Transformer 层使用自注意力机制

## 测试

运行集成测试（需要先编译 pointops_cuda）：

```bash
python tests/test_point_transformer_backbone.py
```

测试包括：
1. 模型实例化测试
2. 不同输入维度的前向传播（xyz, xyz+rgb, xyz+rgb+mask）
3. 与 PointNet2 的接口兼容性测试
4. 批次大小一致性测试

## 已知问题

1. **依赖编译**: Point Transformer 需要编译 CUDA 扩展，这可能需要：
   - CUDA toolkit (与 PyTorch 版本匹配)
   - C++ 编译器
   - 适当的编译环境配置

2. **内存使用**: Point Transformer 的注意力机制可能比 PointNet2 消耗更多内存，建议：
   - 适当减小 batch size
   - 使用 gradient checkpointing（如果支持）
   - 监控 GPU 内存使用

## 性能对比

| Backbone | Output Dim | 参数量 | 内存 | 速度 |
|----------|-----------|--------|------|------|
| PointNet2 | 512 | ~5M | 低 | 快 |
| PTv3 | 512 | ~40M | 中 | 中 |
| Point Transformer | 512 | ~15M | 中高 | 中 |

## 相关文件

- 模型实现: `models/backbone/point_transformer.py`
- Wrapper 类: `models/backbone/point_transformer_backbone.py`
- 配置文件: `config/model/diffuser/decoder/backbone/point_transformer.yaml`
- 测试脚本: `tests/test_point_transformer_backbone.py`
- Backbone 注册: `models/backbone/__init__.py`

## 参考

- Point Transformer 论文: "Point Transformer" (ICCV 2021)
- 官方实现: https://github.com/POSTECH-CVLab/point-transformer

