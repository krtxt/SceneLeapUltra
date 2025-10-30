# 几何注意力偏置实施文档

## 概述

本文档记录了在DiT模型中引入"抓取-点云几何注意力偏置"（Geometric Attention Bias）的实施过程和使用方法。

## 实施动机

在scene cross-attention中加入基于相对几何关系的注意力偏置，以增强模型对grasp与点云空间关系的感知能力，提升定位精度和收敛速度。

参考：3DETR、V-DETR等模型在3D目标检测中通过引入3D相对位置编码/Vertex RPE实现了显著的性能提升。

## 核心公式

在scene cross-attention的打分机制中加入相对几何项：

$$
\text{score}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + w^\top \phi(R_i^\top(p_j - t_i))
$$

其中：
- $t_i, R_i$：第i个grasp token的平移和旋转
- $p_j$：第j个场景点的坐标
- $\phi(\cdot)$：MLP网络，将几何特征映射为标量偏置
- 输入特征：相对位置$\Delta x$、距离$||\Delta x||$、方向向量、法线夹角等

## 实施内容

### 1. 新增模块

#### `models/decoder/geometric_attention_bias.py`

核心模块，实现几何注意力偏置的计算：

- **`GeometricAttentionBias`类**：
  - 支持quat和r6d两种旋转表示
  - 可配置的几何特征类型（relative_pos、distance、direction、distance_log）
  - 多层MLP网络，输出per-head偏置

- **关键方法**：
  - `_rotation_repr_to_matrix()`: 旋转表示转换为旋转矩阵
  - `_quaternion_to_matrix()`: 四元数转旋转矩阵
  - `_r6d_to_matrix()`: 6D表示转旋转矩阵
  - `_compute_geometric_features()`: 计算几何特征
  - `forward()`: 计算attention bias

- **`extract_scene_xyz()`函数**：从data中提取场景点云xyz坐标

### 2. 修改的模块

#### `models/decoder/dit_memory_optimization.py`

- **`EfficientAttention`类**：
  - 新增`attention_bias`参数到`forward()`方法
  - 当提供bias时，自动切换到标准attention实现
  - 在`_standard_attention_forward()`中添加bias到scores

#### `models/decoder/dit.py`

- **`DiTBlock`类**：
  - 新增`use_geometric_bias`和`geometric_bias_module`参数
  - 在`forward()`中添加`grasp_poses`和`scene_xyz`参数
  - 在scene cross-attention前计算几何偏置并传递给attention

- **`DiTModel`类**：
  - 新增几何偏置配置参数
  - 初始化`GeometricAttentionBias`模块（如果启用）
  - 在forward中提取scene_xyz并传递给DiT blocks
  - 更新`_run_dit_blocks()`方法签名

#### `models/decoder/dit_fm.py`

- **`DiTFM`类**：
  - 与DiT类似的修改
  - 支持Flow Matching模式下的几何偏置

### 3. 配置文件更新

#### `config/model/diffuser/diffuser.yaml`

```yaml
# Geometric Attention Bias 配置
use_geometric_bias: false  # 是否启用几何注意力偏置
geometric_bias_hidden_dims: [128, 64]  # MLP隐藏层维度
geometric_bias_feature_types: ['relative_pos', 'distance']  # 特征类型
```

#### `config/model/flow_matching/decoder/dit_fm.yaml`

相同的配置项。

### 4. 测试脚本

#### `tests/test_geometric_attention_bias.py`

全面的单元测试，包括：
- 旋转表示转换正确性（quat、r6d）
- 几何特征计算正确性
- 前向传播测试
- 与attention机制的集成测试

**测试结果**：✅ 所有测试通过

#### `tests/demo_geometric_bias.py`

演示脚本，展示：
- 如何在DiT和DiT-FM中使用几何偏置
- 启用/禁用几何偏置的对比
- 详细的使用说明

## 使用方法

### 启用几何偏置

在配置文件中设置：

```yaml
use_geometric_bias: true
geometric_bias_hidden_dims: [128, 64]
geometric_bias_feature_types: ['relative_pos', 'distance']
```

或通过命令行覆盖：

```bash
python train_lightning.py model.decoder.use_geometric_bias=true
```

### 特征类型说明

- **`relative_pos`** (3D): 在grasp局部坐标系下的相对位置 $R_i^\top(p_j - t_i)$
- **`distance`** (1D): 欧氏距离 $||\Delta x||$
- **`direction`** (3D): 归一化方向向量 $\frac{\Delta x}{||\Delta x||}$
- **`distance_log`** (1D): 对数距离 $\log(||\Delta x|| + \epsilon)$

可以组合使用多种特征类型。

### 对比实验

1. **训练baseline（无几何偏置）**：
   ```bash
   python train_lightning.py model.decoder.use_geometric_bias=false \
       save_root=./experiments/baseline
   ```

2. **训练增强版（有几何偏置）**：
   ```bash
   python train_lightning.py model.decoder.use_geometric_bias=true \
       save_root=./experiments/with_geo_bias
   ```

3. **评估对比**：
   ```bash
   python test_lightning.py +checkpoint_path=experiments/baseline/checkpoints/best.ckpt
   python test_lightning.py +checkpoint_path=experiments/with_geo_bias/checkpoints/best.ckpt
   ```

## 技术细节

### 计算流程

1. **从grasp poses提取姿态**：
   - 前3维：平移向量 $t_i$
   - 后续维度：旋转表示（quat: 4维, r6d: 6维）

2. **转换旋转表示为旋转矩阵** $R_i$

3. **计算相对位置**：
   - 世界坐标系：$\Delta p = p_j - t_i$
   - 局部坐标系：$\Delta x = R_i^\top \Delta p$

4. **提取几何特征**：
   - 根据配置选择的特征类型计算

5. **通过MLP映射为偏置**：
   - 输入：几何特征向量
   - 输出：per-head标量偏置 (num_heads维)

6. **添加到attention scores**：
   - $\text{scores} = \text{scores}_{\text{QK}} + \text{bias}_{\text{geo}}$

### 内存和性能考虑

- **内存消耗**：几何偏置计算需要额外的 $O(B \times N_{\text{grasps}} \times N_{\text{points}} \times d_{\text{feature}})$ 内存
- **计算开销**：MLP前向传播和旋转矩阵计算
- **优化限制**：启用几何偏置时，会禁用Flash Attention和SDPA优化，回退到标准attention实现

**建议**：
- 对于大规模点云（>2048点），考虑下采样或增加batch size限制
- 使用较小的MLP隐藏层维度以减少开销
- 在较小数据集上先验证效果

## 设计原则

### 最小侵入性

- ✅ 新功能通过配置开关控制，默认关闭
- ✅ 不修改现有接口，只添加可选参数
- ✅ 保持向后兼容性
- ✅ 新增代码集中在独立模块中

### 模块化设计

- ✅ `GeometricAttentionBias`作为独立模块
- ✅ 清晰的职责分离
- ✅ 易于扩展和维护

### 可配置性

- ✅ 特征类型可配置
- ✅ MLP结构可配置
- ✅ 支持不同旋转表示

## 预期效果

根据3DETR、V-DETR等研究，引入几何注意力偏置可以带来：

1. **加速收敛**：模型更快地学习空间关系
2. **提升定位精度**：更准确的grasp位置预测
3. **增强泛化能力**：对不同场景的适应性更强

## 后续工作

可能的改进方向：

1. **优化计算效率**：
   - 实现CUDA kernel加速几何特征计算
   - 支持在SDPA中使用additive bias

2. **扩展特征类型**：
   - 添加与场景法线的夹角
   - 引入语义特征（如物体类别）

3. **自适应偏置**：
   - 学习特征权重而非固定组合
   - 引入注意力门控机制

4. **多尺度几何特征**：
   - 不同距离范围使用不同的特征表示
   - 层次化的几何建模

## 参考文献

1. Misra et al., "3DETR: An End-to-End Transformer Model for 3D Object Detection", ICCV 2021
2. Wang et al., "V-DETR: DETR with Vertex Relative Position Encoding for 3D Object Detection", NeurIPS 2022
3. Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019

## 版本历史

- **v1.0** (2025-10-25): 初始实现
  - 支持quat和r6d旋转表示
  - 4种几何特征类型
  - 完整的测试覆盖
  - DiT和DiT-FM双模型支持

