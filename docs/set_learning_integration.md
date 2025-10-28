# 抓取集合学习损失与指标集成文档

## 概述

本文档描述了为 DDPM/Flow Matching 项目集成的抓取集合学习功能，包括集合损失（Set Losses）和集合指标（Set Metrics）。

## 实现的功能

### 1. 核心距离函数 (`models/loss/set_distance.py`)

实现了抓取元素距离函数 \( c(g, \tilde{g}) \)，用于所有集合损失和指标的计算。

**距离公式：**

```
c(g, g') = α_t * ||t - t'||_2 / s_obj
           + α_R * d_SO(3)(R, R')
           + α_q * ||(q - q') / range(q)||_2
           + α_sym * min_{S∈S} d_SO(3)(R, R'S)
           + α_cnt * d_contact(g, g')
```

**主要组件：**
- **平移距离**: L2 距离，可选物体尺度归一化
- **旋转距离**: SO(3) 上的测地距离
  - 对于 quaternion: `d = 2 * arccos(|<q1, q2>|)`
  - 对于 rot6d: 转换为旋转矩阵后计算 `arccos((trace(R1^T R2) - 1) / 2)`
- **关节角度距离**: 归一化的 L2 距离
- **对称性距离**: 考虑物体对称性（可选，当前为占位符）
- **接触距离**: 接触点/法向一致性（可选，当前为占位符）

**关键函数：**
- `GraspSetDistance`: 距离计算模块
- `extract_grasp_components()`: 从归一化姿态提取组件
- `compute_pairwise_grasp_distance()`: 计算成对距离矩阵

### 2. 集合损失 (`models/loss/set_losses.py`)

实现了四种集合学习损失：

#### 2.1 Sinkhorn Optimal Transport Loss (`SinkhornOTLoss`)

使用 Sinkhorn 迭代算法实现的不平衡最优传输损失，用于对齐预测和目标抓取集合的分布。

**参数：**
- `epsilon`: 熵正则化参数（默认: 0.1）
- `tau`: 不平衡正则化参数（默认: 1.0）
- `max_iter`: 最大 Sinkhorn 迭代次数（默认: 100）
- `threshold`: 收敛阈值（默认: 1e-3）
- `max_samples`: 最大样本数（子采样以提高效率）（默认: 256）

#### 2.2 Chamfer Distance Loss (`ChamferDistanceLoss`)

简单的集合间双向最小距离度量。

**公式：**
```
L_CD = (1/N) Σ min_j C_ij + (1/M) Σ min_i C_ij
```

#### 2.3 Repulsion Loss (`RepulsionLoss`)

多样性正则化，惩罚生成集合内部过于相似的抓取。

**方法：** kNN-Riesz 斥力
**公式：**
```
L_rep = (1/N) Σ_i Σ_{j∈kNN(i)} λ / (δ + ||z_i - z_j||^2)^(s/2)
```

**参数：**
- `k`: 最近邻数量（默认: 5）
- `lambda_repulsion`: 斥力强度（默认: 1.0）
- `delta`: 避免奇异性的正则化项（默认: 0.01）
- `s`: Riesz 参数（默认: 2.0）

#### 2.4 Physics Feasibility Loss (`PhysicsFeasibilityLoss`)

物理可行性约束，惩罚碰撞/穿透和接触不一致（当前部分实现）。

**参数：**
- `penetration_weight`: 穿透惩罚权重（默认: 1.0）
- `contact_weight`: 接触一致性权重（默认: 0.5）
- `penetration_threshold`: 穿透阈值（默认: 0.005）

#### 时间步调度

集合损失支持时间步依赖的权重调度：

**调度类型：**
- `constant`: 恒定权重
- `linear`: 线性增长 `λ_set(t) = t_norm * final_weight`
- `cosine`: 余弦调度 `λ_set(t) = 0.5 * (1 + cos(π * (1 - t_norm))) * final_weight`
- `quadratic`: 二次增长 `λ_set(t) = t_norm^2 * final_weight`

### 3. 集合指标 (`models/loss/set_metrics.py`)

实现了四种验证指标：

#### 3.1 Coverage (COV) (`CoverageMetric`)

衡量真值抓取被生成抓取覆盖的比例。

**公式：**
```
COV_τ = (1/M) Σ_j 1[min_i c(g_i, g'_j) ≤ τ]
```

**参数：**
- `thresholds`: 距离阈值列表（默认: [0.05, 0.1, 0.2]）

#### 3.2 Minimum Matching Distance (MMD) (`MinimumMatchingDistanceMetric`)

衡量真值抓取到最近生成抓取的平均距离（保真度）。

**公式：**
```
MMD = (1/M) Σ_j min_i c(g_i, g'_j)
```

#### 3.3 Diversity Metrics (`DiversityMetric`)

基于最近邻距离（NND）的统计量衡量生成集合的多样性。

**指标：**
- `mean_nnd`: 平均最近邻距离
- `std_nnd`: 最近邻距离标准差
- `cv_nnd`: 变异系数 (CV = std/mean) - 越小越均匀

#### 3.4 Precision/Recall (`PrecisionRecallMetric`)

集合的精确率和召回率（可选）。

**定义：**
- Precision: 生成抓取中有多少接近任何真值
- Recall: 真值抓取中有多少有接近的生成抓取

## 配置说明

### 配置文件结构

在 `config/model/diffuser/criterion/loss_standardized.yaml` 中添加了三个新配置块：

#### 1. `set_distance` - 距离函数配置

```yaml
set_distance:
  alpha_translation: 1.0      # 平移距离权重
  alpha_rotation: 1.0         # 旋转距离权重
  alpha_qpos: 0.5             # 关节角度距离权重
  alpha_symmetry: 0.0         # 对称性权重（0=禁用）
  alpha_contact: 0.0          # 接触距离权重（0=禁用）
  
  rot_type: ${rot_type}       # 旋转表示类型
  normalize_translation: true # 是否归一化平移
  qpos_range: [0.0, 1.0]      # 关节角度范围
  use_symmetry: false         # 启用对称性感知
```

#### 2. `set_loss` - 集合损失配置

```yaml
set_loss:
  enabled: false              # 总开关
  
  # 损失权重
  lambda_ot: 1.0              # OT 损失权重
  gamma_cd: 0.0               # Chamfer 距离权重
  eta_repulsion: 0.1          # 斥力（多样性）权重
  zeta_physics: 0.0           # 物理可行性权重
  
  # 时间步调度
  schedule_type: "constant"   # 选项: constant, linear, cosine, quadratic
  final_weight: 1.0           # t=0 时的最终权重
  
  # OT 配置
  ot_config:
    epsilon: 0.1
    tau: 1.0
    max_iter: 100
    threshold: 1e-3
    max_samples: 256
  
  # 斥力配置
  repulsion_config:
    k: 5
    lambda_repulsion: 1.0
    delta: 0.01
    s: 2.0
  
  # 物理配置
  physics_config:
    penetration_weight: 1.0
    contact_weight: 0.5
    penetration_threshold: 0.005
```

#### 3. `set_metrics` - 集合指标配置

```yaml
set_metrics:
  enabled: false              # 总开关
  
  # 指标开关
  compute_coverage: true      # 计算覆盖率
  compute_mmd: true           # 计算 MMD
  compute_diversity: true     # 计算多样性
  compute_precision_recall: false  # 计算精确率/召回率（可选）
  
  # 覆盖率配置
  coverage_config:
    thresholds: [0.05, 0.1, 0.2]
  
  # Precision/Recall 配置
  pr_config:
    threshold: 0.1
```

### 启用集合学习

要启用集合学习功能，修改配置文件：

```yaml
set_loss:
  enabled: true               # 启用集合损失
  lambda_ot: 1.0              # 启用 OT 损失
  gamma_cd: 1.0               # 启用 Chamfer 损失
  eta_repulsion: 0.1          # 启用斥力损失

set_metrics:
  enabled: true               # 启用集合指标
```

## 代码集成

### 1. `GraspLossPose` 修改

在 `models/loss/grasp_loss_pose.py` 中添加了：

**初始化：**
```python
# Set-based grasp learning configuration
self.set_distance_cfg = getattr(loss_cfg, 'set_distance', None)
self.set_loss_cfg = getattr(loss_cfg, 'set_loss', None)
self.set_metrics_cfg = getattr(loss_cfg, 'set_metrics', None)

self.use_set_losses = (self.set_loss_cfg is not None and 
                       getattr(self.set_loss_cfg, 'enabled', False))
self.use_set_metrics = (self.set_metrics_cfg is not None and 
                        getattr(self.set_metrics_cfg, 'enabled', False))
```

**新方法：**
- `_calculate_set_losses()`: 计算集合损失
- `_calculate_set_metrics()`: 计算集合指标

**集成点：**
- 训练损失：在 `_calculate_losses()` 末尾添加集合损失
- 验证指标：在 `_forward_val()` 中计算并返回集合指标

### 2. Lightning 模块修改

**DDPM** (`models/diffuser_lightning.py`) 和 **Flow Matching** (`models/fm_lightning.py`):

在 `validation_step()` 中添加：
```python
# Extract set-based metrics if present
set_metrics = loss_dict.pop('_set_metrics', {})

# Log set-based metrics if present
if set_metrics:
    for metric_name, metric_value in set_metrics.items():
        self.log(f"val/set_{metric_name}", metric_value, 
                 prog_bar=False, batch_size=batch_size, sync_dist=True)
```

## 使用示例

### 训练时启用集合损失

编辑 `config/model/diffuser/criterion/loss_standardized.yaml`：

```yaml
set_loss:
  enabled: true
  lambda_ot: 1.0
  gamma_cd: 0.5
  eta_repulsion: 0.1
  schedule_type: "cosine"
  final_weight: 2.0
```

然后正常训练：
```bash
./train_distributed.sh --gpus 4
```

### 验证时查看集合指标

启用集合指标后，验证日志将包含：
- `val/set_coverage@0.05`
- `val/set_coverage@0.1`
- `val/set_coverage@0.2`
- `val/set_mmd`
- `val/set_mean_nnd`
- `val/set_std_nnd`
- `val/set_cv_nnd`

在 WandB 中可以实时查看这些指标。

## 测试

运行测试脚本验证实现：

```bash
cd /home/engine/project
python tests/test_set_learning_direct.py
```

测试包括：
1. 距离计算模块测试
2. 集合损失模块测试（OT, Chamfer, Repulsion）
3. 集合指标模块测试（Coverage, MMD, Diversity）
4. 集成测试

## 日志输出

实现包含充分的日志输出用于调试：

**INFO 级别：**
- 初始化信息（权重、参数）
- 计算结果摘要（损失值、指标值）

**DEBUG 级别：**
- 中间计算步骤
- 距离矩阵统计
- Sinkhorn 收敛信息

**ERROR 级别：**
- 异常捕获和错误信息
- 计算失败时的详细堆栈

## 注意事项

### 1. 姿态格式

抓取姿态格式：`[translation (3), qpos (16), rotation (rot_dim)]`

- `rot6d`: pose_dim = 3 + 16 + 6 = **25**
- `quat`: pose_dim = 3 + 16 + 4 = **23**
- `euler`: pose_dim = 3 + 16 + 3 = **22**

### 2. 性能考虑

- OT 损失在 N 或 M > 256 时会自动子采样
- 可通过 `max_samples` 参数调整
- 对于大规模抓取集合（N, M > 100），考虑：
  - 降低 OT 迭代次数（`max_iter`）
  - 增大 epsilon（加快收敛）
  - 禁用不需要的损失/指标

### 3. 多抓取要求

集合学习功能需要多抓取数据：
- 输入张量必须是 3D: `[B, num_grasps, pose_dim]`
- 单抓取模式（2D）下会跳过集合损失和指标
- 日志会输出 DEBUG 信息说明跳过原因

### 4. 当前限制

- 对称性处理为占位符（返回零距离）
- 接触距离为占位符（返回零距离）
- 物理损失部分实现（穿透可用，接触待实现）

## 未来扩展

可能的改进方向：

1. **对称性支持**：
   - 实现完整的对称群处理
   - 支持从数据集加载对称矩阵

2. **接触距离**：
   - 实现基于接触点的距离
   - 添加法向一致性检查

3. **物理损失增强**：
   - 完整的碰撞检测
   - 接触质量评估
   - 力闭合检查

4. **其他指标**：
   - Sliced Wasserstein Distance
   - Precision-Recall for Distributions (PRD)
   - Fréchet Grasp Distance (FGD)
   - DPP 对数似然
   - Success@K / Recall@K

5. **性能优化**：
   - GPU 加速 Sinkhorn
   - 更高效的距离计算
   - 批处理优化

## 参考文献

- Lipman et al. (2023): Flow Matching for Generative Modeling
- Cuturi (2013): Sinkhorn Distances: Lightspeed Computation of Optimal Transport
- Chizat et al. (2018): Scaling Algorithms for Unbalanced Optimal Transport Problems
- Zhou et al. (2019): On the Continuity of Rotation Representations in Neural Networks

## 维护者

实现日期: 2025-10-26
版本: 1.0
