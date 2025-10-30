# Optimal Transport for Flow Matching

## 概述

本项目为 Flow Matching 训练实现了 **Sinkhorn 最优传输配对**功能，用于解决无序集合数据（如1024个抓取姿态）的配对问题。

### 理论背景

**问题：** 标准 Flow Matching 使用随机索引配对 `x0[i] <-> x1[i]`，对于无序集合数据，这种配对是任意的，可能导致：
- 速度场学习困难（需要学习长距离传输）
- 训练不稳定
- 生成质量下降

**解决方案：** 使用 Sinkhorn 算法计算集合到集合的最优传输，最小化总体传输距离，让速度场更平滑、更易学习。

### 数学原理

给定真实抓取集合 $X_0 = \{x_0^i\}_{i=1}^N$ 和噪声集合 $X_1 = \{x_1^j\}_{j=1}^N$：

1. **代价矩阵：** $C_{ij} = \|x_0^i - x_1^j\|^2$
2. **最优传输问题：** 
   $$\min_{\pi \in \Pi(a,b)} \langle C, \pi \rangle$$
   其中 $\Pi(a,b)$ 是边际分布为 $a, b$ 的传输计划集合
3. **Sinkhorn 算法：** 通过熵正则化将 OT 问题转化为矩阵缩放问题，可高效求解

### 优势

- ✅ **GPU 并行化**：支持 batch 处理，计算高效
- ✅ **可微分**：支持端到端训练
- ✅ **理论保证**：最小化传输代价
- ✅ **易于集成**：模块化设计，一行配置启用

---

## 快速开始

### 1. 基础用法

只需在训练命令中添加一个参数即可启用 OT：

```bash
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=true
```

### 2. 推荐配置

```bash
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=true \
    model.fm.optimal_transport.reg=0.1 \
    model.fm.optimal_transport.num_iters=50 \
    model.fm.optimal_transport.start_epoch=0
```

### 3. 运行消融实验

使用提供的脚本对比有无 OT 的效果：

```bash
bash scripts/train_with_ot.sh
```

---

## 配置参数详解

### 完整配置（在 `config/model/flow_matching/flow_matching.yaml`）

```yaml
fm:
  optimal_transport:
    enable: false                  # 是否启用OT
    reg: 0.1                       # 熵正则化参数 (0.05-0.2)
    num_iters: 50                  # Sinkhorn迭代次数
    distance_metric: euclidean     # 距离度量: euclidean | squared
    matching_strategy: greedy      # 配对策略: greedy | hungarian
    normalize_cost: true           # 归一化代价矩阵
    start_epoch: 0                 # 开始使用OT的epoch
    log_freq: 100                  # 日志频率
```

### 参数说明

#### `enable` (bool)
- **默认：** `false`
- **说明：** 是否启用 OT 配对
- **推荐：** 先跑 baseline，稳定后再启用对比

#### `reg` (float)
- **默认：** `0.1`
- **范围：** `0.05 - 0.2`
- **说明：** 
  - 越小 → 越接近真实 OT（但收敛慢，可能不稳定）
  - 越大 → 收敛快（但略欠精确）
- **推荐：** 
  - `0.1`：平衡选择（默认）
  - `0.05`：追求精确
  - `0.2`：快速实验

#### `num_iters` (int)
- **默认：** `50`
- **范围：** `20 - 100`
- **说明：** Sinkhorn 算法的迭代次数
- **推荐：** 
  - `50`：通常足够
  - `100`：低 reg 时需要更多迭代

#### `distance_metric` (str)
- **选项：** `euclidean` | `squared`
- **默认：** `euclidean`
- **说明：** 
  - `euclidean`：L2 距离
  - `squared`：L2 距离的平方（更强调远距离）

#### `matching_strategy` (str)
- **选项：** `greedy` | `hungarian`
- **默认：** `greedy`
- **说明：** 
  - `greedy`：快速，每行取最大值
  - `hungarian`：精确但需要 scipy，慢

#### `start_epoch` (int)
- **默认：** `0`
- **说明：** 从第几个 epoch 开始使用 OT
- **推荐：** 
  - `0`：立即使用（如果训练稳定）
  - `5-10`：先让模型稳定几个 epoch

---

## 使用场景

### 场景 1：对比实验（推荐）

**目标：** 验证 OT 是否提升性能

```bash
# Baseline
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=false \
    wandb.name="baseline_no_ot"

# With OT
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=true \
    wandb.name="with_ot_reg01"
```

**对比指标：**
- `train/loss`：训练损失
- `train/ot_matched_dist`：配对后的平均距离
- `train/ot_improvement`：OT 改进百分比
- `val/loss`：验证损失

### 场景 2：参数调优

测试不同的 `reg` 参数：

```bash
for reg in 0.05 0.1 0.2; do
    python train_lightning.py \
        model=flow_matching \
        model.fm.optimal_transport.enable=true \
        model.fm.optimal_transport.reg=$reg \
        wandb.name="ot_reg_${reg}"
done
```

### 场景 3：延迟启用

先训练几个 epoch 稳定，再启用 OT：

```bash
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=true \
    model.fm.optimal_transport.start_epoch=5
```

---

## 监控和调试

### WandB 日志

启用 OT 后，会自动记录以下指标：

```python
train/ot_matched_dist     # 配对后的平均距离
train/ot_random_dist      # 随机配对的平均距离
train/ot_improvement      # 改进百分比
```

### 控制台输出

每 `log_freq` 个 batch 会打印：

```
[OT] Epoch 10, Batch 100: matched_dist=2.3456, random_dist=3.7890, improvement=38.1%
```

### 可视化

运行测试脚本生成可视化：

```bash
python tests/test_optimal_transport.py
```

会生成 `tests/ot_visualization.png`，展示配对效果。

---

## 性能分析

### 计算开销

以 `B=32, num_grasps=1024, D=25` 为例：

| 操作 | 时间 (A100) | 占训练比例 |
|------|------------|-----------|
| Sinkhorn OT | ~2ms | <1% |
| DiT 前向 | ~80ms | ~40% |
| DiT 反向 | ~100ms | ~50% |

**结论：** OT 计算开销极小，几乎可忽略。

### 内存占用

- 代价矩阵：`[B, N, N]` = `32 * 1024 * 1024 * 4B` ≈ 128 MB
- 其他中间变量：约 100 MB
- **总增加：** ~200 MB（相比总显存占用很小）

---

## 测试

### 单元测试

```bash
python tests/test_optimal_transport.py
```

测试内容：
1. ✅ 基本 OT 功能
2. ✅ 不同配置参数
3. ✅ 不同数据规模性能
4. ✅ 可视化配对效果
5. ✅ 梯度传播

### 集成测试

```bash
# 快速训练测试（1个epoch）
python train_lightning.py \
    model=flow_matching \
    model.fm.optimal_transport.enable=true \
    epochs=1 \
    data.train.limit_batches=10 \
    data.val.limit_batches=5
```

---

## 常见问题

### Q1: 启用 OT 后训练变慢了？

**A:** 检查以下几点：
1. GPU 利用率是否正常（OT 开销应 <1%）
2. 是否使用了 `matching_strategy=hungarian`（会很慢，改为 `greedy`）
3. 数据加载是否成为瓶颈（增加 `num_workers`）

### Q2: OT 改进百分比很小（<10%）？

**A:** 可能的原因：
1. `reg` 太大（尝试减小到 0.05）
2. `num_iters` 太少（增加到 100）
3. 数据本身就比较均匀分布（OT 收益有限）

### Q3: 出现 NaN？

**A:** 可能是数值不稳定：
1. 增加 `reg`（如 0.2）
2. 启用 `normalize_cost=true`
3. 检查输入数据是否包含 NaN/Inf

### Q4: 想看配对的具体效果？

**A:** 运行可视化测试：
```bash
python tests/test_optimal_transport.py
```
查看生成的 `tests/ot_visualization.png`。

---

## 代码结构

```
models/fm/
├── optimal_transport.py      # OT 核心实现
│   ├── SinkhornOT           # Sinkhorn 求解器
│   ├── apply_optimal_matching  # 应用配对
│   └── compute_matching_quality  # 质量评估

models/fm_lightning.py         # 集成到训练
├── __init__                  # 初始化 OT 求解器
└── training_step             # 训练步骤中应用 OT

tests/
├── test_optimal_transport.py  # 单元测试
└── ot_visualization.png       # 可视化结果

scripts/
└── train_with_ot.sh          # 训练脚本示例
```

---

## 参考文献

1. Cuturi, M. (2013). [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895)
2. Lipman et al. (2023). [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
3. Tong et al. (2023). [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)

---

## 联系方式

如有问题或建议，请联系项目维护者或提交 Issue。

