# 抓取集合学习快速入门

## 简介

本项目现在支持**抓取集合学习**（Set-based Grasp Learning），可以一次性生成多个抓取姿态并优化整个集合的质量、覆盖率和多样性。

## 快速开始

### 1. 启用集合学习

编辑你的配置文件（或使用提供的示例配置）：

```bash
# 使用带有集合学习的配置
cp config/model/diffuser/criterion/loss_with_set_learning.yaml \
   config/model/diffuser/criterion/my_experiment.yaml
```

关键配置项：

```yaml
set_loss:
  enabled: true               # 启用集合损失
  lambda_ot: 0.5              # OT 损失权重
  gamma_cd: 0.5               # Chamfer 损失权重
  eta_repulsion: 0.05         # 多样性正则权重

set_metrics:
  enabled: true               # 启用集合指标
```

### 2. 运行训练

```bash
./train_distributed.sh --gpus 4
```

### 3. 查看结果

在 WandB 中查看新的指标：
- `train/ot_loss`: 最优传输损失
- `train/chamfer_loss`: Chamfer 距离损失
- `train/repulsion_loss`: 多样性斥力损失
- `val/set_coverage@0.1`: 覆盖率（阈值 0.1）
- `val/set_mmd`: 最小匹配距离
- `val/set_mean_nnd`: 平均最近邻距离（多样性）
- `val/set_cv_nnd`: 变异系数（越小越均匀）

## 功能说明

### 集合损失 (Set Losses)

1. **Optimal Transport Loss** (`lambda_ot`)
   - 对齐预测和目标抓取集合的分布
   - 使用 Sinkhorn 算法求解
   - 适合优化整体覆盖率

2. **Chamfer Distance Loss** (`gamma_cd`)
   - 简单的双向最小距离
   - 计算更快，适合快速迭代
   - 可与 OT 损失结合使用

3. **Repulsion Loss** (`eta_repulsion`)
   - 促进生成抓取的多样性
   - 惩罚过于相似的抓取
   - 提高探索能力

4. **Physics Loss** (`zeta_physics`)
   - 物理可行性约束
   - 惩罚碰撞和穿透
   - 当前部分实现

### 集合指标 (Set Metrics)

验证阶段自动计算：

1. **Coverage (COV)**
   - 真值抓取被覆盖的比例
   - 在多个阈值下评估

2. **Minimum Matching Distance (MMD)**
   - 真值到预测的平均最小距离
   - 衡量保真度

3. **Diversity Metrics**
   - 平均最近邻距离
   - 变异系数 (CV)
   - 衡量生成集合的均匀性

## 配置建议

### 初次尝试

```yaml
set_loss:
  enabled: true
  lambda_ot: 0.5              # 中等权重
  gamma_cd: 0.0               # 先禁用
  eta_repulsion: 0.05         # 小权重
  schedule_type: "constant"   # 恒定权重
```

### 优化覆盖率

```yaml
set_loss:
  enabled: true
  lambda_ot: 1.0              # 提高 OT 权重
  gamma_cd: 0.5
  eta_repulsion: 0.02         # 降低多样性惩罚
  schedule_type: "cosine"     # 渐进加权
```

### 优化多样性

```yaml
set_loss:
  enabled: true
  lambda_ot: 0.3
  gamma_cd: 0.3
  eta_repulsion: 0.2          # 提高多样性惩罚
  
  repulsion_config:
    k: 8                      # 考虑更多邻居
    lambda_repulsion: 2.0     # 更强的斥力
```

### 平衡质量与多样性

```yaml
set_loss:
  enabled: true
  lambda_ot: 0.8
  gamma_cd: 0.5
  eta_repulsion: 0.1
  schedule_type: "cosine"
  final_weight: 2.0           # 后期加强集合损失
```

## 时间步调度

集合损失支持随训练进度调整权重：

```yaml
set_loss:
  schedule_type: "cosine"     # 选项: constant, linear, cosine, quadratic
  final_weight: 2.0           # t=0 时的最终权重
```

- `constant`: 始终使用相同权重
- `linear`: 线性增长（适合稳定训练）
- `cosine`: 余弦调度（推荐，平滑过渡）
- `quadratic`: 二次增长（后期快速增强）

## 性能优化

对于大规模抓取集合（N > 100）：

```yaml
set_loss:
  ot_config:
    max_samples: 64           # 降低子采样大小
    max_iter: 50              # 减少迭代次数
    epsilon: 0.2              # 增大 epsilon 加快收敛
```

## 常见问题

### Q1: 训练变慢了？

集合损失会增加计算开销。优化方法：
- 降低 `max_samples`（OT 子采样大小）
- 减少 `max_iter`（Sinkhorn 迭代次数）
- 禁用不需要的损失项

### Q2: 损失不收敛？

尝试：
- 降低集合损失权重（从 0.1 开始）
- 使用 `schedule_type: "cosine"` 渐进加权
- 检查数据是否为多抓取格式（3D张量）

### Q3: 如何调整权重？

建议顺序：
1. 先只启用 OT 或 Chamfer，权重 0.5
2. 验证损失正常下降
3. 逐步添加 Repulsion，权重 0.05-0.1
4. 根据验证指标微调

### Q4: 指标没有显示？

检查：
- `set_metrics.enabled: true`
- 数据是否为多抓取格式（[B, N, D]）
- 日志中是否有错误信息

## 测试

验证实现：

```bash
cd /home/engine/project
python tests/test_set_learning_direct.py
```

应该看到：
```
✓ Distance matrix: torch.Size([2, 8, 12])
✓ OT Loss: 0.8969
✓ Chamfer Loss: 7.3912
✓ Repulsion Loss: 0.0249
✓ Coverage: {'coverage@0.1': 0.0, 'coverage@0.2': 0.0}
✓ MMD: {'mmd': 3.7326}
✓ Diversity: {'mean_nnd': 6.029, 'std_nnd': 0.512, 'cv_nnd': 0.085}
✓ ALL TESTS PASSED
```

## 详细文档

完整文档请参考：`docs/set_learning_integration.md`

## 支持

如有问题，请查看：
1. 日志输出（包含 DEBUG 级别的详细信息）
2. WandB 面板中的损失曲线
3. 配置文件是否正确

## 示例实验

### 基线（不使用集合学习）

```bash
./train_distributed.sh --gpus 4
```

### 启用集合学习

```bash
./train_distributed.sh --gpus 4 \
  +criterion=diffuser/criterion/loss_with_set_learning
```

对比两个实验的验证指标即可看到集合学习的效果！
