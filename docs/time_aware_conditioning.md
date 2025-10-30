# 时间相关的条件门控 (Time-aware Conditioning)

## 概述

时间门控机制（t-aware conditioning schedule）是一种根据扩散步数动态调节条件影响力的技术。核心思想是：
- **扩散早期**（噪声较大，t≈0）：施加**强条件约束**，帮助模型快速锁定大致方向
- **扩散后期**（接近生成结果，t≈1）：**减弱约束**，给模型更多自由度来精细调整输出

## 实现方式

通过引入时间门控因子 `α(t)` 来调制 cross-attention 的输出：
```
scene_attn_out = scene_attn_out * α_scene(t)
text_attn_out = text_attn_out * α_text(t)
```

其中 `α(t) ∈ [0, 1]` 是根据时间 t 计算的门控因子。

## 配置使用

### 1. 启用时间门控

在配置文件中（`dit.yaml` 或 `dit_fm.yaml`）设置：

```yaml
use_t_aware_conditioning: true  # 启用时间门控
```

### 2. 门控类型选择

#### 余弦平方门控（推荐用于baseline）

```yaml
t_gate:
  type: "cos2"  # 余弦平方函数: α(t) = cos²(π/2 * t)
  apply_to: "both"  # 应用于场景和文本
  scene_scale: 1.0  # 场景条件的缩放因子
  text_scale: 1.0   # 文本条件的缩放因子
```

**特性**：
- 零参数，无需训练
- 单调递减：t=0 时 α=1.0（强约束），t=1 时 α=0.0（无约束）
- 稳定可靠，作为baseline很好用

**门控曲线**：
```
t=0.00 -> α=1.0000 (100% 强度)
t=0.25 -> α=0.8536 (85% 强度)
t=0.50 -> α=0.5000 (50% 强度)
t=0.75 -> α=0.1464 (15% 强度)
t=1.00 -> α=0.0000 (0% 强度)
```

#### 可学习 MLP 门控（用于进阶实验）

```yaml
t_gate:
  type: "mlp"  # 通过MLP学习门控函数
  apply_to: "both"
  mlp_hidden_dims: [256, 128]  # MLP隐藏层维度
  init_value: 1.0  # 初始输出值
  warmup_steps: 1000  # 训练前N步固定输出为init_value
```

**特性**：
- 可学习，能适应数据分布
- 支持warmup机制，训练稳定
- 输出经sigmoid限制在[0, 1]

### 3. 应用范围控制

```yaml
t_gate:
  apply_to: "both"  # 可选: "both" | "scene" | "text"
```

- `"both"`: 同时应用于场景和文本条件（默认）
- `"scene"`: 仅应用于场景条件
- `"text"`: 仅应用于文本条件

### 4. 分别控制场景和文本强度

```yaml
t_gate:
  type: "cos2"
  apply_to: "both"
  scene_scale: 1.0   # 场景条件使用100%强度
  text_scale: 0.8    # 文本条件使用80%强度
  separate_text_gate: false  # 场景和文本共享门控函数，仅缩放不同
```

如果需要完全独立的门控：
```yaml
t_gate:
  separate_text_gate: true  # 场景和文本使用独立的门控模块
```

## 使用示例

### DiT (DDPM) 路径

在Lightning层调用DiT前，需要在data字典中添加`t_scalar`：

```python
# 假设当前步数为 current_step，总步数为 total_steps
data['t_scalar'] = torch.tensor([current_step / (total_steps - 1)]).to(device)

# 调用模型
output = model(x_t, ts, data)
```

**注意**：`t_scalar` 应该在 [0, 1] 范围内，其中：
- t=0: 扩散起始（噪声最大）
- t=1: 扩散结束（接近生成结果）

### DiT-FM (Flow Matching) 路径

DiT-FM会自动从`ts`中计算`t_scalar`，无需手动添加：

```python
# ts 已经是连续时间 [0, 1]
output = model(x_t, ts, data)  # t_scalar 会自动计算
```

## 对比实验

### Baseline（关闭时间门控）

```yaml
use_t_aware_conditioning: false
```

### 实验组（启用时间门控）

```yaml
use_t_aware_conditioning: true
t_gate:
  type: "cos2"
  apply_to: "both"
  scene_scale: 1.0
  text_scale: 1.0
```

通过修改配置文件即可轻松切换，无需改动代码。

## 测试验证

运行测试脚本验证功能：

```bash
source ~/.bashrc && conda activate DexGrasp
python tests/test_time_gating.py
```

测试内容包括：
- 余弦平方门控的输出范围和单调性
- MLP门控的warmup机制
- 不同配置下的门控行为
- 与DiTBlock的集成测试

## 调试建议

### 1. 检查门控因子

在模型中添加日志输出：

```python
if self.time_gate is not None:
    alpha_scene = self.time_gate.get_scene_gate(t=t_scalar, time_emb=time_emb)
    print(f"t_scalar: {t_scalar}, alpha_scene: {alpha_scene.squeeze()}")
```

### 2. 可视化门控曲线

```python
import matplotlib.pyplot as plt
import numpy as np

t_values = np.linspace(0, 1, 100)
alpha_values = [np.cos(np.pi / 2 * t) ** 2 for t in t_values]

plt.plot(t_values, alpha_values)
plt.xlabel('Time t')
plt.ylabel('Gate α(t)')
plt.title('Cosine Squared Gate Curve')
plt.grid(True)
plt.savefig('gate_curve.png')
```

### 3. 对比实验建议

建议按以下顺序进行实验：

1. **Baseline**: `use_t_aware_conditioning: false`
2. **Cos2 both**: `type: cos2, apply_to: both`
3. **Cos2 scene only**: `type: cos2, apply_to: scene`
4. **Cos2 different scales**: `scene_scale: 1.0, text_scale: 0.5`
5. **MLP learnable**: `type: mlp` (可选)

## 常见问题

### Q: 为什么默认关闭时间门控？
A: 为了保持与现有实验的兼容性，避免影响已有的训练结果。需要手动启用才能使用。

### Q: DDPM路径下，如何知道总步数？
A: 总步数在采样器/scheduler中配置，Lightning层应该能够访问。如果不确定，可以使用固定值（如1000）或从配置中读取。

### Q: 启用时间门控会影响推理速度吗？
A: 影响很小。余弦平方门控只是简单的数学运算，MLP门控增加的计算量也很少（相比DiT本身可以忽略不计）。

### Q: 可以只在训练时使用，推理时关闭吗？
A: 不建议。如果训练时使用了时间门控，推理时也应该使用相同配置，否则可能导致性能下降。

### Q: 如何选择scene_scale和text_scale？
A: 建议从1.0开始，如果发现某个条件过强或过弱，可以尝试调整。通常场景条件更重要，可以保持1.0；文本条件可以尝试0.5-0.8。

## 参考资料

- 扩散模型中的条件引导：Classifier-Free Guidance
- 时间相关的采样策略：SNR-aware sampling
- 自适应条件强度：Adaptive conditioning strength

## 更新日志

- **2025-10-26**: 初始版本实现
  - 支持余弦平方门控和MLP门控
  - 完整的DiT和DiT-FM集成
  - 灵活的配置选项

