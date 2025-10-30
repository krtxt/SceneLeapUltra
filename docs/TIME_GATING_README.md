# 时间门控注意力机制 - 快速开始

## 🎯 功能简介

时间门控机制（t-aware conditioning）通过动态调节条件强度来提升扩散模型性能：
- **扩散早期**（t≈0）：强条件约束，快速锁定方向
- **扩散后期**（t≈1）：弱条件约束，精细调整输出

## ✅ 实施状态

- ✓ 核心模块实现完成
- ✓ DiT 和 DiT-FM 集成完成
- ✓ 配置文件更新完成
- ✓ 测试验证通过
- ✓ 文档和可视化完成

## 🚀 快速使用

### 1. 启用时间门控

编辑配置文件 `config/model/diffuser/decoder/dit.yaml` 或 `config/model/flow_matching/decoder/dit_fm.yaml`：

```yaml
use_t_aware_conditioning: true  # 从 false 改为 true
```

### 2. 选择门控类型（可选）

使用余弦平方门控（默认，推荐）：
```yaml
t_gate:
  type: "cos2"  # 零参数，稳定
  apply_to: "both"  # 应用于场景和文本
  scene_scale: 1.0
  text_scale: 1.0
```

或使用可学习 MLP 门控（进阶）：
```yaml
t_gate:
  type: "mlp"  # 可学习
  mlp_hidden_dims: [256, 128]
  init_value: 1.0
  warmup_steps: 1000
```

### 3. Lightning 层集成（仅 DDPM 路径需要）

如果使用 DiT (DDPM)，需要在调用模型前添加：

```python
# 计算归一化时间
data['t_scalar'] = torch.tensor([current_step / (total_steps - 1)], 
                                device=device, dtype=torch.float32)

# 调用模型
output = model(x_t, ts, data)
```

**注意**：DiT-FM (Flow Matching) 无需修改，自动处理。

## 📊 验证测试

运行测试确保功能正常：

```bash
# 激活环境
source ~/.bashrc && conda activate DexGrasp

# 运行功能测试
python tests/test_time_gating.py

# 生成可视化图表
python tests/visualize_time_gating.py
```

测试通过标志：
```
============================================================
✓ 所有测试通过！
============================================================
```

## 📈 对比实验建议

按以下顺序进行实验：

1. **Baseline**（关闭门控）
   ```yaml
   use_t_aware_conditioning: false
   ```

2. **Cos2 both**（场景+文本）
   ```yaml
   use_t_aware_conditioning: true
   t_gate:
     type: "cos2"
     apply_to: "both"
   ```

3. **Cos2 scene only**（仅场景）
   ```yaml
   t_gate:
     type: "cos2"
     apply_to: "scene"
   ```

4. **不同缩放因子**
   ```yaml
   t_gate:
     scene_scale: 1.0
     text_scale: 0.6  # 文本条件较弱
   ```

## 📚 详细文档

- **使用指南**: `docs/time_aware_conditioning.md`
- **实施总结**: `docs/time_gating_implementation_summary.md`
- **可视化图表**: `docs/*.png`

## 🔧 核心文件

### 新增文件
- `models/decoder/time_gating.py` - 时间门控模块
- `tests/test_time_gating.py` - 功能测试
- `tests/visualize_time_gating.py` - 可视化脚本

### 修改文件
- `models/decoder/dit.py` - DiT 模型集成
- `models/decoder/dit_fm.py` - DiT-FM 模型集成
- `config/model/diffuser/decoder/dit.yaml` - DiT 配置
- `config/model/flow_matching/decoder/dit_fm.yaml` - DiT-FM 配置

## ⚠️ 注意事项

1. **默认关闭**：时间门控默认关闭，需要手动启用
2. **向后兼容**：关闭时与原模型行为完全一致
3. **DDPM 集成**：使用 DiT (DDPM) 需要在 Lightning 层添加 `t_scalar`
4. **FM 自动**：DiT-FM 无需额外修改

## 🎨 门控曲线示例

余弦平方门控在不同时间的门控值：

```
时间 t → 门控因子 α(t)
t=0.00 → α=1.0000 (强约束)
t=0.25 → α=0.8536
t=0.50 → α=0.5000
t=0.75 → α=0.1464
t=1.00 → α=0.0000 (弱约束)
```

详细曲线图见 `docs/cosine_squared_gate.png`

## ❓ 常见问题

**Q: 如何知道门控是否生效？**  
A: 在训练日志中应该能看到 "Time-aware conditioning enabled" 的信息。

**Q: 会影响推理速度吗？**  
A: 影响极小，余弦平方门控几乎无开销。

**Q: 可以只在训练时使用吗？**  
A: 不建议。训练和推理应使用相同配置。

**Q: 如何选择 scene_scale 和 text_scale？**  
A: 建议从 1.0 开始，根据实验结果调整。

## 📞 技术支持

如有问题，请参考：
1. 详细文档：`docs/time_aware_conditioning.md`
2. 实施总结：`docs/time_gating_implementation_summary.md`
3. 测试代码：`tests/test_time_gating.py`

---

**实施日期**: 2025-10-26  
**版本**: 1.0  
**状态**: ✓ 生产就绪

