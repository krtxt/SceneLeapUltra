# PTv3 Quick Reference Card

## 🚀 快速开始

### 1. 验证集成
```bash
python tests/verify_ptv3_integration.py
```

### 2. 运行测试
```bash
# 基础测试
python tests/test_ptv3_backbone_basic.py

# DiT 冒烟测试
python tests/test_smoke_dit_ptv3.py

# UNet 冒烟测试
python tests/test_smoke_unet_ptv3.py
```

### 3. 开始训练
```bash
# DiT + PTv3 Light (推荐)
python train_lightning.py \
    model/diffuser/decoder=dit \
    model/diffuser/decoder/backbone=ptv3_light

# UNet + PTv3
python train_lightning.py \
    model/diffuser/decoder=unet \
    model/diffuser/decoder/backbone=ptv3
```

## 📊 配置选择

| 配置 | 参数量 | 输出维度 | 速度 | Flash Attn | 推荐场景 |
|------|--------|---------|------|------------|----------|
| `ptv3_light` | 8-12M | 512 | 中速 | ✅ | **推荐首选** |
| `ptv3` | 46M | 512 | 慢 | ✅ | 高精度需求 |
| `ptv3_no_flash` | 46M | 512 | 更慢 | ❌ | 兼容性需求 |

## 🔧 配置文件路径

```
config/model/diffuser/decoder/backbone/
├── ptv3_light.yaml       # 轻量级（推荐）
├── ptv3.yaml            # 标准版
└── ptv3_no_flash.yaml   # 无Flash版
```

## 📦 依赖安装

### 必需
```bash
pip install spconv-cu118  # 替换为你的CUDA版本
pip install torch-scatter
pip install addict
```

### 可选（Flash Attention）
```bash
pip install flash-attn
```

## 🎯 支持的输入格式

| use_rgb | use_object_mask | 输入形状 | 说明 |
|---------|----------------|---------|------|
| ❌ | ❌ | (B,N,3) | 仅 xyz |
| ✅ | ❌ | (B,N,6) | xyz + rgb |
| ❌ | ✅ | (B,N,4) | xyz + mask |
| ✅ | ✅ | (B,N,7) | xyz + rgb + mask |

## 🐛 常见问题

### ImportError: No module named 'flash_attn'
**解决**: 使用 `ptv3_no_flash` 或安装 flash-attn
```bash
pip install flash-attn
```

### CUDA out of memory
**解决**:
1. 使用 `ptv3_light` 而非 `ptv3`
2. 减小 batch size
3. 减少点云数量

### spconv import error
**解决**: 安装对应CUDA版本的spconv
```bash
# CUDA 11.8
pip install spconv-cu118

# CUDA 11.7
pip install spconv-cu117
```

## 📝 配置覆盖示例

### 命令行覆盖
```bash
# 切换backbone
python train_lightning.py \
    model/diffuser/decoder/backbone=ptv3_light

# 调整grid_size
python train_lightning.py \
    model/diffuser/decoder/backbone=ptv3_light \
    model/diffuser/decoder/backbone.grid_size=0.05
```

### 配置文件覆盖
```yaml
# config/experiment/my_experiment.yaml
defaults:
  - /model/diffuser/decoder: dit
  - /model/diffuser/decoder/backbone: ptv3_light

model:
  diffuser:
    decoder:
      backbone:
        grid_size: 0.05  # 覆盖默认值
```

## 🔍 性能监控

### 检查显存占用
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### 分析推理速度
```python
import time
import torch

model.eval()
with torch.no_grad():
    start = time.time()
    output = model(x_t, ts, data)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Inference time: {elapsed*1000:.2f} ms")
```

## 📚 相关文档

- 详细文档: `models/backbone/PTv3_INTEGRATION.md`
- 集成总结: `PTv3_INTEGRATION_SUMMARY.md`
- 变更日志: `CHANGELOG_PTv3.md`
- Backbone配置: `config/model/diffuser/decoder/backbone/README.md`

## 🎓 引用

如使用PTv3，请引用：
```bibtex
@article{wu2023point,
  title={Point transformer v3: Simpler, faster, stronger},
  author={Wu, Xiaoyang and others},
  journal={arXiv preprint arXiv:2312.10035},
  year={2023}
}
```

## 💡 最佳实践

1. **首次使用**: 先运行 `verify_ptv3_integration.py` 验证环境
2. **选择配置**: 优先使用 `ptv3_light`（平衡性能与精度）
3. **显存不足**: 减小 batch size 或使用 `ptv3_light`
4. **无Flash Attn**: 使用 `ptv3_no_flash`
5. **调试**: 使用小数据集和小模型先验证流程

## ⚙️ 高级配置

### 自定义grid_size
```yaml
# 更细的体素化（更多点，更慢）
backbone:
  grid_size: 0.01

# 更粗的体素化（更少点，更快）
backbone:
  grid_size: 0.05
```

### 调整encoder深度
```yaml
backbone:
  encoder_depths: [1, 1, 1, 1, 1]  # 更浅（更快）
  # 或
  encoder_depths: [2, 2, 3, 3, 2]  # 更深（更好）
```

## 📞 支持

- 问题反馈: GitHub Issues
- 文档: 项目 `docs/` 目录
- 测试: 项目 `tests/` 目录

