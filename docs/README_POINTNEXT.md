# PointNext Backbone - 快速开始

## 简介

PointNext 是一个基于改进 PointNet++ 的高效点云编码器，已成功集成到 SceneLeapUltra 项目中。

**核心特性:**
- 输入: (B, N, 3) → 输出: (B, K, d_model)
- 典型配置: N=8192, K=128, d_model=512
- 参数量: ~0.2M (默认)
- 与 PointNet2/PTv3 完全兼容

## 快速测试

```bash
# 1. 安装依赖
pip install multimethod shortuuid easydict einops timm

# 2. 运行快速测试
python tests/test_pointnext_quick.py
```

预期输出:
```
✓ 所有测试通过！PointNext backbone 工作正常
✓ 输入: (2, 8192, 3)
✓ 输出 xyz: (2, 128, 3)
✓ 输出 features: (2, 512, 128)
```

## 使用方法

### 方式 1: 命令行

```bash
python train_lightning.py model/decoder/backbone=pointnext
```

### 方式 2: Python 代码

```python
from omegaconf import OmegaConf
from models.backbone import build_backbone

cfg = OmegaConf.load('config/model/flow_matching/decoder/backbone/pointnext.yaml')
model = build_backbone(cfg).cuda()

# 推理
xyz, features = model(pointcloud)  # (B, 128, 3), (B, 512, 128)
```

### 方式 3: 配置文件

```yaml
model:
  decoder:
    backbone:
      name: pointnext
      num_tokens: 128
      out_dim: 512
```

## 文件结构

```
models/backbone/
  └── pointnext_backbone.py          # 主实现

config/model/flow_matching/decoder/backbone/
  └── pointnext.yaml                  # 配置文件

docs/
  └── pointnext_setup.md              # 详细文档

tests/
  ├── test_pointnext_quick.py         # 快速测试 ← 从这里开始
  ├── test_pointnext_backbone.py      # 完整测试套件
  └── test_pointnext_summary.md       # 测试总结

POINTNEXT_INTEGRATION.md              # 集成完成报告
```

## 重要配置

### Token 数量控制

输出 tokens = 输入点数 / 下采样率

**示例**:
- `strides=[1,2,2,4,4]`: 8192点 → 128 tokens (推荐) ✓
- `strides=[1,2,2,2,4]`: 8192点 → 256 tokens
- `strides=[1,4,4,4,4]`: 8192点 → 32 tokens

### 模型大小

通过 `width` 参数控制:
- `width=32`: ~0.2M 参数 (轻量级，默认)
- `width=64`: ~0.8M 参数 (标准)
- `width=128`: ~3M 参数 (大型)

## 系统要求

- ✅ PyTorch >= 1.8
- ✅ CUDA (必需，不支持 CPU)
- ✅ GPU 显存 >= 2GB

## 常见问题

**Q: Token 数量不匹配？**
A: 调整配置文件中的 `strides` 参数。参考 `docs/pointnext_setup.md`

**Q: 导入错误？**
A: 运行 `pip install multimethod shortuuid easydict einops timm`

**Q: 需要 CPU 支持？**
A: PointNext 需要 CUDA。考虑使用 PointNet2 (支持 CPU)

**Q: 性能对比？**
A: 参见 `POINTNEXT_INTEGRATION.md` 的性能对比表

## 下一步

1. ✅ 运行快速测试: `python tests/test_pointnext_quick.py`
2. 📖 阅读详细文档: `docs/pointnext_setup.md`  
3. 🔧 调整配置: `config/model/flow_matching/decoder/backbone/pointnext.yaml`
4. 🚀 开始训练: `python train_lightning.py model/decoder/backbone=pointnext`

## 支持

- 详细文档: `docs/pointnext_setup.md`
- 集成报告: `POINTNEXT_INTEGRATION.md`
- 测试总结: `tests/test_pointnext_summary.md`

---

**状态**: ✅ 可用  
**版本**: v1.0  
**更新**: 2025-10-29

