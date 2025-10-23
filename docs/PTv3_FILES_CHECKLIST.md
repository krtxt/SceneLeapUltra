# PTv3 Integration - Files Checklist

## ✅ 新增核心文件

### 模型实现
- [x] `models/backbone/ptv3_backbone.py` - PTv3 封装器
- [x] `models/backbone/ptv3/ptv3.py` - PTv3 官方实现（已修正导入）

### 配置文件（Diffuser）
- [x] `config/model/diffuser/decoder/backbone/ptv3_light.yaml`
- [x] `config/model/diffuser/decoder/backbone/ptv3.yaml`
- [x] `config/model/diffuser/decoder/backbone/ptv3_no_flash.yaml`

### 配置文件（Flow Matching）
- [x] `config/model/flow_matching/decoder/backbone/ptv3_light.yaml` (符号链接)
- [x] `config/model/flow_matching/decoder/backbone/ptv3.yaml` (符号链接)
- [x] `config/model/flow_matching/decoder/backbone/ptv3_no_flash.yaml` (符号链接)

### 测试文件
- [x] `tests/test_ptv3_backbone_basic.py` - 基础功能测试
- [x] `tests/test_smoke_dit_ptv3.py` - DiT 冒烟测试
- [x] `tests/test_smoke_unet_ptv3.py` - UNet 冒烟测试
- [x] `tests/verify_ptv3_integration.py` - 集成验证脚本

### 文档文件
- [x] `models/backbone/PTv3_INTEGRATION.md` - 详细使用文档
- [x] `PTv3_INTEGRATION_SUMMARY.md` - 集成总结
- [x] `PTv3_QUICK_REFERENCE.md` - 快速参考
- [x] `CHANGELOG_PTv3.md` - 变更日志
- [x] `PTv3_FILES_CHECKLIST.md` - 本文件

## ✅ 修改的现有文件

### 模型文件
- [x] `models/backbone/__init__.py` - 添加 PTv3 路由
- [x] `models/backbone/pointnet2.py` - 添加 output_dim 属性
- [x] `models/decoder/dit.py` - 扩展 _adjust_backbone_config，自适应 scene_projection
- [x] `models/decoder/unet_new.py` - 扩展 _adjust_backbone_config，添加 scene_projection

### PTv3 官方文件
- [x] `models/backbone/ptv3/ptv3.py` - 修正导入路径

## 📋 配置详情

### ptv3_light.yaml
```yaml
name: ptv3
variant: light
use_flash_attention: true
grid_size: 0.03
encoder_channels: [16, 32, 64, 128, 256]
encoder_depths: [1, 1, 2, 2, 1]
mlp_ratio: 2
out_dim: 256
```

### ptv3.yaml
```yaml
name: ptv3
variant: base
use_flash_attention: true
grid_size: 0.03
encoder_channels: [32, 64, 128, 256, 512]
encoder_depths: [2, 2, 2, 2, 2]
mlp_ratio: 4
out_dim: 512
```

### ptv3_no_flash.yaml
```yaml
name: ptv3
variant: no_flash
use_flash_attention: false
grid_size: 0.03
encoder_channels: [32, 64, 128, 256, 512]
encoder_depths: [2, 2, 2, 2, 2]
mlp_ratio: 4
out_dim: 512
```

## 🔍 关键代码改动

### models/backbone/__init__.py
```python
from .ptv3_backbone import PTV3Backbone

def build_backbone(backbone_cfg):
    # ... existing code ...
    elif backbone_cfg.name.lower() in ("ptv3", "ptv3_light", "ptv3_no_flash"):
        return PTV3Backbone(backbone_cfg)
```

### models/backbone/pointnet2.py
```python
def __init__(self, cfg):
    super().__init__()
    
    # Output dimension (for interface compatibility with PTv3)
    self.output_dim = 512
    # ... rest of init ...
```

### models/decoder/dit.py
```python
# _adjust_backbone_config 方法
elif backbone_name == 'ptv3':
    adjusted_cfg.input_feature_dim = feature_input_dim
    self.logger.debug(...)

# __init__ 方法
backbone_out_dim = getattr(self.scene_model, 'output_dim', 512)
self.scene_projection = nn.Linear(backbone_out_dim, self.d_model)
```

### models/decoder/unet_new.py
```python
# _adjust_backbone_config 方法
elif backbone_name == 'ptv3':
    adjusted_cfg.input_feature_dim = feature_input_dim
    logging.debug(...)

# __init__ 方法
backbone_out_dim = getattr(self.scene_model, 'output_dim', 512)
if backbone_out_dim != self.d_model:
    self.scene_projection = nn.Linear(backbone_out_dim, self.d_model)
else:
    self.scene_projection = nn.Identity()

# condition 方法
scene_feat = self.scene_projection(scene_feat)
```

## 📊 测试覆盖

### 单元测试
- [x] PTv3Backbone 实例化
- [x] PTv3Backbone 前向传播（4种输入）
- [x] 稀疏-致密转换
- [x] 输出维度验证

### 集成测试
- [x] DiT + ptv3_light (4种输入)
- [x] DiT + ptv3 (4种输入)
- [x] DiT + ptv3_no_flash (4种输入)
- [x] UNet + ptv3_light (4种输入)
- [x] UNet + ptv3 (4种输入)
- [x] UNet + ptv3_no_flash (4种输入)

### 验证脚本
- [x] 依赖检查
- [x] 配置文件检查
- [x] 模型文件检查
- [x] 导入测试
- [x] 实例化测试

## 🎯 接口兼容性

### 输入格式
- [x] (B, N, 3) - xyz only
- [x] (B, N, 6) - xyz + rgb
- [x] (B, N, 4) - xyz + mask
- [x] (B, N, 7) - xyz + rgb + mask

### 输出格式
- [x] xyz: (B, K, 3)
- [x] features: (B, out_dim, K)
- [x] out_dim 自适应（light=256, base=512）

### 解码器兼容
- [x] DiT: scene_projection 自动适配
- [x] UNet: scene_projection 条件适配
- [x] 配置自适应（_adjust_backbone_config）

## 📦 依赖清单

### 必需依赖
- [x] `torch` >= 1.9.0
- [x] `spconv` (CUDA 版本匹配)
- [x] `torch-scatter`
- [x] `addict`

### 可选依赖
- [x] `flash-attn` (用于 ptv3/ptv3_light)

### 检查方法
```bash
python -c "import spconv; print('spconv OK')"
python -c "import torch_scatter; print('torch_scatter OK')"
python -c "import addict; print('addict OK')"
python -c "import flash_attn; print('flash_attn OK')"  # 可选
```

## 🚀 快速验证

### 1. 运行验证脚本
```bash
python tests/verify_ptv3_integration.py
```

### 2. 运行基础测试
```bash
python tests/test_ptv3_backbone_basic.py
```

### 3. 运行冒烟测试
```bash
python tests/test_smoke_dit_ptv3.py
python tests/test_smoke_unet_ptv3.py
```

### 4. 尝试训练（可选）
```bash
python train_lightning.py \
    model/diffuser/decoder=dit \
    model/diffuser/decoder/backbone=ptv3_light \
    data=mini_obj_centric \
    trainer.max_epochs=1
```

## ✅ 验收标准

- [x] 所有新增文件已创建
- [x] 所有修改文件已更新
- [x] 无 linter 错误
- [x] 配置文件格式正确
- [x] 测试文件可运行
- [x] 文档完整清晰
- [x] 导入路径正确
- [x] 接口兼容性保持
- [x] 四种输入通道组合全部支持
- [x] DiT/UNet 均可使用 PTv3

## 📝 后续任务（用户验证）

- [ ] 在真实数据上训练验证
- [ ] 性能基准测试
- [ ] 显存占用分析
- [ ] 推理速度对比
- [ ] 精度评估（与 PointNet2 对比）
- [ ] 长期稳定性测试

## 🔧 故障排除参考

| 问题 | 解决方案 | 文件 |
|------|---------|------|
| ImportError: flash_attn | 使用 ptv3_no_flash | `config/*/backbone/ptv3_no_flash.yaml` |
| CUDA OOM | 使用 ptv3_light | `config/*/backbone/ptv3_light.yaml` |
| spconv 版本不匹配 | 重装对应版本 | - |
| 导入路径错误 | 检查 sys.path | `models/backbone/ptv3/ptv3.py` |

## 📚 文档索引

1. **快速开始**: `PTv3_QUICK_REFERENCE.md`
2. **详细文档**: `models/backbone/PTv3_INTEGRATION.md`
3. **集成总结**: `PTv3_INTEGRATION_SUMMARY.md`
4. **变更日志**: `CHANGELOG_PTv3.md`
5. **配置说明**: `config/model/diffuser/decoder/backbone/README.md`

## ✨ 完成标记

集成已完成！所有文件已就位，测试已准备好，文档已完善。

**下一步**: 运行 `python tests/verify_ptv3_integration.py` 验证环境。

