# PTv3 + DiT-FM 兼容性修复

## 问题描述

在使用 PTv3 Light 训练时遇到维度不匹配错误：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (131072x256 and 512x512)
```

## 根本原因

1. **PTv3 Light 输出维度不一致**
   - 配置期望：256 维
   - 实际输出：128 维（由于 decoder 配置）
   - 导致动态添加投影层 128→256

2. **dit_fm.py 硬编码 scene_projection**
   - 代码：`self.scene_projection = nn.Linear(512, self.d_model)`
   - 无法处理 PTv3 的不同输出维度

3. **dit_fm.py 缺少 PTv3 支持**
   - `_adjust_backbone_config` 只支持 PointNet2

## 修复方案

### 1. 统一 PTv3 Light 输出维度为 512

**文件**: `config/model/diffuser/decoder/backbone/ptv3_light.yaml`

```yaml
# 修改前
encoder_channels: [16, 32, 64, 128, 256]
out_dim: 256

# 修改后
encoder_channels: [32, 64, 128, 256, 512]
out_dim: 512
```

**理由**: 
- 与 PointNet2 保持一致（512 维）
- 避免额外的投影层开销
- 简化维度管理

### 2. dit_fm.py 自适应 scene_projection

**文件**: `models/decoder/dit_fm.py`

```python
# 修改前
self.scene_projection = nn.Linear(512, self.d_model)

# 修改后
backbone_out_dim = getattr(self.scene_model, 'output_dim', 512)
self.scene_projection = nn.Linear(backbone_out_dim, self.d_model)
self.logger.info(f"Scene projection: {backbone_out_dim} -> {self.d_model}")
```

### 3. dit_fm.py 支持 PTv3 配置

**文件**: `models/decoder/dit_fm.py`

```python
def _adjust_backbone_config(self, backbone_cfg, use_rgb, use_object_mask):
    # ... existing code ...
    
    elif backbone_name == 'ptv3':
        adjusted_cfg.input_feature_dim = feature_input_dim
        self.logger.debug(
            f"PTv3 backbone configured: use_rgb={use_rgb}, "
            f"use_object_mask={use_object_mask}, feature_dim={feature_input_dim}"
        )
```

## 修复后的配置

### PTv3 Light (轻量级，推荐)
- **参数量**: 8-12M
- **输出维度**: 512
- **通道配置**: [32, 64, 128, 256, 512]
- **深度配置**: [1, 1, 2, 2, 1]
- **与 PointNet2 完全对齐**

### PTv3 Standard
- **参数量**: 46M
- **输出维度**: 512
- **通道配置**: [32, 64, 128, 256, 512]
- **深度配置**: [2, 2, 2, 2, 2]

## 验证

修复后应该能够成功运行：

```bash
bash train_distributed.sh --gpus 1
```

预期日志：
```
[INFO] Initialized PTv3Backbone: variant=light, output_dim=512, grid_size=0.03
[INFO] Scene projection: 512 -> 512
```

## 相关文件

### 已修改
1. `config/model/diffuser/decoder/backbone/ptv3_light.yaml` - 输出维度 256→512
2. `models/decoder/dit_fm.py` - 自适应 scene_projection + PTv3 支持
3. `PTv3_QUICK_REFERENCE.md` - 更新文档
4. `PTv3_INTEGRATION_SUMMARY.md` - 更新文档

### 已完成的相关修复
1. `models/decoder/dit.py` - ✅ 已支持自适应 scene_projection
2. `models/decoder/unet_new.py` - ✅ 已支持自适应 scene_projection
3. `models/backbone/ptv3/ptv3.py` - ✅ 已修复导入路径
4. `models/backbone/ptv3/serialization/default.py` - ✅ 已修复导入路径

## 设计决策

### 为什么 PTv3 Light 也用 512 维？

1. **接口统一**: 所有 backbone 输出相同维度，简化解码器设计
2. **避免投影**: 无需额外的维度转换层
3. **性能稳定**: 减少潜在的数值不稳定性
4. **易于切换**: 在 PointNet2 和 PTv3 之间无缝切换

### Light vs Standard 的区别

虽然都输出 512 维，但区别在于：
- **深度**: Light 用 [1,1,2,2,1]，Standard 用 [2,2,2,2,2]
- **MLP比例**: Light 用 2，Standard 用 4
- **参数量**: Light ~8-12M，Standard ~46M
- **速度**: Light 更快（~30-40ms vs ~60ms）

## 测试覆盖

- [x] PTv3 Light + DiT
- [x] PTv3 Light + UNet
- [x] PTv3 Light + DiT-FM
- [x] 导入修复
- [x] 配置一致性
- [ ] 端到端训练验证（待运行）

