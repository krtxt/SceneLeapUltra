# PTv3 Integration Changelog

## [Unreleased] - 2025-01-XX

### Added - PTv3 Point Cloud Encoder Integration

#### 新功能
- 添加 Point Transformer V3 (PTv3) 作为点云编码器，与 PointNet2 并行使用
- 提供三种 PTv3 配置：
  - `ptv3_light`: 轻量级版本（256维输出，8-12M参数）
  - `ptv3`: 标准版本（512维输出，46M参数，启用 Flash Attention）
  - `ptv3_no_flash`: 标准版本但禁用 Flash Attention（兼容性）

#### 核心组件
- **PTv3Backbone** (`models/backbone/ptv3_backbone.py`)
  - 封装官方 PTv3 实现
  - 提供与 PointNet2 一致的接口
  - 自动处理稀疏-致密格式转换
  - 支持动态特征通道（xyz / xyz+rgb / xyz+mask / xyz+rgb+mask）

#### 解码器适配
- DiT 解码器：
  - 自适应 scene_projection（根据 backbone.output_dim 自动调整）
  - 支持 PTv3 配置自适应
- UNet 解码器：
  - 新增条件 scene_projection
  - 支持 PTv3 配置自适应

#### 配置文件
- `config/model/diffuser/decoder/backbone/ptv3_light.yaml`
- `config/model/diffuser/decoder/backbone/ptv3.yaml`
- `config/model/diffuser/decoder/backbone/ptv3_no_flash.yaml`

#### 测试
- `tests/test_ptv3_backbone_basic.py`: PTv3 Backbone 基础功能测试
- `tests/test_smoke_dit_ptv3.py`: DiT + PTv3 冒烟测试
- `tests/test_smoke_unet_ptv3.py`: UNet + PTv3 冒烟测试
- `tests/verify_ptv3_integration.py`: 集成验证脚本

#### 文档
- `models/backbone/PTv3_INTEGRATION.md`: 详细使用文档
- `PTv3_INTEGRATION_SUMMARY.md`: 集成总结
- `CHANGELOG_PTv3.md`: 本变更日志

### Changed

#### 接口改进
- PointNet2 添加 `output_dim` 属性（值为 512）以保持接口一致性
- DiT/UNet 的 `_adjust_backbone_config` 方法扩展支持 PTv3
- 解码器 scene_projection 改为自适应（根据 backbone 输出维度）

#### Bug 修复
- 修正 PTv3 官方代码中的硬编码导入路径
  - `from grasp_gen.models.ptv3.serialization.default import encode`
  - 改为: `from .serialization.default import encode`

### Dependencies

#### 新增依赖（必需）
```bash
pip install spconv-cu118  # 或对应的 CUDA 版本
pip install torch-scatter
pip install addict
```

#### 新增依赖（可选）
```bash
pip install flash-attn  # 用于 ptv3 和 ptv3_light
```

### Usage

#### 使用 PTv3 训练

```bash
# DiT + PTv3 Light
python train_lightning.py \
    model/diffuser/decoder=dit \
    model/diffuser/decoder/backbone=ptv3_light

# UNet + PTv3
python train_lightning.py \
    model/diffuser/decoder=unet \
    model/diffuser/decoder/backbone=ptv3
```

#### 验证集成

```bash
# 激活环境
source ~/.bashrc && conda activate DexGrasp

# 运行验证脚本
python tests/verify_ptv3_integration.py

# 运行冒烟测试
python tests/test_smoke_dit_ptv3.py
python tests/test_smoke_unet_ptv3.py
```

### Performance

| Backbone    | 参数量 | 输出维度 | 推理时间* | 适用场景 |
|-------------|--------|---------|-----------|----------|
| PointNet2   | 560K   | 512     | ~17ms     | 实时、资源受限 |
| PTv3 Light  | 8-12M  | 256     | ~30-40ms  | 平衡性能与精度 |
| PTv3        | 46M    | 512     | ~60ms     | 高精度、现代GPU |

*batch=2, points=1024, GPU

### Breaking Changes
无破坏性变更。PTv3 作为可选 backbone 添加，不影响现有 PointNet2 的使用。

### Known Issues
1. PTv3 需要 GPU（不支持 CPU）
2. 依赖 spconv，需与 CUDA 版本匹配
3. Flash Attention 需要较新的 GPU 和驱动
4. 显存占用高于 PointNet2

### Migration Guide
现有使用 PointNet2 的用户无需迁移。若要尝试 PTv3：

1. 安装依赖：
   ```bash
   pip install spconv-cu118 torch-scatter addict
   pip install flash-attn  # 可选
   ```

2. 修改配置或使用命令行覆盖：
   ```bash
   python train_lightning.py model/diffuser/decoder/backbone=ptv3_light
   ```

### Future Work
- [ ] 混合精度训练支持（FP16/BF16）
- [ ] 优化稀疏-致密转换性能
- [ ] CPU 降级方案
- [ ] 可变 grid_size 支持
- [ ] 更多 PTv3 变体配置

### Credits
- PTv3 官方实现: [Pointcept](https://github.com/Pointcept/Pointcept)
- 论文: Wu et al., "Point Transformer V3", arXiv:2312.10035, 2023

