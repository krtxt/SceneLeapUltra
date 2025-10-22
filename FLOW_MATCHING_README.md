# Flow Matching 实现完成 ✅

恭喜!Flow Matching 已成功集成到 SceneLeapUltra 项目中。

## 实现总结

### 已完成的功能

1. ✅ **Flow Matching 核心工具** (`models/utils/flow_matching_utils.py`)
   - Optimal Transport (OT) 条件流
   - Euler 和 Heun ODE 求解器
   - 速度场损失函数

2. ✅ **Flow Matching Core Mixin** (`models/utils/flow_matching_core.py`)
   - 训练逻辑 (时间采样、条件流、速度匹配)
   - 推理逻辑 (ODE 积分、CFG 支持)
   - 完全兼容 DDPM 的接口

3. ✅ **PyTorch Lightning 模块** (`models/flow_matching_lightning.py`)
   - 完整的训练/验证/测试循环
   - 复用现有 DiT/UNet decoder
   - 兼容所有现有数据管道和评估指标

4. ✅ **配置文件**
   - `config/model/flow_matching/flow_matching.yaml` - 基础配置
   - `config/model/flow_matching/flow_matching_dit.yaml` - DiT 版本
   - `config/model/flow_matching/flow_matching_unet.yaml` - UNet 版本

5. ✅ **训练/测试集成**
   - `train_lightning.py` - 支持 Flow Matching 模型选择
   - `test_lightning.py` - 支持 Flow Matching 测试
   - 与现有训练脚本完全兼容

6. ✅ **测试和文档**
   - `test_flow_matching.py` - 单元测试 (所有测试通过!)
   - `docs/flow_matching_guide.md` - 完整使用指南

## 快速开始

### 1. 验证安装

```bash
python test_flow_matching.py
```

如果看到 "✓ All tests passed!", 说明安装成功!

### 2. 训练 Flow Matching 模型

#### 使用 DiT decoder (推荐)

```bash
# 激活环境
source ~/.bashrc && conda activate DexGrasp

# 启动训练
./train_distributed.sh model=flow_matching/flow_matching_dit
```

#### 使用 UNet decoder

```bash
./train_distributed.sh model=flow_matching/flow_matching_unet
```

#### 自定义超参数

```bash
./train_distributed.sh \
    model=flow_matching/flow_matching_dit \
    batch_size=128 \
    model.num_sampling_steps=50 \
    model.sampler_type=heun
```

### 3. 测试模型

```bash
# 测试单个 checkpoint
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +"checkpoint_path='experiments/flow_exp/checkpoints/epoch=50.ckpt'"

# 测试整个实验目录
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +train_root='experiments/flow_exp'
```

## 关键特性

### 与 DDPM 的对比

| 特性 | DDPM | Flow Matching |
|------|------|---------------|
| 训练目标 | 噪声预测 | 速度场回归 |
| 采样步数 | 100-1000 | **10-50** |
| 训练稳定性 | 需要调 schedule | ✅ 更稳定 |
| 推理速度 | 基准 | ✅ **2-10x 更快** |
| 超参数调优 | 较复杂 | ✅ 更简单 |

### 完全兼容性

- ✅ **复用所有 decoder**: DiT 和 UNet 无需修改
- ✅ **复用所有条件模块**: PointNet2/PTv3 场景编码器, CLIP/T5 文本编码器
- ✅ **相同的数据管道**: 无需修改 dataset 或 dataloader
- ✅ **相同的评估指标**: Q1, penetration, valid_q1 等

## 配置说明

### Flow Matching 特有参数

```yaml
# ODE 求解器
sampler_type: heun  # 'euler' 或 'heun'

# 采样步数 (比 DDPM 少得多!)
num_sampling_steps: 50  # DiT+Heun: 50, UNet+Heun: 25-50

# 最小噪声水平
sigma_min: 1.0e-4

# 损失类型
loss_type: l2  # 'l2' 或 'l1'
```

### 推荐配置

| 场景 | Decoder | Sampler | 步数 | 速度 | 质量 |
|------|---------|---------|------|------|------|
| 快速原型 | UNet | Euler | 50 | 最快 | 好 |
| 标准训练 | DiT | Heun | 50 | 快 | 很好 |
| 最佳质量 | DiT | Heun | 100 | 中等 | 最佳 |

## 目录结构

```
SceneLeapUltra/
├── models/
│   ├── flow_matching_lightning.py      # PyTorch Lightning 模块
│   └── utils/
│       ├── flow_matching_utils.py      # OT flow, 采样器
│       └── flow_matching_core.py       # 核心训练/采样逻辑
├── config/
│   └── model/
│       └── flow_matching/
│           ├── flow_matching.yaml      # 基础配置
│           ├── flow_matching_dit.yaml  # DiT 版本
│           └── flow_matching_unet.yaml # UNet 版本
├── docs/
│   └── flow_matching_guide.md          # 详细文档
└── test_flow_matching.py                # 单元测试
```

## 使用示例

### 训练

```bash
# 基础训练
./train_distributed.sh model=flow_matching/flow_matching_dit

# 多 GPU 训练
./train_distributed.sh --gpus 4 model=flow_matching/flow_matching_dit

# 启用 Classifier-Free Guidance
./train_distributed.sh \
    model=flow_matching/flow_matching_dit \
    model.use_cfg=false  # 训练时保持 false, 推理时启用

# 调整采样步数
./train_distributed.sh \
    model=flow_matching/flow_matching_dit \
    model.num_sampling_steps=25  # 更快但质量稍低
```

### 推理时启用 CFG

```bash
python test_lightning.py \
    +"checkpoint_path='...'" \
    model.use_cfg=true \
    model.guidance_scale=7.5
```

## 技术细节

### 训练流程

1. 采样时间: `t ~ Uniform(0, 1)`
2. 构建流: `x_t = (1-t)*x_0 + t*x_1 + σ*ε`
3. 目标速度: `u_t = x_1 - x_0`
4. 模型预测: `v_θ(x_t, t, cond)`
5. 损失: `L = ||v_θ - u_t||²`

### 推理流程 (ODE 积分)

**Euler 方法**:
```
x_{t+dt} = x_t + dt * v_θ(x_t, t)
```

**Heun 方法** (推荐):
```
k1 = v_θ(x_t, t)
k2 = v_θ(x_t + dt*k1, t+dt)
x_{t+dt} = x_t + dt/2 * (k1 + k2)
```

## 常见问题

**Q: 为什么测试要先运行 `python test_flow_matching.py`?**

A: 这个脚本验证核心功能是否正常,避免浪费时间在完整训练上。

**Q: Flow Matching 和 DDPM 的 checkpoint 兼容吗?**

A: 不兼容。虽然它们共享 decoder 架构,但训练目标不同 (速度 vs 噪声)。

**Q: 如何选择采样步数?**

A:
- **快速测试**: 25 步 (Heun)
- **标准使用**: 50 步 (Heun)
- **最佳质量**: 100 步 (Heun)

**Q: Flow Matching 适合我的场景吗?**

A: 如果你需要:
- ✅ 快速推理 (实时或大规模测试)
- ✅ 简化训练 (减少超参数调优)
- ✅ 稳定训练 (不用担心 noise schedule)

那么 Flow Matching 是很好的选择!

## 下一步

1. **运行测试**: `python test_flow_matching.py`
2. **阅读文档**: `docs/flow_matching_guide.md`
3. **开始训练**: `./train_distributed.sh model=flow_matching/flow_matching_dit`
4. **对比 DDPM**: 训练相同 epoch 数,比较推理速度和质量

## 参考资料

- **Flow Matching 论文**: [Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)
- **详细文档**: `docs/flow_matching_guide.md`
- **配置文件**: `config/model/flow_matching/`

---

**祝训练顺利! 如有问题,请查看 `docs/flow_matching_guide.md` 或运行测试脚本。**
