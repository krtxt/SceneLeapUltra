# Flow Matching for Grasp Synthesis

本文档介绍如何在 SceneLeapUltra 项目中使用 Flow Matching 模型。

## 什么是 Flow Matching?

Flow Matching 是一种生成模型方法,通过学习速度场(velocity field)将先验分布(如高斯噪声)传输到数据分布。相比 DDPM (扩散模型):

### 优势
- **更简单的训练目标**: 直接回归速度场 `v(x,t)`,无需复杂的噪声调度
- **更快的采样**: 通常只需要 10-50 步,而 DDPM 需要 100-1000 步
- **数学上更优雅**: 使用 Optimal Transport (OT) 路径,理论基础清晰
- **训练更稳定**: 不需要调整 beta schedule 等超参数

### 与 DDPM 的兼容性
- **完全复用现有 decoder**: DiT 和 UNet 架构无需修改
- **共享所有条件模块**: 场景编码器、文本编码器等完全兼容
- **相同的数据管道**: 使用相同的 dataset 和 dataloader
- **相同的评估指标**: 抓取质量、穿透度等评估方法不变

## 快速开始

### 1. 运行测试

首先验证 Flow Matching 实现是否正常:

```bash
python test_flow_matching.py
```

如果所有测试通过,你会看到:
```
✓ All tests passed! Flow Matching implementation is ready to use.
```

### 2. 训练 Flow Matching 模型

#### 使用 DiT decoder (推荐用于多抓取)

```bash
# 自动检测所有 GPU 并训练
./train_distributed.sh model=flow_matching/flow_matching_dit

# 指定 GPU 数量
./train_distributed.sh --gpus 4 model=flow_matching/flow_matching_dit

# 自定义超参数
./train_distributed.sh model=flow_matching/flow_matching_dit \
    batch_size=128 \
    model.optimizer.lr=0.001 \
    model.num_sampling_steps=50
```

#### 使用 UNet decoder (更轻量)

```bash
./train_distributed.sh model=flow_matching/flow_matching_unet
```

### 3. 测试已训练模型

```bash
# 测试单个 checkpoint
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +"checkpoint_path='experiments/my_flow_exp/checkpoints/epoch=100.ckpt'" \
    data.test.batch_size=64

# 测试实验目录下所有 checkpoints
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +train_root='experiments/my_flow_exp'
```

## 配置说明

### Flow Matching 特有参数

在配置文件 `config/model/flow_matching/flow_matching.yaml` 中:

```yaml
# 最小噪声水平 (数值稳定性)
sigma_min: 1.0e-4

# ODE 求解器类型
#   - 'euler': 一阶 Euler 方法 (更快,需要更多步数)
#   - 'heun': 二阶 Heun 方法 (每步慢,总步数更少)
sampler_type: heun

# ODE 积分步数
#   - Heun: 25-50 步通常足够
#   - Euler: 50-100 步推荐
num_sampling_steps: 50

# 速度匹配损失类型
#   - 'l2': MSE 损失 (默认,更稳定)
#   - 'l1': MAE 损失 (对异常值更鲁棒)
loss_type: l2
```

### 采样步数建议

| Decoder | Sampler | 推荐步数 | 质量 | 速度 |
|---------|---------|----------|------|------|
| DiT     | Heun    | 50       | 最佳 | 中等 |
| DiT     | Euler   | 100      | 好   | 较慢 |
| UNet    | Heun    | 25-50    | 好   | 快   |
| UNet    | Euler   | 50-100   | 好   | 中等 |

**对比 DDPM**: DDPM 通常需要 100-1000 步才能达到相同质量。

## 架构细节

### 核心组件

```
models/
├── flow_matching_lightning.py      # PyTorch Lightning 模块
├── utils/
│   ├── flow_matching_utils.py      # OT flow, 采样器
│   └── flow_matching_core.py       # 核心训练/采样逻辑
└── decoder/
    ├── dit.py                       # DiT (与 DDPM 共享)
    └── unet_new.py                  # UNet (与 DDPM 共享)
```

### 训练流程

1. **采样时间步**: `t ~ Uniform(0, 1)`
2. **构建条件流**: `x_t = (1-t)*x_0 + t*x_1 + σ*ε`
   - `x_0`: 高斯噪声 (先验)
   - `x_1`: 真实抓取姿态 (数据)
3. **目标速度**: `u_t = x_1 - x_0`
4. **模型预测**: `v_θ(x_t, t, condition)`
5. **损失函数**: `L = ||v_θ - u_t||^2`

### 推理流程

使用 ODE 求解器从 `t=0` 积分到 `t=1`:

**Euler 方法**:
```
x_{t+dt} = x_t + dt * v_θ(x_t, t)
```

**Heun 方法** (更精确):
```
k1 = v_θ(x_t, t)
k2 = v_θ(x_t + dt*k1, t+dt)
x_{t+dt} = x_t + dt/2 * (k1 + k2)
```

## 高级用法

### Classifier-Free Guidance (CFG)

在推理时启用 CFG 可以提高生成质量:

```bash
# 训练时保持 use_cfg=false
./train_distributed.sh model=flow_matching/flow_matching_dit

# 测试时启用 CFG
python test_lightning.py \
    +"checkpoint_path='...'" \
    model.use_cfg=true \
    model.guidance_scale=7.5
```

### 负向提示 (Negative Prompts)

```bash
./train_distributed.sh \
    model=flow_matching/flow_matching_dit \
    use_negative_prompts=true \
    model.use_negative_guidance=true \
    model.negative_guidance_scale=1.0
```

### 调整采样步数

在推理时可以动态调整步数:

```python
# 在代码中
trajectory = model.sample(data, num_steps=25)  # 更快但质量稍低
trajectory = model.sample(data, num_steps=100) # 更慢但质量更高
```

### 固定抓取数量

```bash
./train_distributed.sh \
    model=flow_matching/flow_matching_dit \
    fix_num_grasps=true \
    target_num_grasps=10
```

## 与 DDPM 性能对比

基于文献和实验经验:

| 指标 | DDPM | Flow Matching |
|------|------|---------------|
| 训练速度 | 基准 | 相似或略快 |
| 推理速度 | 基准 | **2-10x 更快** |
| 生成质量 | 基准 | 相似或略好 |
| 训练稳定性 | 需要调 schedule | **更稳定** |
| 超参数敏感度 | 较高 | **较低** |

## 常见问题

### Q: Flow Matching 和 DDPM 可以用相同的 checkpoint 吗?

**A**: 不行。虽然它们共享 decoder 架构,但训练目标不同 (速度 vs 噪声),因此 checkpoint 不兼容。需要分别训练。

### Q: 何时选择 Flow Matching 而非 DDPM?

**A**: 考虑以下场景:
- ✅ **需要快速推理** (实时应用、大规模测试)
- ✅ **希望简化训练流程** (减少超参数调优)
- ✅ **计算资源有限** (更少的采样步数)
- ❌ 已有良好调优的 DDPM 模型 (迁移成本)

### Q: 推荐的采样步数是多少?

**A**:
- **快速测试**: 25 步 (Heun) 或 50 步 (Euler)
- **正常使用**: 50 步 (Heun) 或 100 步 (Euler)
- **最佳质量**: 100 步 (Heun)

实验建议: 从 50 步开始,根据质量需求调整。

### Q: 可以混合使用不同 decoder 吗?

**A**: 可以! Flow Matching 支持:
- DiT: `model=flow_matching/flow_matching_dit`
- UNet: `model=flow_matching/flow_matching_unet`

选择建议:
- **多抓取场景** (>5 grasps): DiT
- **单/少抓取** (≤5 grasps): UNet
- **内存受限**: UNet

## 参考文献

1. **Flow Matching for Generative Modeling**
   Lipman et al., ICLR 2023
   [论文链接](https://arxiv.org/abs/2210.02747)

2. **Improving and Generalizing Flow-Based Generative Models**
   Tong et al., NeurIPS 2023
   [论文链接](https://arxiv.org/abs/2302.00482)

3. **Stable Target Field for Reduced Variance Score Estimation**
   Liu et al., ICLR 2023

## 获取帮助

如果遇到问题:

1. **运行测试**: `python test_flow_matching.py`
2. **检查配置**: 确保使用正确的 `model=flow_matching/...`
3. **查看日志**: 训练日志在 `experiments/*/lightning_logs/`
4. **对比 DDPM**: 如果 DDPM 工作正常,Flow Matching 应该也能工作

## 致谢

本实现参考了 Flow Matching 原始论文和 PyTorch 社区的优秀实现。
