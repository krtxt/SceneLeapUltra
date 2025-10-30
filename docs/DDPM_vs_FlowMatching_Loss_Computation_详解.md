# DDPM vs Flow Matching 损失计算全过程详解

## 目录
1. [概述](#概述)
2. [DDPM 损失计算全过程](#ddpm-损失计算全过程)
3. [Flow Matching 损失计算全过程](#flow-matching-损失计算全过程)
4. [核心差异对比](#核心差异对比)
5. [数学原理](#数学原理)

---

## 概述

本文档详细讲解 SceneLeapUltra 项目中 DDPM 和 Flow Matching 两种生成模型在训练时的损失计算全过程。

### 核心区别
- **DDPM**: 基于离散时间步的扩散过程，预测噪声 ε
- **Flow Matching**: 基于连续时间的流模型，预测速度场 v

---

## DDPM 损失计算全过程

### 1. 整体流程概览

```
输入数据 → 预处理 → 采样时间步 → 前向扩散 → 模型预测 → 计算损失
```

### 2. 详细步骤分解

#### Step 1: 数据预处理
```python
# 位置: diffuser_lightning.py::training_step() -> _compute_loss()
processed_batch = process_hand_pose(batch, rot_type=self.rot_type, mode=self.mode)
norm_pose = processed_batch['norm_pose']  # [B, num_grasps, D]
```

**作用**: 
- 将原始手部姿态归一化到标准空间
- `norm_pose` 形状: `[B, num_grasps, D]`
  - B: batch size
  - num_grasps: 每个场景的抓取候选数量
  - D: 姿态维度 (25 for r6d, 23 for quat)

#### Step 2: 采样离散时间步
```python
# 位置: diffuser_lightning.py::_compute_loss()
ts = self._sample_timesteps(B)  # [B]
```

**实现细节**:
```python
# 位置: diffusion_core.py::_sample_timesteps()
def _sample_timesteps(self, batch_size: int):
    if self.rand_t_type == 'all':
        # 完全随机采样: t ∈ [0, T-1]
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
    elif self.rand_t_type == 'half':
        # 对称采样: 采样一半，镜像另一半 (t 和 T-t-1)
        ts = torch.randint(0, self.timesteps, ((batch_size + 1) // 2,), device=self.device)
        if batch_size % 2:
            return torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
        return torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
```

**关键点**:
- DDPM 使用**离散时间步**: t ∈ {0, 1, 2, ..., T-1}
- 典型配置: T=100 或 T=1000
- 采样策略影响训练效率和收敛性

#### Step 3: 生成噪声
```python
# 位置: diffuser_lightning.py::_compute_loss()
noise = torch.randn_like(norm_pose, device=self.device)  # ε ~ N(0, I)
```

**作用**:
- 生成标准高斯噪声 ε，形状与 norm_pose 相同
- 这个噪声将在前向扩散过程中添加到数据中

#### Step 4: 前向扩散 (Forward Diffusion)
```python
# 位置: diffuser_lightning.py::_compute_loss()
x_t = self.q_sample(x0=norm_pose, t=ts, noise=noise)
```

**核心实现**:
```python
# 位置: diffusion_core.py::q_sample()
def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    前向扩散过程: q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
    
    一步到位公式:
    x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    """
    B, num_grasps, _ = x0.shape
    t_expanded = t.unsqueeze(1).expand(-1, num_grasps)  # [B, num_grasps]
    
    # 获取扩散系数
    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_expanded].unsqueeze(-1)  # √ᾱ_t
    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_expanded].unsqueeze(-1)  # √(1-ᾱ_t)
    
    # 前向扩散
    x_t = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    return x_t
```

**数学原理**:
- **扩散调度**: β_t 定义噪声添加速率
  - α_t = 1 - β_t
  - ᾱ_t = ∏_{s=1}^{t} α_s (累积乘积)
  
- **前向过程**: 
  ```
  q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
  ```
  
- **一步采样公式**:
  ```
  x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε,  ε ~ N(0, I)
  ```

**关键特性**:
- 可以**一步到位**从 x_0 直接采样 x_t，无需逐步扩散
- 随着 t 增大，√ᾱ_t → 0，√(1-ᾱ_t) → 1
- 当 t=T 时，x_T ≈ N(0, I) (纯噪声)

#### Step 5: 计算条件特征
```python
# 位置: diffuser_lightning.py::_compute_loss()
condition_dict = self.eps_model.condition(processed_batch)
processed_batch.update(condition_dict)
```

**作用**:
- 从场景点云提取特征 → `scene_cond`
- 从文本提示提取特征 → `text_cond`
- 这些特征将作为条件输入模型

#### Step 6: 模型预测噪声
```python
# 位置: diffuser_lightning.py::_compute_loss()
output = self.eps_model(x_t, ts, processed_batch)
```

**模型输入**:
- `x_t`: 噪声化的姿态 [B, num_grasps, D]
- `ts`: 时间步索引 [B]
- `processed_batch`: 条件特征 (scene_cond, text_cond, etc.)

**模型输出**:
- 如果 `pred_x0=True`: 直接预测 x_0 (去噪后的数据)
- 如果 `pred_x0=False`: 预测噪声 ε_θ(x_t, t)

**网络架构**:
- 通常使用 DiT (Diffusion Transformer) 或 UNet
- 时间步 t 通过正弦位置编码嵌入
- 条件特征通过 cross-attention 或 AdaLN 注入

#### Step 7: 构建预测字典
```python
# 位置: diffuser_lightning.py::_compute_loss()
if self.pred_x0:
    pred_dict = {"pred_pose_norm": output, "noise": noise}
else:
    pred_x0 = self._compute_pred_x0_from_noise(x_t, ts, output)
    pred_dict = {"pred_noise": output, "pred_pose_norm": pred_x0, "noise": noise}
```

**如果预测噪声 (pred_x0=False)**:
```python
# 位置: diffusion_core.py::_compute_pred_x0_from_noise()
def _compute_pred_x0_from_noise(self, x_t, t, pred_noise):
    """
    从预测的噪声反推 x_0:
    x_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
    """
    sqrt_recip = self.sqrt_recip_alphas_cumprod[t_expanded].unsqueeze(-1)  # 1/√ᾱ_t
    sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t_expanded].unsqueeze(-1)  # √(1-ᾱ_t)/√ᾱ_t
    pred_x0 = sqrt_recip * x_t - sqrt_recipm1 * pred_noise
    return pred_x0
```

#### Step 8: 计算损失
```python
# 位置: diffuser_lightning.py::_compute_loss()
loss_dict = self.criterion(pred_dict, processed_batch, mode=mode)
loss = sum(v * self.loss_weights[k] for k, v in loss_dict.items() if k in self.loss_weights)
```

**损失组成** (在 `GraspLossPose` 中):
1. **噪声损失** (主要):
   ```python
   noise_loss = F.mse_loss(pred_noise, target_noise)
   ```
   或
   ```python
   noise_loss = F.mse_loss(pred_x0, target_x0)
   ```

2. **姿态损失** (可选):
   - 旋转损失 (rotation_loss)
   - 平移损失 (translation_loss)
   - 关节角度损失 (qpos_loss)

3. **加权总损失**:
   ```python
   total_loss = w_noise * noise_loss + w_rot * rotation_loss + ...
   ```

---

## Flow Matching 损失计算全过程

### 1. 整体流程概览

```
输入数据 → 预处理 → 采样连续时间 → 插值路径 → 模型预测速度 → 计算损失
```

### 2. 详细步骤分解

#### Step 1: 数据预处理
```python
# 位置: fm_lightning.py::training_step()
processed_batch = process_hand_pose(batch, self.rot_type, self.mode)
x0 = processed_batch['norm_pose']  # [B, num_grasps, D]
```

**与 DDPM 相同**: 归一化手部姿态到标准空间

#### Step 2: 采样连续时间
```python
# 位置: fm_lightning.py::training_step()
t = self._sample_time(B)  # [B], values in [0, 1]
```

**核心实现**:
```python
# 位置: fm_lightning.py::_sample_time()
def _sample_time(self, batch_size: int) -> torch.Tensor:
    """
    采样连续时间 t ∈ [0, 1]
    """
    if self.t_sampler == 'uniform':
        # 均匀分布
        t = torch.rand(batch_size, device=self.device)
    elif self.t_sampler == 'cosine':
        # 余弦调度: 强调中间时间步
        u = torch.rand(batch_size, device=self.device)
        t = (torch.acos(1 - 2*u) / math.pi)
    elif self.t_sampler == 'beta':
        # Beta 分布: 灵活控制时间分布
        alpha, beta = 2.0, 2.0
        dist = torch.distributions.Beta(alpha, beta)
        t = dist.sample((batch_size,)).to(self.device)
    
    return t
```

**关键点**:
- Flow Matching 使用**连续时间**: t ∈ [0, 1]
- t=0: 数据分布 (x_0)
- t=1: 噪声分布 (x_1)
- 采样策略可调节不同时间段的训练权重

#### Step 3: 采样噪声端点
```python
# 位置: fm_lightning.py::training_step()
x1 = torch.randn_like(x0, device=self.device)  # x_1 ~ N(0, I)
```

**作用**:
- 采样噪声端点 x_1，代表 t=1 时的状态
- x_1 ~ N(0, I) 标准高斯分布

#### Step 4: 计算插值路径和目标速度
```python
# 位置: fm_lightning.py::training_step()
x_t, v_star = self.path_fn(x0, x1, t)
```

**核心实现 (Linear OT Path)**:
```python
# 位置: fm/paths.py::linear_ot_path()
def linear_ot_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
    """
    线性最优传输路径 (Linear Optimal Transport)
    
    这是 Flow Matching 的默认路径，提供最简单直接的插值。
    
    路径: x_t = (1-t)·x_0 + t·x_1
    速度: v* = dx_t/dt = x_1 - x_0 (常数!)
    """
    # 扩展 t 用于广播
    t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
    
    # 线性插值
    x_t = (1 - t_expanded) * x0 + t_expanded * x1
    
    # 常数速度 (对于线性路径)
    v_star = x1 - x0
    
    return x_t, v_star
```

**数学原理**:

1. **概率路径**: 
   ```
   p_t(x) = (1-t)·p_0(x) + t·p_1(x)
   ```
   其中 p_0(x) 是数据分布，p_1(x) = N(0, I) 是噪声分布

2. **条件路径 (Conditional Path)**:
   ```
   x_t | x_0 = (1-t)·x_0 + t·x_1,  x_1 ~ N(0, I)
   ```

3. **速度场 (Velocity Field)**:
   ```
   v*(x_t, t) = dx_t/dt = x_1 - x_0
   ```

4. **连续方程 (Continuity Equation)**:
   ```
   ∂p_t/∂t + ∇·(p_t·v_t) = 0
   ```

**关键特性**:
- **线性插值**: 最简单的路径，从 x_0 直线到 x_1
- **常数速度**: 对于给定的 (x_0, x_1) 对，速度 v* 不依赖于时间 t
- **最优传输**: 最小化传输成本 (Wasserstein-2 距离)

#### Step 5: 可选随机性 (Stochastic FM)
```python
# 位置: fm_lightning.py::training_step()
if self.sfm_sigma and self.sfm_sigma > 0:
    v_star = add_stochasticity(v_star, sigma=float(self.sfm_sigma))
```

**实现**:
```python
# 位置: fm/paths.py::add_stochasticity()
def add_stochasticity(v: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """
    将确定性 ODE 转换为 SDE:
    dx = v(x,t)dt + σ·dW
    """
    if sigma <= 0:
        return v
    
    noise = torch.randn_like(v) * sigma
    return v + noise
```

**作用**:
- 添加随机性可以提高生成多样性
- 纯消融研究，通常 sigma=0 (确定性)

#### Step 6: 计算条件特征
```python
# 位置: fm_lightning.py::training_step()
condition_dict = self.model.condition(processed_batch)
processed_batch.update(condition_dict)
```

**与 DDPM 类似**: 提取场景和文本条件特征

#### Step 7: 条件 Dropout (CFG 训练)
```python
# 位置: fm_lightning.py::training_step()
if self.use_cfg and self.training:
    drop_mask = torch.rand(B, device=self.device) < self.cond_drop_prob
    if drop_mask.any():
        for key in ['scene_cond', 'text_cond']:
            if key in processed_batch and processed_batch[key] is not None:
                mask_expanded = drop_mask.view(B, *([1] * (processed_batch[key].dim() - 1)))
                processed_batch[key] = processed_batch[key] * (~mask_expanded).float()
```

**作用**:
- 训练 Classifier-Free Guidance (CFG)
- 随机丢弃条件 (概率 `cond_drop_prob`, 通常 0.1)
- 使模型学习条件和无条件预测

#### Step 8: 模型预测速度场
```python
# 位置: fm_lightning.py::training_step()
v_pred = self.model(x_t, t, processed_batch)  # [B, num_grasps, D]
```

**模型输入**:
- `x_t`: 插值状态 [B, num_grasps, D]
- `t`: 连续时间 [B]，值在 [0, 1]
- `processed_batch`: 条件特征

**模型输出**:
- `v_pred`: 预测的速度场 v_θ(x_t, t)

**关键区别**:
- DDPM 预测**噪声** ε_θ(x_t, t)
- Flow Matching 预测**速度** v_θ(x_t, t)

#### Step 9: 计算损失
```python
# 位置: fm_lightning.py::training_step()
if self.t_weight is not None:
    # 时间加权损失
    weight = self._compute_time_weight(t)
    loss = (F.mse_loss(v_pred, v_star, reduction='none') * weight.view(-1, 1, 1)).mean()
else:
    # 标准 MSE 损失
    loss = F.mse_loss(v_pred, v_star)
```

**基础损失**:
```
L_FM = E_{t, x_0, x_1} [‖v_θ(x_t, t) - v*‖²]
```
其中:
- v_θ(x_t, t): 模型预测的速度场
- v* = x_1 - x_0: 目标速度 (对于 linear OT)

**可选时间加权**:
```python
# 位置: fm_lightning.py::_compute_time_weight()
def _compute_time_weight(self, t: torch.Tensor) -> torch.Tensor:
    if self.t_weight == 'cosine':
        # 强调中间时间步
        weight = torch.sin(math.pi * t)
    elif self.t_weight == 'beta':
        # Beta 权重
        weight = 4 * t * (1 - t)
    else:
        weight = torch.ones_like(t)
    
    return weight
```

#### Step 10: 兼容性处理 (可选)
```python
# 位置: fm_lightning.py::training_step()
if self.keep_ddpm_interface:
    # 为了与现有损失函数兼容，计算 pred_x0
    t_expanded = t.view(-1, 1, 1)
    if self.path_type == 'linear_ot':
        # 从速度反推 x_0: x_0 = x_t - t·v
        pred_x0 = x_t - t_expanded * v_pred
    
    pred_dict = {
        "pred_pose_norm": pred_x0,
        "pred_noise": v_pred,
        "noise": v_star
    }
    
    # 使用现有的 criterion 计算额外损失
    loss_dict = self.criterion(pred_dict, processed_batch, mode='train')
```

**作用**:
- 复用 DDPM 的损失函数框架
- 可以同时计算姿态空间的额外损失项

---

## 核心差异对比

### 1. 时间表示

| 方面 | DDPM | Flow Matching |
|------|------|---------------|
| **时间类型** | 离散: t ∈ {0, 1, ..., T-1} | 连续: t ∈ [0, 1] |
| **典型 T 值** | 100 或 1000 | N/A (连续) |
| **采样策略** | 随机整数采样 | 均匀/余弦/Beta 分布 |
| **时间编码** | 整数索引 → 位置编码 | 标量值 → 位置编码 |

### 2. 前向过程

| 方面 | DDPM | Flow Matching |
|------|------|---------------|
| **数学形式** | x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε | x_t = (1-t)·x_0 + t·x_1 |
| **噪声系数** | 基于扩散调度 (β_t, ᾱ_t) | 简单线性插值 |
| **噪声添加** | 累积噪声 (马尔可夫过程) | 单次插值 |
| **可逆性** | 需要后向过程 | 直接 ODE 反向 |

### 3. 模型预测

| 方面 | DDPM | Flow Matching |
|------|------|---------------|
| **预测目标** | 噪声 ε_θ(x_t, t) 或 x_0 | 速度场 v_θ(x_t, t) |
| **物理意义** | 去噪方向 | 流动方向 |
| **时间依赖** | 依赖时间步 t | 依赖连续时间 t |

### 4. 损失函数

#### DDPM 损失:
```python
# 预测噪声
L_DDPM = E_t,ε [‖ε_θ(x_t, t) - ε‖²]

# 或预测 x_0
L_DDPM = E_t,ε [‖x_θ(x_t, t) - x_0‖²]
```

**特点**:
- 噪声预测: 优化去噪方向
- 可以添加时间权重 (如 SNR 权重)
- 需要扩散调度参数

#### Flow Matching 损失:
```python
# 预测速度
L_FM = E_{t,x_0,x_1} [‖v_θ(x_t, t) - v*‖²]

# 对于 linear OT: v* = x_1 - x_0 (常数)
```

**特点**:
- 速度预测: 优化流动方向
- 目标速度简单 (线性路径下为常数)
- 无需复杂的扩散调度

### 5. 采样过程

#### DDPM 采样:
```python
# 反向扩散: 从 x_T ~ N(0,I) 到 x_0
for t in reversed(range(T)):
    # 预测噪声
    ε_pred = model(x_t, t)
    
    # 计算后验均值和方差
    μ_t = posterior_mean(x_t, ε_pred, t)
    σ_t = posterior_std(t)
    
    # 采样 (SDE)
    x_{t-1} = μ_t + σ_t·z,  z ~ N(0,I)
```

**特点**:
- **离散步骤**: T 步反向采样
- **随机过程** (SDE): 每步添加噪声
- **确定性也可**: DDIM 使用确定性采样

#### Flow Matching 采样:
```python
# ODE 积分: 从 x_1 ~ N(0,I) 到 x_0
def ode_func(x, t):
    return model(x, t)  # 速度场

x_0 = odeint(ode_func, x_1, t=[1, 0], solver='rk4')
```

**特点**:
- **ODE 积分**: 使用数值求解器 (RK4, Dopri5)
- **确定性**: 给定 x_1，结果唯一
- **自适应步长**: 可以根据误差调整步数

### 6. 计算复杂度

| 方面 | DDPM | Flow Matching |
|------|------|---------------|
| **训练** | O(1) per step | O(1) per step |
| **采样** | T 次模型评估 (固定) | NFE 次评估 (可调) |
| **典型 NFE** | 100-1000 | 20-50 |
| **效率** | 较慢 | 较快 |

### 7. 理论基础

#### DDPM:
- 基于**扩散过程**和**SDE 理论**
- 训练目标来自**变分下界** (ELBO)
- 噪声调度影响生成质量

#### Flow Matching:
- 基于**连续归一化流** (CNF)
- 训练目标来自**连续方程**
- 路径选择影响传输成本

---

## 数学原理

### DDPM 数学框架

#### 1. 前向扩散过程
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)
```

**累积形式**:
```
q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)

其中 ᾱ_t = ∏_{s=1}^{t} (1-β_s)
```

#### 2. 反向过程
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²·I)
```

**预测均值**:
```
μ_θ(x_t, t) = (1/√α_t) [x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, t)]
```

#### 3. 训练目标
```
L_simple = E_{t,x_0,ε} [‖ε - ε_θ(x_t, t)‖²]
```

### Flow Matching 数学框架

#### 1. 连续方程
```
∂p_t/∂t + ∇·(p_t·v_t) = 0
```

其中 v_t 是速度场，满足:
```
dx_t = v_t(x_t, t)·dt
```

#### 2. 条件流匹配
给定条件路径 x_t | x_0，定义条件速度:
```
u_t(x | x_0) = d/dt x_t | x_0
```

**边缘速度场**:
```
v_t(x) = E_{x_0~p_0} [u_t(x | x_0) | x_t = x]
```

#### 3. 训练目标
```
L_CFM = E_{t,x_0,x_t|x_0} [‖v_θ(x_t, t) - u_t(x_t | x_0)‖²]
```

**对于 Linear OT**:
```
x_t | x_0 = (1-t)·x_0 + t·x_1,  x_1 ~ N(0, I)
u_t(x_t | x_0) = x_1 - x_0
```

因此:
```
L_FM = E_{t,x_0,x_1} [‖v_θ(x_t, t) - (x_1 - x_0)‖²]
```

---

## 实践建议

### 何时使用 DDPM?
- ✅ 需要随机采样 (SDE)
- ✅ 理论成熟，社区支持多
- ✅ 对采样速度要求不高
- ✅ 想要可控的噪声调度

### 何时使用 Flow Matching?
- ✅ 需要快速采样 (更少 NFE)
- ✅ 确定性生成
- ✅ 简洁的训练目标
- ✅ 灵活的概率路径设计

### 两者结合
- 可以混合使用: DDPM 训练 → FM 采样
- 共享模型架构 (如 DiT)
- 利用各自优势

---

## 代码位置索引

### DDPM 关键代码
- 训练入口: `models/diffuser_lightning.py::training_step()`
- 损失计算: `models/diffuser_lightning.py::_compute_loss()`
- 前向扩散: `models/utils/diffusion_core.py::q_sample()`
- 反向采样: `models/utils/diffusion_core.py::p_sample_loop()`
- 噪声调度: `models/utils/diffusion_utils.py::make_schedule_ddpm()`

### Flow Matching 关键代码
- 训练入口: `models/fm_lightning.py::training_step()`
- 路径函数: `models/fm/paths.py::linear_ot_path()`
- 时间采样: `models/fm_lightning.py::_sample_time()`
- ODE 采样: `models/fm_lightning.py::_integrate_ode()`
- 速度预测: `models/fm_lightning.py::_velocity_fn()`

---

## 总结

### DDPM 特点
- **优势**: 理论成熟、效果稳定、社区资源丰富
- **劣势**: 采样慢 (需要 T 步)、调度复杂
- **核心**: 预测噪声，逐步去噪

### Flow Matching 特点
- **优势**: 采样快 (自适应 NFE)、训练简单、数学优雅
- **劣势**: 相对较新、资源较少
- **核心**: 预测速度，ODE 积分

### 统一视角
两者本质上都在学习**从噪声到数据的映射**，只是:
- DDPM: 通过噪声预测学习去噪方向
- Flow Matching: 通过速度预测学习流动方向

选择哪个取决于具体需求和权衡！

---

**最后更新**: 2025-10-27  
**作者**: GitHub Copilot  
**项目**: SceneLeapUltra
