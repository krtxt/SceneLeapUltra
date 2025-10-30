# Scene Token 提取策略对比分析

> 针对 SceneLeapUltra 抓取生成任务的点云编码器改进方案

## 📋 任务背景

- **输入**: `(B, N, 3)` 点云，N=8192
- **输出**: `(B, K, d_model)` scene tokens，K=128, d_model=512
- **应用场景**: 作为 Double Stream DiT 的条件输入，用于控制抓取生成
- **关键需求**: 
  - 覆盖场景的全局空间结构
  - 突出可抓取区域（边缘、角落、表面特征）
  - 高效的attention交互（与grasp tokens）

---

## 🔍 现有方案回顾

### ① last_layer - 直接使用PTv3最后一层

```python
xyz_out, feat_out = self._strategy_last_layer(xyz_sparse, feat_sparse)
```

**特点**:
- ✅ 实现简单，速度快
- ✅ 直接利用PTv3的层次化特征
- ⚠️ 依赖grid_size和stride配置
- ❌ 可能丢失关键几何细节

**适用场景**: 快速原型，baseline实验

---

### ② fps - 最远点采样

```python
xyz_out, feat_out = self._strategy_fps(xyz_sparse, feat_sparse)
```

**特点**:
- ✅ 空间分布均匀
- ✅ 保证覆盖率
- ✅ 算法成熟稳定
- ❌ 不考虑任务相关性
- ❌ 不关注几何特征

**适用场景**: 需要均匀采样的场景，如场景重建

---

### ③ grid - 规则网格聚合

```python
xyz_out, feat_out = self._strategy_grid(xyz_sparse, feat_sparse, orig_coords)
```

**特点**:
- ✅ 结构化表示
- ✅ 易于理解和调试
- ✅ 计算稳定
- ❌ 固定分辨率，不适应复杂度
- ❌ 空网格浪费token资源

**适用场景**: 规则场景，需要结构化表示

---

### ④ learned - 可学习的Cross-Attention Tokenizer

```python
xyz_out, feat_out = self._strategy_learned(xyz_sparse, feat_sparse)
# 使用 query tokens + cross-attention
```

**特点**:
- ✅ 端到端学习
- ✅ 灵活，可学习任务相关特征
- ✅ 类似TokenLearner/Perceiver
- ⚠️ 需要大量训练数据
- ❌ 可能过拟合

**适用场景**: 有充足训练数据，追求最优性能

---

### ⑤ multiscale - 多尺度特征融合

```python
xyz_out, feat_out = self._strategy_multiscale(pos, data_dict)
# 从encoder的多个阶段提取特征
```

**特点**:
- ✅ 包含粗细不同尺度的信息
- ✅ 层次化表示
- ✅ 适合复杂场景
- ⚠️ 计算成本高（多次前向传播）
- ⚠️ tokens来源不同层，特征分布可能不一致

**适用场景**: 复杂场景，需要多尺度信息

---

## 🆕 新提案

### ⑥ surface_aware - 表面感知采样 ⭐⭐⭐⭐⭐

**核心思想**: 
- 计算局部几何特征（曲率估计）
- 优先采样高曲率区域（边缘、角落）
- 结合FPS保证空间覆盖

**伪代码**:
```python
# 计算曲率（使用k近邻距离方差作为代理）
curvature = compute_local_curvature(xyz, features, k=16)

# 60% tokens来自高曲率区域
high_curv_tokens = topk_sample(xyz, curvature, k=0.6*K)

# 40% tokens使用FPS均匀采样
uniform_tokens = fps_sample(xyz_remaining, k=0.4*K)

# 合并
return concat(high_curv_tokens, uniform_tokens)
```

**优势**:
- ✅ **直接针对抓取任务**: 边缘和角落是关键抓取点
- ✅ **保留几何细节**: 不会丢失重要表面特征
- ✅ **不需要训练**: 基于几何启发式
- ✅ **计算高效**: 只增加曲率计算开销

**劣势**:
- ⚠️ 需要调整曲率阈值
- ⚠️ 对噪声点云可能敏感

**推荐指数**: ⭐⭐⭐⭐⭐  
**实现难度**: 中等  
**计算开销**: 低-中

---

### ⑦ hybrid - 混合策略 ⭐⭐⭐⭐⭐

**核心思想**:
- Grid tokens: 提供全局空间结构（均匀覆盖）
- Learned tokens: 提供局部细节和任务相关特征
- 两者互补

**伪代码**:
```python
# 50% tokens来自grid聚合
grid_xyz, grid_feat = grid_aggregate(xyz, features, k=K//2)

# 50% tokens来自learned attention
learned_xyz, learned_feat = learned_aggregate(xyz, features, k=K//2)
  # 内部使用 query_tokens + cross_attention

# 拼接
return concat(grid_xyz, learned_xyz), concat(grid_feat, learned_feat)
```

**优势**:
- ✅ **全局+局部平衡**: 结合结构化和灵活性
- ✅ **与Double Stream配合好**: grid保证覆盖，learned关注重点
- ✅ **可调节比例**: grid_ratio可配置

**劣势**:
- ⚠️ 参数较多（需要训练learned部分）
- ⚠️ 实现稍复杂

**推荐指数**: ⭐⭐⭐⭐⭐  
**实现难度**: 中-高  
**计算开销**: 中

---

### ⑧ graspability_guided - 可抓取性引导 ⭐⭐⭐⭐⭐

**核心思想**:
- 训练一个"可抓取性预测器"
- 预测每个点的抓取分数
- Top-K采样高分区域

**伪代码**:
```python
# 预测抓取性（MLP）
graspability = graspability_head(features)  # (B, N, 1)

# 按抓取性Top-K采样
selected_idx = topk(graspability, k=K)

# 提取tokens
return xyz[selected_idx], features[selected_idx]
```

**优势**:
- ✅ **直接针对任务**: 端到端优化抓取相关性
- ✅ **自动聚焦**: 模型学习重要区域
- ✅ **可监督训练**: 使用抓取成功/失败标注

**劣势**:
- ❌ **需要标注数据**: 抓取成功区域标注
- ⚠️ 可能过拟合训练分布

**推荐指数**: ⭐⭐⭐⭐⭐ (如果有训练数据)  
**实现难度**: 中  
**计算开销**: 低

**训练建议**:
- 监督信号: 成功抓取点附近 → 高分，失败区域 → 低分
- 损失函数: BCELoss 或 FocalLoss
- 可与主任务联合训练

---

### ⑨ hierarchical_attention - 层次化注意力池化 ⭐⭐⭐⭐

**核心思想**:
- 多层逐步下采样（类似Set Transformer）
- 每层使用cross-attention pooling
- 保留层次结构

**伪代码**:
```python
x = features  # (B, N, C)

# Level 1: N → N/4
x = pooling_attention_1(x)  # query tokens: N/4

# Level 2: N/4 → N/16
x = pooling_attention_2(x)  # query tokens: N/16

# Level 3: N/16 → K
x = pooling_attention_3(x)  # query tokens: K

return x
```

**优势**:
- ✅ 渐进式抽象
- ✅ 保留多尺度信息
- ✅ 完全可学习

**劣势**:
- ❌ 计算复杂，慢
- ❌ 参数量大

**推荐指数**: ⭐⭐⭐⭐  
**实现难度**: 高  
**计算开销**: 高

---

### ⑩ adaptive_density - 自适应密度采样 ⭐⭐⭐⭐

**核心思想**:
- 预测每个空间区域的重要性
- 重要区域分配更多tokens
- 类似于adaptive mesh refinement

**伪代码**:
```python
# 将空间划分为R个区域（octree）
regions = spatial_partition(xyz, num_regions=8)

# 预测每个区域的重要性
importance = importance_predictor(features)  # (B, N, 1)
region_importance = aggregate_by_region(importance, regions)

# 按重要性分配tokens
tokens_per_region = region_importance * K / sum(region_importance)

# 在每个区域内FPS采样
tokens = []
for r in regions:
    tokens_r = fps_sample(xyz[r], n=tokens_per_region[r])
    tokens.append(tokens_r)

return concat(tokens)
```

**优势**:
- ✅ 动态资源分配
- ✅ 适应不同复杂度
- ✅ 细节和效率平衡

**劣势**:
- ⚠️ 实现复杂
- ⚠️ 需要训练importance predictor

**推荐指数**: ⭐⭐⭐⭐  
**实现难度**: 高  
**计算开销**: 中-高

---

## 📊 对比总结表

| 方案 | 适用性 | 计算开销 | 需要训练 | 几何感知 | 任务针对性 | 推荐 |
|------|--------|---------|---------|---------|-----------|------|
| ① last_layer | ⭐⭐⭐ | 低 | ❌ | ⚠️ | ❌ | - |
| ② fps | ⭐⭐⭐ | 低 | ❌ | ❌ | ❌ | - |
| ③ grid | ⭐⭐ | 低 | ❌ | ❌ | ❌ | - |
| ④ learned | ⭐⭐⭐⭐ | 低 | ✅ | ⚠️ | ✅ | - |
| ⑤ multiscale | ⭐⭐⭐⭐ | 高 | ❌ | ✅ | ⚠️ | - |
| ⑥ surface_aware | ⭐⭐⭐⭐⭐ | 低-中 | ❌ | ✅✅ | ✅✅ | ✅ |
| ⑦ hybrid | ⭐⭐⭐⭐⭐ | 中 | ⚠️ | ✅ | ✅✅ | ✅ |
| ⑧ graspability | ⭐⭐⭐⭐⭐ | 低 | ✅✅ | ⚠️ | ✅✅✅ | ✅ |
| ⑨ hierarchical | ⭐⭐⭐⭐ | 高 | ✅ | ✅ | ✅ | - |
| ⑩ adaptive | ⭐⭐⭐⭐ | 中-高 | ✅ | ✅ | ✅ | - |

---

## 🎯 实施建议

### 方案一：最小改动，快速验证

**实现: 方案⑥ (Surface-Aware)**

```python
# 在 ptv3_sparse_encoder.py 中添加
def _strategy_surface_aware(self, xyz, feat, orig_coords):
    # 计算曲率
    curvature = self._compute_curvature(xyz)
    
    # 60% 高曲率 + 40% 均匀
    high_curv_xyz, high_curv_feat = self._sample_by_curvature(
        xyz, feat, curvature, n=int(0.6 * self.target_num_tokens)
    )
    uniform_xyz, uniform_feat = self._fps_sample(
        xyz, feat, n=int(0.4 * self.target_num_tokens)
    )
    
    return torch.cat([high_curv_xyz, uniform_xyz], dim=1), \
           torch.cat([high_curv_feat, uniform_feat], dim=2)
```

**优势**:
- 实现简单（~100行代码）
- 不需要训练
- 直接针对抓取任务
- 可作为其他方案的预处理

---

### 方案二：最佳效果，平衡性能

**实现: 方案⑥ + 方案⑦ (Surface + Hybrid)**

两阶段策略：

```python
# Stage 1: Surface-aware粗筛（选2K个候选）
candidates_xyz, candidates_feat = surface_aware_sample(
    xyz_sparse, feat_sparse, 
    target=self.target_num_tokens * 2
)

# Stage 2: Hybrid精选（grid + learned）
# 从候选中选K个最终tokens
final_xyz, final_feat = hybrid_tokenize(
    candidates_xyz, candidates_feat,
    target=self.target_num_tokens,
    grid_ratio=0.5  # 50% grid + 50% learned
)
```

**优势**:
- 兼具几何感知和学习能力
- 全局覆盖 + 局部细节
- learned部分更容易训练（候选已筛选）

---

### 方案三：端到端最优（如果有训练数据）

**实现: 方案⑧ (Graspability-Guided)**

```python
# 添加graspability head
self.graspability_head = nn.Sequential(
    nn.Linear(feat_dim, feat_dim // 2),
    nn.ReLU(),
    nn.Linear(feat_dim // 2, 1),
    nn.Sigmoid()
)

# 训练时的损失
grasp_scores = self.graspability_head(features)  # (B, N, 1)

# 监督信号：抓取成功点附近 = 1, 失败区域 = 0
grasp_labels = compute_grasp_labels(xyz, successful_grasps)  # (B, N, 1)
grasp_loss = F.binary_cross_entropy(grasp_scores, grasp_labels)

# 总损失
total_loss = main_task_loss + lambda_grasp * grasp_loss
```

**优势**:
- 端到端优化
- 直接优化抓取性能
- 可与主任务联合训练

**需要**:
- 抓取成功/失败标注
- 或者使用主任务梯度作为伪标签

---

## 🧪 消融实验建议

### 实验设置

1. **Baseline**: 方案① (last_layer + FPS)
2. **几何改进**: 方案⑥ (surface_aware)
3. **混合策略**: 方案⑦ (hybrid)
4. **任务引导**: 方案⑧ (graspability_guided)
5. **多尺度增强**: 改进方案⑤ (multiscale + FPN)

### 评估指标

1. **主要指标** - 抓取成功率
   - Top-1 success rate
   - Top-5 success rate
   - Coverage (成功抓取的物体比例)

2. **辅助指标** - Token质量
   - Spatial coverage: tokens覆盖多少%的空间
   - Diversity: tokens之间的平均距离
   - Attention entropy: tokens在attention中的分布均匀性

3. **效率指标**
   - 推理速度 (ms/sample)
   - 内存占用
   - FLOPs

### 可视化

- Scene tokens在点云上的分布
- Attention weights热力图（scene tokens ← grasp tokens）
- 成功/失败案例的token分布对比

---

## 💡 与Double Stream Block的配合

你的架构使用16个Double Stream Blocks + 32个Single Stream Blocks：

```
Double Stream:
  Grasp Tokens (M, 512) ←→ Scene Tokens (128, 512)
       ↓                          ↓
   Attention交互              Attention交互
```

**Scene tokens应满足**:

1. **全局覆盖** → 方案②(fps), ③(grid), ⑦(hybrid)的grid部分
2. **局部细节** → 方案⑥(surface), ⑦(hybrid)的learned部分
3. **任务相关** → 方案⑧(graspability), ⑦(hybrid)

**最优方案: ⑦ (Hybrid)**
- Grid tokens: 保证grasp tokens能找到对应空间区域
- Learned tokens: 突出重要抓取区域，提高attention效率

---

## 📝 实现代码示例

见 `tests/test_scene_token_strategies.py` 中的完整实现。

关键模块：
- `SurfaceAwareTokenizer`: 方案⑥
- `HybridTokenizer`: 方案⑦
- `GraspabilityGuidedTokenizer`: 方案⑧
- `HierarchicalAttentionTokenizer`: 方案⑨
- `AdaptiveDensityTokenizer`: 方案⑩

---

## 🚀 下一步

1. **立即可做**:
   - 实现方案⑥ (Surface-Aware) 作为新的token_strategy
   - 在现有数据上评估 vs baseline

2. **短期目标**:
   - 实现方案⑦ (Hybrid)
   - 消融实验：对比①②③⑥⑦

3. **长期目标**:
   - 如果有标注，实现方案⑧
   - 端到端训练grasp score predictor
   - 多任务学习：token selection + grasp generation

---

## 📚 参考文献

- **TokenLearner** (NeurIPS 2021): 可学习的token生成
- **Set Transformer** (ICML 2019): 层次化attention pooling
- **Perceiver** (ICML 2021): cross-attention based tokenization
- **Point Transformer V3** (CVPR 2023): 多尺度点云特征
- **Flow Matching for Generative Modeling** (ICLR 2023): 条件生成

---

**总结**: 
- 如果只选一个: **方案⑥ (Surface-Aware)** - 简单、有效、不需要训练
- 如果选两个: **方案⑥ + ⑦** - 兼具几何感知和学习能力
- 如果有充足资源: **全部实现并做消融实验** - 找到最适合你数据的方案

祝实验顺利！🎉

