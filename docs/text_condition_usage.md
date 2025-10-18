# 文本特征开关使用说明

## 概述

现在你可以通过配置文件方便地控制是否在训练中使用文本特征（CLIP模型）。这个功能允许你：
- 仅使用object_mask进行目标物体引导
- 节省GPU内存（不加载CLIP模型）
- 对比文本引导和掩码引导的效果

## 快速开始

### 方式1：命令行参数（推荐）

**启用文本特征（默认）：**
```bash
python train_lightning.py
```

**禁用文本特征：**
```bash
python train_lightning.py use_text_condition=False use_object_mask=True
```

### 方式2：修改配置文件

编辑 `config/config.yaml`：

```yaml
# 禁用文本特征
use_text_condition: &use_text_condition False
use_object_mask: &use_object_mask True

# 或启用文本特征（默认）
use_text_condition: &use_text_condition True
use_object_mask: &use_object_mask True
```

## 实现细节

### 修改的文件

1. **config/config.yaml** (第20行)
   - 添加了全局参数 `use_text_condition: &use_text_condition True`

2. **config/model/diffuser/decoder/unet.yaml** (第20行)
   - 从 `use_text_condition: true` 改为 `use_text_condition: ${use_text_condition}`

3. **config/model/diffuser/decoder/dit.yaml** (第17行)
   - 从 `use_text_condition: true` 改为 `use_text_condition: ${use_text_condition}`

4. **models/diffuser_lightning.py** (第458-470行)
   - 修复了checkpoint加载逻辑，只在启用文本特征时才初始化CLIP模型

### 工作原理

当 `use_text_condition=False` 时：

1. **数据加载阶段**：数据集仍会加载positive_prompt和negative_prompts，但不会影响性能
2. **模型条件处理**：`condition()` 方法会检测到 `use_text_condition=False`，直接返回不包含文本特征的条件字典
3. **CLIP模型**：不会被初始化（懒加载机制），节省约1-2GB GPU内存
4. **前向传播**：模型内部的交叉注意力机制会检测到 `text_cond=None`，自动跳过文本融合
5. **损失计算**：neg_loss会被自动跳过（因为neg_pred为None）

### 关键代码位置

- **UNet条件处理**：`models/decoder/unet_new.py:287-288`
- **DiT条件处理**：`models/decoder/dit.py:663-664`
- **文本编码器懒加载**：
  - `models/decoder/unet_new.py:324-335`
  - `models/decoder/dit.py:739-754`
- **Checkpoint加载**：`models/diffuser_lightning.py:460-468`

## 注意事项

### 重要建议

1. **必须启用object_mask**：禁用文本特征时，务必确保 `use_object_mask=True`，否则模型将缺少目标物体的引导信息

2. **兼容性**：
   - 现有checkpoint可以正常加载
   - 如果checkpoint包含text_encoder权重但设置了 `use_text_condition=False`，这些权重会被安全地忽略
   - 从启用文本特征的checkpoint恢复训练到禁用文本特征的训练是支持的

3. **性能对比**：建议分别训练两个版本进行对比：
   ```bash
   # 版本1：使用文本特征
   python train_lightning.py use_text_condition=True use_object_mask=True
   
   # 版本2：不使用文本特征
   python train_lightning.py use_text_condition=False use_object_mask=True
   ```

## 预期效果

### 启用文本特征时 (`use_text_condition=True`)

- ✅ CLIP模型被加载
- ✅ positive_prompt通过CLIP编码为文本特征
- ✅ 文本特征通过交叉注意力融入模型
- ✅ 支持负样本引导（如果 `use_negative_prompts=True`）
- 📊 GPU内存占用：约多1-2GB

### 禁用文本特征时 (`use_text_condition=False`)

- ✅ CLIP模型不会被加载
- ✅ 仅使用scene_pc和object_mask作为条件
- ✅ text_cond在condition_dict中为None
- ✅ 模型自动跳过文本相关的处理
- 📊 GPU内存占用：节省约1-2GB

## 验证方法

### 检查日志

训练开始时，查看日志输出：

**启用文本特征：**
```
[INFO] Text encoder lazily initialized on device: cuda:0
[INFO] Text encoder initialized for checkpoint loading
```

**禁用文本特征：**
```
[INFO] Skipping text encoder initialization (use_text_condition=False)
```

### 监控GPU内存

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

对比两种模式下的GPU内存占用差异。

## 故障排查

### 问题1：训练效果很差

**可能原因**：禁用了文本特征但也禁用了object_mask
**解决方法**：确保 `use_object_mask=True`

### 问题2：加载checkpoint时出错

**可能原因**：checkpoint包含text_encoder权重，但设置strict=True
**解决方法**：代码已经处理了这种情况，会自动跳过text_encoder的权重加载

### 问题3：想查看条件处理的详细过程

**解决方法**：在 `models/decoder/unet_new.py` 或 `models/decoder/dit.py` 的 `condition()` 方法中添加日志：
```python
logging.info(f"use_text_condition={self.use_text_condition}, positive_prompt in data={('positive_prompt' in data)}")
```

## 总结

这个实现采用最小改动的方式，充分利用了现有代码的条件判断逻辑，具有以下优势：

- ✅ **最小改动**：只修改了4个文件，共约20行代码
- ✅ **向后兼容**：默认行为保持不变（`use_text_condition=True`）
- ✅ **鲁棒安全**：利用现有的fallback机制，不会引入新的错误
- ✅ **易于使用**：通过配置文件或命令行参数即可切换
- ✅ **节省资源**：禁用文本特征时不加载CLIP模型

现在你可以轻松地对比使用文本特征和仅使用object_mask的效果差异！

