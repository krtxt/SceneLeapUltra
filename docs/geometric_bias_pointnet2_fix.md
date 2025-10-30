# 几何注意力偏置与PointNet2兼容性修复

## 问题描述

用户在训练时启用几何注意力偏置功能时遇到形状不匹配错误：

```
RuntimeError: The size of tensor a (128) must match the size of tensor b (8192) at non-singleton dimension 3
```

错误发生在：
```python
File "models/decoder/dit_memory_optimization.py", line 427, in _standard_attention_forward
    scores = scores + attention_bias
```

## 问题根源

### PointNet2的采样机制

PointNet2 通过 **Set Abstraction** 层对点云进行多次下采样：

```python
# models/backbone/pointnet2.py
def forward(self, pointcloud: Tensor):
    """
    Returns:
        xyz: (B, K, 3) - 采样后的点坐标
        features: (B, D, K) - 特征
    """
    xyz, features = self._break_up_pc(pointcloud)
    xyz, features, fps_inds = self.sa1(xyz, features)  # 第1次采样
    xyz, features, fps_inds = self.sa2(xyz, features)  # 第2次采样
    xyz, features, fps_inds = self.sa3(xyz, features)  # 第3次采样
    xyz, features, fps_inds = self.sa4(xyz, features)  # 第4次采样
    return xyz, features
```

**结果**：
- **输入点云**：8192 个点
- **PointNet2输出**：128 个点（经过4层采样）

### 形状不匹配的原因

在原始实现中：

1. **scene_context** (特征)：基于采样后的 128 个点 → `(B, 128, d_model)`
2. **scene_xyz** (坐标)：使用原始的 8192 个点 → `(B, 8192, 3)`
3. **attention scores**：基于 scene_context 计算 → `(B, num_heads, N_grasps, 128)`
4. **geometric_bias**：基于 scene_xyz 计算 → `(B, num_heads, N_grasps, 8192)`

**冲突**：scores 和 bias 的最后一维不匹配（128 vs 8192）

## 解决方案

### 1. 保存采样后的坐标

在 `_prepare_scene_features` 方法中，保存 PointNet2 返回的采样后坐标：

```python
# models/decoder/dit.py - _prepare_scene_features()

# PointNet2 返回采样后的 xyz 和特征
sampled_xyz, scene_feat = self.scene_model(scene_points)

# 保存采样后的 xyz 到 data 中，供几何偏置使用
data['scene_xyz_sampled'] = sampled_xyz  # (B, 128, 3)
```

### 2. 优先使用采样后的坐标

在 forward 方法中，优先使用采样后的坐标：

```python
# models/decoder/dit.py - forward()

if self.use_geometric_bias:
    # 优先使用采样后的 xyz（与 scene_context 的点数匹配）
    if 'scene_xyz_sampled' in data and data['scene_xyz_sampled'] is not None:
        scene_xyz = data['scene_xyz_sampled']  # (B, 128, 3) ✓ 匹配
    else:
        # 回退到原始点云坐标（可能导致形状不匹配）
        scene_xyz = extract_scene_xyz(scene_context, data)  # (B, 8192, 3) ✗ 不匹配
```

### 3. Global-Local Conditioning 兼容性

在使用 Global-Local Conditioning 时，也需要使用采样后的坐标：

```python
# models/decoder/dit.py - forward()

if self.use_global_local_conditioning:
    # 优先使用采样后的 xyz（与 scene_context 匹配）
    if 'scene_xyz_sampled' in data:
        local_scene_xyz = data['scene_xyz_sampled']
    elif 'scene_xyz' in data:
        local_scene_xyz = data['scene_xyz']
    # ...
```

## 修改的文件

### 1. `models/decoder/dit.py`
- `_prepare_scene_features()`: 保存 `scene_xyz_sampled`
- `forward()`: 优先使用 `scene_xyz_sampled`
- Global-Local Conditioning 部分：使用 `local_scene_xyz`

### 2. `models/decoder/dit_fm.py`
- `forward()`: 优先使用 `scene_xyz_sampled`
- Global-Local Conditioning 部分：使用 `local_scene_xyz`

### 3. `models/decoder/dit.py` (DiTBlock)
- `forward()`: 在组合 scene_context 时，也组合对应的 scene_xyz

## 测试验证

修复后，运行相同的训练命令应该不再出现形状不匹配错误：

```bash
python train_lightning.py \
    trainer.devices=1 \
    save_root=./experiments/fm_sweep_condition/exp_03_use_geometric_bias \
    model.decoder.use_geometric_bias=True \
    batch_size=8
```

## 兼容性说明

### ✅ 支持的Backbone

修复后，几何注意力偏置与以下 backbone 兼容：

- **PointNet2**: ✓ 使用采样后的坐标
- **PTv3**: ✓ 需要确保也保存采样后的坐标
- **其他采样型backbone**: ✓ 只要在 condition() 中保存 `scene_xyz_sampled`

### ⚠️ 注意事项

1. **必须保存采样后的坐标**：所有进行点云采样的 backbone 都需要将采样后的 xyz 保存到 `data['scene_xyz_sampled']`

2. **坐标系一致性**：确保采样后的坐标与特征对应，顺序一致

3. **Global-Local Conditioning**：当同时启用 `use_geometric_bias` 和 `use_global_local_conditioning` 时，会在 DiTBlock 中进一步组合坐标

## 预期效果

- ✅ 几何注意力偏置正常工作
- ✅ 形状完全匹配：scores 和 bias 都是 `(B, num_heads, N_grasps, 128)`
- ✅ 支持所有采样型 backbone
- ✅ 与 Global-Local Conditioning 兼容

## 技术细节

### 坐标维度对应关系

| 数据 | 原始实现 | 修复后 | 说明 |
|------|---------|--------|------|
| 输入点云 | (B, 8192, 3) | (B, 8192, 3) | 原始输入 |
| scene_context | (B, 128, d_model) | (B, 128, d_model) | PointNet2特征 |
| scene_xyz | (B, 8192, 3) ❌ | (B, 128, 3) ✅ | **使用采样后坐标** |
| attention scores | (B, 8, N, 128) | (B, 8, N, 128) | 基于scene_context |
| geometric_bias | (B, 8, N, 8192) ❌ | (B, 8, N, 128) ✅ | **形状匹配** |

### 采样流程

```
原始点云 (8192 points)
    ↓
PointNet2 Set Abstraction 层
    ↓ (FPS采样 + 特征提取)
采样点云 (128 points) + 特征 (128 points)
    ↓
保存到 data['scene_xyz_sampled']
    ↓
用于几何偏置计算
    ↓
形状匹配 ✓
```

## 相关Issue

- 几何注意力偏置功能实施：#1
- PointNet2 backbone兼容性：#2

## 版本历史

- **v1.0** (2025-10-26): 初始实现
- **v1.1** (2025-10-26): 修复 PointNet2 兼容性问题
  - 保存采样后的坐标
  - 优先使用采样后坐标进行几何偏置计算
  - 支持 Global-Local Conditioning 场景

