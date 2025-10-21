# ObjectCentricGraspDataset v1.1 更新说明

## 更新日期
2025-10-21

## 核心变更

### 1. 坐标系统变更：从OMF改为GW+居中归一化

#### v1.0 逻辑（已废弃）
```
加载grasp_qpos(GW) + obj_pose(GW)
    ↓
GW → OMF 坐标转换
    ↓
LEAP反转换
    ↓
所有数据在OMF坐标系
```

#### v1.1 新逻辑
```
加载grasp_qpos(GW) + obj_pose(GW)
    ↓
在GW坐标系直接做LEAP反转换（跳过GW→OMF）
    ↓
mesh从OMF转到GW：V_gw = V_omf @ R_obj.T + t_obj
    ↓
xy平面居中：所有数据 - [t_obj_x, t_obj_y, 0]
    ↓
添加桌面平面
    ↓
最终：物体在xy平面居中，z保留高度
```

### 2. xy平面居中（z保持不变）

**修改的方法**：
- `_center_normalize()`
- `_center_normalize_hand_poses()`

**关键代码**：
```python
# 只在xy平面居中，z保持不变
offset = torch.zeros(3, dtype=center.dtype, device=center.device)
offset[:2] = center[:2]  # 只使用x, y
return points - offset.unsqueeze(0)
```

**物理意义**：
- ✅ 符合物体放在桌上的物理约束
- ✅ z坐标保留物体高度信息
- ✅ 帮助模型学习合理的抓取高度

### 3. 添加桌面平面

**新增方法**：`_add_table_plane()`

**桌面规格**：
- 尺寸：0.12 x 0.12 米（正方形）
- z坐标：0（与居中后的物体底部对齐）
- 结构：4个顶点 + 2个三角形面片

**顶点坐标**（居中在xy平面原点）：
```
(-0.06, -0.06, 0)  # 左下
(0.06, -0.06, 0)   # 右下
(0.06, 0.06, 0)    # 右上
(-0.06, 0.06, 0)   # 左上
```

**面片**：
```
三角形1: [V+0, V+1, V+2]
三角形2: [V+0, V+2, V+3]
其中V是原物体顶点数
```

**影响**：
- scene_pc 中包含桌面采样点
- obj_verts 只包含物体顶点（不含桌面）
- 更真实地模拟物体放置场景

## 数据流对比

### v1.0 数据流
```
.npy文件(GW) → GW→OMF → LEAP反转换 → mesh(OMF) → 采样 → 返回(OMF)
```

### v1.1 数据流
```
.npy文件(GW) → LEAP反转换(GW) → mesh(OMF→GW) → xy居中 → +桌面 → 采样 → 返回(GW_centered)
```

## API变更

### 内部变更
- `_load_and_process_hand_pose_data()` 返回类型变更：
  - v1.0: 返回 `Dict[object_code, Tensor]`
  - v1.1: 返回 `Tuple[Dict, Dict]` - (hand_pose_data, obj_pose_data)

### 新增方法
1. `_transform_mesh_to_grasp_world()` - mesh OMF→GW转换
2. `_add_table_plane()` - 添加桌面平面

### 修改的方法
1. `_center_normalize()` - 只在xy平面居中
2. `_center_normalize_hand_poses()` - 只移动xy位置

### 外部API（保持兼容）
- `__init__()` 参数不变
- `__getitem__()` 返回格式不变
- `collate_fn()` 不变

## 测试更新

新增测试用例：
1. ✅ xy平面居中验证（x,y中心接近0）
2. ✅ z坐标保留验证
3. ✅ 桌面点存在性验证（z=0附近有点）
4. ✅ 桌面点数统计

## 使用建议

### 适用场景
- ✅ 需要保留物体高度信息的任务
- ✅ 桌面抓取场景模拟
- ✅ 需要xy位置不变性的训练

### 不适用场景
- ❌ 需要完全3D居中的任务（考虑使用v1.0或SceneLeapPlus）
- ❌ 非桌面抓取场景（空中抓取等）

## 迁移指南

从v1.0迁移到v1.1：
1. **无需修改训练代码** - 数据格式完全兼容
2. **配置文件** - 可以直接使用
3. **模型输入** - scene_pc仍为(N, 3)，无需修改模型

## 性能影响

- 🟢 **加载速度**：基本无变化
- 🟢 **内存占用**：增加桌面顶点（+4顶点，+2面片），几乎可忽略
- 🟢 **采样质量**：更真实的环境模拟

## 验证方法

运行测试脚本：
```bash
cd /home/xiantuo/source/grasp/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python tests/test_objectcentric_dataset.py
```

预期输出：
```
测试5: 验证居中归一化和桌面平面
✓ 点云中心: [-0.0xxx, -0.0xxx, z]  # xy接近0
✓ 顶点中心: [-0.00xx, -0.00xx, z]  # xy更接近0
✓ xy平面居中验证通过
桌面点统计:
  z=0附近的点数: >0
✓ 桌面平面验证通过
```

## 可视化验证

使用可视化脚本查看效果：
```bash
python scripts/vis_objectcentric_dataset.py --num_grasps 8
```

应该看到：
- 物体在xy平面居中
- z轴保留物体高度
- z=0处有桌面平面
- 抓取位置围绕物体分布

## v1.1.1 更新 (2025-10-21)

### 新增功能

1. **可控的物体/桌面采样比例**
   - 新增参数：`object_sampling_ratio`（默认0.8）
   - 支持分别从物体和桌面采样，按比例混合
   - 采样后随机打乱点顺序，避免顺序偏差

2. **可配置桌面尺寸**
   - 新增参数：`table_size`（默认0.4米）
   - 从0.12米增加到0.4米，更好地覆盖物体周围区域

3. **改进的采样方法**
   - `_sample_points_from_mesh_separated()`: 分离采样物体和桌面
   - `_create_table_plane()`: 独立创建桌面mesh（不与物体合并）

### 配置示例

```yaml
object_sampling_ratio: 0.8  # 80%物体 + 20%桌面
table_size: 0.4             # 40cm x 40cm 桌面
max_points: 4096            # 总点数
```

实际采样点数：
- 物体点：4096 × 0.8 = 3276 点
- 桌面点：4096 × 0.2 = 820 点

### 影响

- ✅ 更灵活的环境建模
- ✅ 可根据任务调整物体/环境比例
- ✅ 更大的桌面提供更丰富的上下文

## 未来计划

可能的增强功能：
- [x] 可配置桌面尺寸 ✅ (v1.1.1)
- [x] 可控采样比例 ✅ (v1.1.1)
- [ ] 支持多层桌面或复杂环境
- [ ] 可选的完全3D居中模式
- [ ] 桌面纹理/材质信息

---

**维护者**: AI Assistant  
**版本**: v1.1  
**最后更新**: 2025-10-21

