# ObjectCentricGraspDataset 坐标系统对比

## 可视化对比

### v1.0: OMF坐标系（已废弃）
```
        Y
        ↑
        |     物体mesh
        |    ╱────╲
        |   ╱      ╲
        |  │  obj   │  ← 在OMF原点
        |   ╲      ╱
        |    ╲────╱
        |
        └──────────→ X
       ╱
      ╱ Z
```
- 物体可能不在原点
- 抓取位置相对物体坐标系
- 无桌面信息

### v1.1: GW坐标系 + xy居中 + 桌面
```
        Y (GW)
        ↑
        |      🤚 grasps
        |     ╱────╲
        |    ╱ obj  ╲  ← xy居中，z保留
        |   │        │
    ----│--[table]--│---- z=0 桌面平面
        |   (0.12x0.12)
        |
        └──────────────→ X
       ╱
      ╱ Z (保留高度信息)
```
- 物体在xy平面居中
- z坐标保留物体高度
- 包含桌面平面模拟

## 坐标转换详解

### 步骤1: Mesh转换 OMF → GW

**输入**：
- `obj_verts_omf`: 物体顶点在OMF坐标系 (V, 3)
- `obj_pose`: 物体7D姿态 (7,) = [t_obj_gw(3), q_obj_gw(4)]

**转换**：
```python
# 提取旋转和平移
t_obj = obj_pose[:3]        # (3,)
q_obj = obj_pose[3:7]       # (4,) [w,x,y,z]

# 四元数 → 旋转矩阵
R_obj = quaternion_to_matrix(q_obj)  # (3, 3)

# 应用变换
obj_verts_gw = obj_verts_omf @ R_obj.T + t_obj
```

**数学原理**：
```
T_omf_to_gw = [R_obj | t_obj]
              [  0   |   1  ]

P_gw = R_obj @ P_omf + t_obj
     = P_omf @ R_obj.T + t_obj  (向量形式)
```

### 步骤2: xy平面居中归一化

**输入**：
- 任意3D点 `points_gw` (N, 3)
- 中心点 `t_obj` (3,)

**转换**：
```python
# 构造xy平面的offset
offset = [t_obj_x, t_obj_y, 0]

# 应用offset
points_centered = points_gw - offset
```

**效果**：
```
原始 GW 坐标:
  物体中心: (t_obj_x, t_obj_y, t_obj_z)
  
居中后:
  物体中心: (0, 0, t_obj_z)  ← xy居中，z不变
```

### 步骤3: 添加桌面平面

**桌面定义**：
```python
# 4个顶点（正方形，居中在xy原点）
half_size = 0.06  # 0.12 / 2
table_verts = [
    (-0.06, -0.06, 0),  # 左下
    (0.06, -0.06, 0),   # 右下
    (0.06, 0.06, 0),    # 右上
    (-0.06, 0.06, 0)    # 左上
]

# 2个三角形
table_faces = [
    [V+0, V+1, V+2],  # 下三角
    [V+0, V+2, V+3]   # 上三角
]
```

**合并mesh**：
```python
combined_verts = torch.cat([obj_verts_centered, table_verts], dim=0)
combined_faces = torch.cat([obj_faces, table_faces], dim=0)
```

## 手部姿态处理

### v1.0 处理链
```
grasp_qpos(GW) [P_gw, Q_gw, Joints]
    ↓ transform_to_object_centric
[P_omf, Q_omf, Joints]
    ↓ revert_leap_qpos
[P_omf', Q_omf', Joints]
    ↓ (保持OMF坐标系)
最终输出
```

### v1.1 处理链
```
grasp_qpos(GW) [P_gw, Q_gw, Joints]
    ↓ revert_leap_qpos (直接在GW)
[P_gw', Q_gw', Joints]
    ↓ xy居中归一化
P_centered = P_gw' - [t_obj_x, t_obj_y, 0]
    ↓
[P_centered, Q_gw', Joints]
    ↓ 重排序
[P_centered, Joints, Q_gw']  (最终输出格式)
```

## 数据特征对比

| 特征 | v1.0 | v1.1 |
|------|------|------|
| 坐标系 | OMF | GW + xy居中 |
| 物体位置 | 在OMF原点附近 | xy平面原点，z保留 |
| z坐标含义 | OMF坐标系的z | 物体高度（相对桌面） |
| 桌面 | 无 | 0.12x0.12平面，z=0 |
| scene_pc来源 | 仅物体表面 | 物体表面 + 桌面 |
| 物理真实性 | 中等 | 高（模拟桌面） |

## 优势分析

### v1.1 相比v1.0的改进

1. **更自然的坐标系**
   - 使用抓取生成时的原始坐标系
   - 避免额外的GW→OMF转换
   - 减少累积误差

2. **保留高度信息**
   - z坐标有明确的物理意义（距桌面高度）
   - 帮助模型学习高度依赖的抓取策略
   - 符合"物体在桌上"的物理约束

3. **桌面环境模拟**
   - 点云包含桌面信息
   - 更接近真实RGB-D扫描
   - 提供上下文信息

4. **训练稳定性**
   - xy居中消除平移不变性
   - z保留提供额外监督信号
   - 桌面提供环境上下文

## 注意事项

### 兼容性
- ✅ 数据格式与SceneLeapPlusDataset完全兼容
- ✅ collate_fn保持不变
- ✅ 无需修改训练代码

### 桌面尺寸选择
- 当前设置：0.12x0.12米（12cm x 12cm）
- 考虑因素：
  - 需要大于大部分物体的xy投影
  - 不能太大，避免桌面点占比过高
  - 当前值适合小型桌面物体（杯子、碗等）

### 采样点分布
- 桌面是平面，采样点会分布在z=0
- 物体表面采样点分布在物体几何上
- pytorch3d的sample_points_from_meshes会按面积均匀采样
- 桌面和物体的点数比例取决于表面积比

## 示例

### 坐标值示例

假设一个物体：
- obj_pose: t_obj = [0.5, 0.3, 0.15], q_obj = [1, 0, 0, 0]
- mesh底部z_min = -0.02（OMF坐标系）
- mesh顶部z_max = 0.08（OMF坐标系）

**v1.0 输出（OMF）**：
```
obj_verts: z范围 [-0.02, 0.08]
scene_pc:  z范围 [-0.02, 0.08]  (只有物体)
hand_poses: 围绕物体中心
```

**v1.1 输出（GW + xy居中）**：
```
转换后(GW): t_obj将mesh整体平移到[0.5, 0.3, 0.15]
            z范围变为 [0.13, 0.23]

xy居中后:   x,y居中到0，z不变
            obj_verts: 中心约 [0, 0, 0.18]
            z范围仍为 [0.13, 0.23]

添加桌面:   table_verts: z=0, xy范围[-0.06, 0.06]

scene_pc:   包含物体点(z: 0.13-0.23) + 桌面点(z: 0)
            xy居中在原点，z范围 [0, 0.23]

hand_poses: xy居中，z保留，围绕物体分布
```

## 可视化效果

使用可视化脚本：
```bash
python scripts/vis_objectcentric_dataset.py --num_grasps 8 --colormap height
```

应该观察到：
1. 物体在xy平面居中（x,y坐标围绕0分布）
2. z坐标保持物理高度（物体底部 > 0）
3. z=0处有桌面平面（可以用颜色映射看到）
4. 抓取位置分布在物体周围

---

**总结**：v1.1版本通过保留z高度和添加桌面平面，提供了更真实、更符合物理约束的训练数据。

