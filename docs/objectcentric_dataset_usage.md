# ObjectCentricGraspDataset 使用文档

## 概述

`ObjectCentricGraspDataset` 是专门为居中归一化的手物对训练设计的数据集类。与 `SceneLeapPlusDataset` 的主要区别在于：

1. **无场景依赖**：不需要场景数据（深度图、RGB图、实例mask等）
2. **点云来源**：从物体mesh表面采样，而非深度图反投影
3. **无碰撞过滤**：直接使用所有成功抓取，无需collision_free_grasp_info
4. **坐标处理**：使用抓取生成时的原始坐标系（GW）+居中归一化

## 坐标处理流程

新的坐标处理逻辑：

```
1. 加载抓取姿态（GW坐标系）+ obj_pose
   ↓
2. 在GW坐标系做LEAP格式反转换（跳过GW→OMF转换）
   ↓
3. 加载mesh（OMF坐标系，已应用mesh_scale）
   ↓
4. 使用obj_pose将mesh从OMF转到GW坐标系
   - 提取: t_obj_gw (3,), q_obj_gw (4,)
   - 转换: V_gw = V_omf @ R_obj.T + t_obj
   ↓
5. 居中归一化：使用t_obj_gw的xy分量作为中心点（只在xy平面移动）
   - obj_verts_centered = obj_verts_gw - [t_obj_x, t_obj_y, 0]
   - hand_poses_centered: 位置xy - [t_obj_x, t_obj_y], z/旋转/关节不变
   ↓
6. 添加桌面平面（0.12x0.12米正方形，z=0）
   - 4个顶点 + 2个三角形面片
   - 模拟物体放置在桌上的真实情况
   ↓
7. 从合并mesh（物体+桌面）采样点云
   ↓
8. 最终：物体在xy平面居中于原点，z保留高度信息，包含桌面点
```

**关键特性**：
- ✅ 使用抓取生成时的原始坐标系（更自然）
- ✅ xy平面居中（z保留），符合物体在桌上的物理约束
- ✅ 添加桌面平面（0.12x0.12米），模拟真实桌面环境
- ✅ 保持手物相对位置关系
- ✅ 物体在xy平面位于原点，z高度保留物理信息

## 数据格式

### 输入数据要求

1. **成功抓取文件** (`succ_grasp_dir`)
   - 格式：`{object_code}.npy` 文件
   - 内容：包含 `grasp_qpos` 和 `obj_pose` 的字典
   - `grasp_qpos`: (N, 23) 在GW坐标系的手部姿态
   - `obj_pose`: (7,) 物体在GW坐标系的7D姿态

2. **物体mesh文件** (`obj_root_dir`)
   - 路径：`{obj_root_dir}/{object_code}/mesh/simplified.obj`
   - 格式：标准OBJ文件

### 输出数据格式

每个样本返回一个字典，包含：

```python
{
    'scene_pc': torch.Tensor,         # (max_points, 3) 物体表面采样点，xyz only
    'hand_model_pose': torch.Tensor,  # (num_grasps, 23) 手部姿态 [P|Joints|Q]
    'se3': torch.Tensor,              # (num_grasps, 4, 4) SE(3)变换矩阵
    'obj_verts': torch.Tensor,        # (V, 3) 物体顶点
    'obj_faces': torch.Tensor,        # (F, 3) 物体面片
    'obj_code': str,                  # 物体代码
    'scene_id': str,                  # = obj_code
    'depth_view_index': int,          # = 0
    'category_id_from_object_index': int,  # = 0
    'positive_prompt': str,           # = obj_code
    'negative_prompts': list,         # 空列表
    'object_mask': torch.Tensor       # (0,) 空mask
}
```

### 批处理后格式

```python
{
    'scene_pc': torch.Tensor,         # (B, max_points, 3)
    'hand_model_pose': torch.Tensor,  # (B, num_grasps, 23)
    'se3': torch.Tensor,              # (B, num_grasps, 4, 4)
    'obj_verts': List[torch.Tensor],  # B个 (V_i, 3) tensors
    'obj_faces': List[torch.Tensor],  # B个 (F_i, 3) tensors
    ...
}
```

## 使用方法

### 基本用法

```python
from datasets.objectcentric_grasp_dataset import ObjectCentricGraspDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = ObjectCentricGraspDataset(
    succ_grasp_dir="/path/to/succ_collect",
    obj_root_dir="/path/to/objects",
    num_grasps=8,           # 每个样本返回8个抓取
    max_points=4096,        # 点云点数
    max_grasps_per_object=1024,
    mesh_scale=0.1,
    grasp_sampling_strategy="farthest_point",
    use_exhaustive_sampling=True,
    exhaustive_sampling_strategy="sequential"
)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

# 迭代数据
for batch in dataloader:
    scene_pc = batch['scene_pc']            # (16, 4096, 3)
    hand_poses = batch['hand_model_pose']   # (16, 8, 23)
    se3 = batch['se3']                      # (16, 8, 4, 4)
    
    # 训练代码...
```

### 使用配置文件

```yaml
# config/data_cfg/objectcentric.yaml
name: objectcentric
mode: object_centric

train:
  succ_grasp_dir: /path/to/succ_collect
  obj_root_dir: /path/to/objects
  num_grasps: ${target_num_grasps}
  max_grasps_per_object: 1024
  mesh_scale: 0.1
  max_points: 4096
  grasp_sampling_strategy: farthest_point
  use_exhaustive_sampling: true
  exhaustive_sampling_strategy: ${exhaustive_sampling_strategy}
  batch_size: ${batch_size}
  num_workers: 16
```

## 配置参数说明

### 必需参数

- `succ_grasp_dir`: 成功抓取数据目录
- `obj_root_dir`: 物体mesh目录
- `num_grasps`: 每个样本返回的抓取数量

### 可选参数

- `max_points`: 物体表面采样点数（默认4096）
- `max_grasps_per_object`: 每个物体最多保留的抓取数（默认None，无限制）
- `mesh_scale`: mesh缩放因子（默认0.1）
- `grasp_sampling_strategy`: 抓取采样策略
  - `"random"`: 随机采样
  - `"first_n"`: 取前N个
  - `"repeat"`: 重复采样
  - `"farthest_point"`: 最远点采样（基于位置）
  - `"nearest_point"`: 最近点采样（基于位置）
- `use_exhaustive_sampling`: 是否使用穷尽采样（默认False）
- `exhaustive_sampling_strategy`: 穷尽采样策略
  - `"sequential"`: 顺序chunks
  - `"random"`: 随机chunks
  - `"interleaved"`: 交错chunks
  - `"chunk_farthest_point"`: 空间FPS chunks
  - `"chunk_nearest_point"`: 空间NPS chunks
- `object_sampling_ratio`: 物体采样点比例（默认0.8，即80%物体+20%桌面）
- `table_size`: 桌面尺寸，正方形边长（默认0.4米）

## 点云采样详解

### 物体/桌面分离采样

数据集使用分离采样策略，从物体mesh和桌面mesh分别采样点，然后按比例混合：

**采样流程**：
```python
# 1. 计算采样点数
num_obj_points = int(max_points * object_sampling_ratio)    # 例如：4096 * 0.8 = 3276
num_table_points = max_points - num_obj_points              # 例如：4096 - 3276 = 820

# 2. 分别采样
obj_points = sample_points_from_meshes(obj_mesh, num_samples=num_obj_points)
table_points = sample_points_from_meshes(table_mesh, num_samples=num_table_points)

# 3. 合并并打乱
combined_points = torch.cat([obj_points, table_points], dim=0)
combined_points = combined_points[torch.randperm(max_points)]
```

**比例配置建议**：
- `0.8` (默认): 平衡的物体/桌面比例，适合大多数场景
- `0.9`: 更关注物体细节，减少桌面点
- `0.7`: 增加环境信息，更多桌面点
- `1.0`: 纯物体点，无桌面（不推荐，失去环境上下文）

**注意**：实际采样点分布会受pytorch3d的sample_points_from_meshes影响，该方法按面积加权均匀采样。

## 采样策略详解

### 抓取采样策略

当 `use_exhaustive_sampling=False` 时，每个物体生成一个样本，从所有可用抓取中采样 `num_grasps` 个：

1. **random**: 随机选择
2. **first_n**: 选择前N个
3. **farthest_point**: 使用FPS算法，基于抓取位置的空间分布
4. **nearest_point**: 选择距离随机中心点最近的N个

### 穷尽采样策略

当 `use_exhaustive_sampling=True` 时，将每个物体的抓取分成多个chunks，实现100%数据利用：

1. **sequential**: 顺序分组 `[0,1,2,3], [4,5,6,7], ...`
2. **random**: 随机打乱后分组
3. **interleaved**: 交错分组 `[0,4,8,12], [1,5,9,13], ...`
4. **chunk_farthest_point**: 使用迭代FPS为每个chunk选择空间分散的抓取
5. **chunk_nearest_point**: 使用迭代NPS为每个chunk选择空间聚集的抓取

## 与SceneLeapPlusDataset的对比

| 特性 | SceneLeapPlusDataset | ObjectCentricGraspDataset |
|------|---------------------|--------------------------|
| 数据来源 | 场景数据（depth/rgb/mask） | 仅物体mesh |
| 点云生成 | 深度图反投影 | mesh表面采样 |
| 坐标系 | 支持多种（CF/OMF/归一化） | 固定OMF |
| 碰撞过滤 | 使用collision_free_grasp_info | 不过滤，使用所有成功抓取 |
| scene_pc维度 | (N, 6) xyz+rgb | (N, 3) xyz only |
| 依赖文件 | scene目录 + collision info | 仅.npy文件 |
| 适用场景 | 场景级训练 | 物体级训练 |

## 数据处理Pipeline

```
1. 扫描succ_grasp_dir/*.npy文件
   ↓
2. 加载grasp_qpos (GW坐标系) + obj_pose
   ↓
3. GW → OMF坐标转换
   ↓
4. LEAP格式反转换
   ↓
5. 应用max_grasps_per_object限制（可选）
   ↓
6. 构建数据索引（标准或穷尽）
   ↓
7. __getitem__时:
   - 加载mesh
   - 表面采样点云
   - 采样固定数量抓取
   - 重排序为[P|Joints|Q]
   - 创建SE(3)矩阵
```

## 测试

运行测试脚本验证数据集：

```bash
cd /home/xiantuo/source/grasp/SceneLeapUltra
source ~/.bashrc && conda activate DexGrasp
python tests/test_objectcentric_dataset.py
```

测试包括：
1. 基本数据加载
2. DataLoader批处理
3. 穷尽采样模式
4. 不同采样策略
5. OMF坐标系验证

## 注意事项

1. **点云维度**：scene_pc为(N, 3)，不包含RGB信息
2. **坐标系一致性**：所有数据（点云、抓取、顶点）均在OMF坐标系
3. **无场景信息**：scene_id使用obj_code填充，depth_view_index为0
4. **空字段**：object_mask为空tensor，negative_prompts为空列表
5. **mesh缩放**：确保mesh_scale与训练数据一致（通常为0.1）

## 常见问题

### Q: 为什么scene_pc没有RGB？
A: ObjectCentric模式专注于物体几何，RGB信息来自场景纹理，对纯几何学习不是必需的。如果需要RGB，可以修改代码或使用SceneLeapPlus。

### Q: 如何确保数据在正确的坐标系？
A: 数据集内部自动处理GW→OMF转换和LEAP反转换，输出保证在OMF坐标系。可以运行测试验证。

### Q: 穷尽采样有什么好处？
A: 穷尽采样确保每个抓取都被使用，提高数据利用率，特别适合数据量较少的情况。

### Q: 采样策略如何选择？
A: 
- 训练初期：使用`random`增加多样性
- 训练后期：使用`farthest_point`提高空间覆盖
- 快速验证：使用`first_n`确保可重复性

### Q: 桌面平面是如何添加的？
A: 在居中归一化后，在z=0处添加一个0.12x0.12米的正方形平面，由4个顶点和2个三角形面片组成。桌面点会在采样时与物体表面点混合，模拟真实的物体放置场景。

### Q: 为什么只在xy平面居中，保留z坐标？
A: 这样做符合物理约束：物体放在桌上时，xy位置可以变化，但z高度（物体底部到桌面的距离）是有物理意义的。保留z信息有助于模型学习合理的抓取高度。

## 更新日志

### v1.1 (2025-10-21)
- 修改坐标处理逻辑：使用GW原始坐标系而非OMF
- 添加xy平面居中归一化（z保持不变）
- 添加桌面平面（0.12x0.12米，z=0）模拟真实环境
- 提高物理真实性和训练稳定性

### v1.0 (2025-10-21)
- 初始版本
- 支持标准和穷尽采样模式
- 支持多种抓取采样策略
- 完全兼容SceneLeapPlus的训练代码

