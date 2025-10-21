# SceneLeapPlusDataset 数据处理 Pipeline 完整分析

## 概览

`SceneLeapPlusDataset` 继承自 `_BaseLeapProDataset`，专门设计用于并行多抓取分布学习。每个样本返回固定数量（`num_grasps`）的抓取姿态，适用于需要批量处理多个抓取的场景。

**核心返回数据**：
- `scene_pc`: 场景点云 (N_points, 6) - xyz + rgb
- `obj_verts`: 目标物体顶点 (N_verts, 3)
- `obj_faces`: 目标物体面片 (N_faces, 3)
- `hand_model_pose`: 手部姿态 (num_grasps, 23)
- `se3`: SE(3)变换矩阵 (num_grasps, 4, 4)

---

## 一、初始化阶段（数据加载与预处理）

### 1.1 父类初始化 `_BaseLeapProDataset.__init__()`

#### 1.1.1 场景目录扫描
```python
# 位置: sceneleappro_dataset.py: 76-77
self.scene_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                   if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('scene')]
```
**输入**: `root_dir` 路径  
**输出**: `self.scene_dirs` - 所有以 "scene" 开头的子目录列表

#### 1.1.2 场景元数据加载 `load_scene_metadata()`
```python
# 位置: sceneleappro_dataset.py: 80
# 实现: data_processing_utils.py: 163-201
self.instance_maps, self.scene_gt, self.collision_free_grasp_info = load_scene_metadata(self.scene_dirs)
```

**加载三个关键JSON文件**：

1. **instance_attribute_maps.json** → `self.instance_maps`
   - 路径: `{scene_dir}/instance_attribute_maps.json`
   - 内容: 每个视图的实例分割属性映射
   - 格式: `{view_idx: [{"category_id": int, ...}, ...]}`

2. **scene_gt.json** → `self.scene_gt`
   - 路径: `{scene_dir}/train_pbr/000000/scene_gt.json`
   - 内容: 场景ground truth，包含物体6D姿态
   - 格式: `{view_idx: [{"obj_id": int, "cam_R_m2c": [...], "cam_t_m2c": [...]}, ...]}`
   - 关键字段:
     - `cam_R_m2c`: 物体模型坐标系到相机坐标系的旋转矩阵 (9个元素，reshape成3x3)
     - `cam_t_m2c`: 平移向量 (3个元素，单位：毫米，需要转换为米)

3. **collision_free_grasp_indices.json** → `self.collision_free_grasp_info`
   - 路径: `{scene_dir}/collision_free_grasp_indices.json`
   - 内容: 每个物体的无碰撞抓取索引列表
   - 格式: `[{"object_name": str, "uid": str, "object_index": int, "collision_free_indices": [int, ...]}, ...]`

#### 1.1.3 手部姿态数据加载与处理 `load_and_process_hand_pose_data()`
```python
# 位置: sceneleappro_dataset.py: 82-85
# 实现: data_processing_utils.py: 111-160
self.hand_pose_data = load_and_process_hand_pose_data(
    self.scene_dirs, self.collision_free_grasp_info, succ_grasp_dir
)
```

**处理流程**：

**步骤1: 加载原始抓取数据**
```python
# 文件路径: {succ_grasp_dir}/{object_name}_{uid}.npy
grasp_data_dict = np.load(hand_pose_path, allow_pickle=True).item()
```
- **字典结构**:
  - `grasp_qpos`: (N, 23) numpy数组 - 在抓取世界坐标系(grasp world frame)中的手部姿态
    - [0:3]: 位置 P_gw (x, y, z)
    - [3:7]: 四元数 Q_gw (w, x, y, z)
    - [7:23]: 关节角度 (16个DOF)
  - `obj_pose`: (7,) numpy数组 - 物体在抓取世界坐标系中的7D姿态
    - [0:3]: 物体位置 t_obj_gw
    - [3:7]: 物体四元数 q_obj_gw (w, x, y, z)

**步骤2: 坐标系转换 - 抓取世界坐标系 → 物体中心坐标系**
```python
# 实现: transform_utils.py: 18-57
qpos_object_centric_tensor = transform_hand_poses_to_object_centric_frame(
    qpos_gw_tensor,      # (N, 23) 在grasp world中的手部姿态
    obj_pose_tensor      # (7,) 物体在grasp world中的姿态
)
```

**数学原理**:
```
给定:
- 手部姿态在GW中: P_h_gw, Q_h_gw
- 物体姿态在GW中: t_obj_gw, q_obj_gw

求: 手部姿态在物体坐标系(OMF)中: P_h_omf, Q_h_omf

计算:
1. 物体姿态求逆:
   q_obj_gw_inv = quaternion_invert(q_obj_gw)
   R_obj_gw_inv = quaternion_to_matrix(q_obj_gw_inv)

2. 位置转换:
   P_h_omf = R_obj_gw_inv @ (P_h_gw - t_obj_gw)

3. 旋转转换:
   Q_h_omf = q_obj_gw_inv * Q_h_gw  (四元数乘法)

输出: [P_h_omf, Q_h_omf, Joints] (N, 23)
```

**步骤3: LEAP格式反转换 `revert_leap_qpos_static()`**
```python
# 实现: transform_utils.py: 60-118
reverted_qpos = revert_leap_qpos_static(qpos_object_centric_tensor)
```

**LEAP格式转换的数学原理**:
```
LEAP格式使用特定的偏移和旋转变换来调整手部姿态表示。

给定LEAP格式姿态: [P_leap, Q_leap_wxyz, Joints]

反转换过程:
1. 四元数到旋转矩阵:
   R_leap = quaternion_to_matrix(Q_leap_wxyz)

2. 应用Delta旋转:
   delta_rot_quat = [0, 1, 0, 0]  # 绕Y轴旋转180度
   DeltaR = quaternion_to_matrix(delta_rot_quat)

3. 应用局部偏移:
   T_offset_local = [0, 0, 0.1]  # Z轴偏移0.1米
   Offset_world = R_leap @ T_offset_local

4. 计算原始位置和旋转:
   P_orig = P_leap + Offset_world
   R_orig = R_leap @ DeltaR
   Q_orig = matrix_to_quaternion(R_orig)

输出: [P_orig, Q_orig, Joints] (N, 23)
```

**最终存储格式**:
```python
# self.hand_pose_data 结构:
{
    "scene_id": {
        "object_name_uid": torch.Tensor (N_grasps, 23),  # 在物体模型坐标系(OMF)中的手部姿态
        ...
    },
    ...
}
```

**姿态23维度详解**:
- [0:3]: 手腕位置 (x, y, z) 在物体模型坐标系中
- [3:7]: 手腕旋转四元数 (w, x, y, z)
- [7:23]: 16个关节角度 (手指关节配置)

#### 1.1.4 相机参数加载
```python
# 位置: sceneleappro_dataset.py: 88, 90-103
self.camera = self._load_camera_config(root_dir)
```

**加载camera.json**:
```json
{
    "width": 640,
    "height": 480,
    "fx": 572.4114,
    "fy": 573.5704,
    "cx": 325.2611,
    "cy": 242.0489,
    "depth_scale": 0.1  // 深度图scale，会乘以1000转换为mm
}
```

**相机内参矩阵**:
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

### 1.2 SceneLeapPlus特有的初始化

#### 1.2.1 过滤无碰撞抓取 `_filter_collision_free_poses()`
```python
# 位置: sceneleapplus_dataset.py: 111
self._filter_collision_free_poses()
```

**处理流程**:
1. 遍历 `self.hand_pose_data` 中的每个场景和物体
2. 从 `self.collision_free_grasp_info` 获取该物体的无碰撞抓取索引
3. 使用索引过滤 `hand_pose_data`，只保留无碰撞的姿态
4. 如果需要，应用 `max_grasps_per_object` 限制
5. 如果设置了采样策略，使用 `_sample_indices_from_available()` 进行智能采样

**采样策略**:
- `random`: 随机采样
- `first_n`: 取前N个
- `repeat`: 重复采样
- `farthest_point`: 基于3D位置的最远点采样(FPS)
- `nearest_point`: 基于3D位置的最近点采样(NPS)

#### 1.2.2 构建数据索引
```python
# 位置: sceneleapplus_dataset.py: 113-118
if self.use_exhaustive_sampling:
    self.data = self._build_exhaustive_data_index()
else:
    self.data = self._build_data_index()
```

**标准索引构建 `_build_data_index()`**:
```python
# 每个数据项结构:
{
    'scene_id': str,                    # 场景ID
    'object_code': str,                 # "{object_name}_{uid}"
    'category_id_for_masking': int,     # 用于mask提取的物体ID
    'depth_view_index': int             # 深度视图索引
}
```

**穷尽采样索引 `_build_exhaustive_data_index()`**:
- 将每个物体的抓取分成多个chunks，每个chunk包含 `num_grasps` 个抓取
- 生成策略: `sequential`, `random`, `interleaved`, `chunk_farthest_point`, `chunk_nearest_point`
- 额外字段:
  - `is_exhaustive`: bool
  - `chunk_idx`: int
  - `total_chunks`: int
  - `grasp_indices`: List[int] - 预先指定的抓取索引

---

## 二、数据获取阶段 `__getitem__(idx)`

### 2.1 主流程概览
```python
# 位置: sceneleapplus_dataset.py: 420-445
def __getitem__(self, idx):
    item_data = self.data[idx]
    
    # 步骤1: 加载场景数据（深度图、RGB图、实例mask）
    scene_data = self._load_scene_data(item_data)
    
    # 步骤2: 处理点云和mask
    processed_data = self._process_point_cloud_and_masks(scene_data, item_data)
    
    # 步骤3: 加载抓取和物体数据
    grasp_object_data = self._load_grasp_and_object_data_for_fixed_grasps(item_data)
    
    # 步骤4: 根据mode进行坐标转换
    transformed_data = self._transform_data_by_mode(...)
    
    # 步骤5: 打包最终数据
    return self._package_data_for_fixed_grasps(...)
```

---

### 2.2 步骤1: 加载场景数据 `_load_scene_data()`

#### 2.2.1 构建文件路径
```python
# 位置: io_utils.py: 133-148
scene_dir = os.path.join(self.root_dir, scene_id)
depth_view_index = item_data['depth_view_index']

depth_filename = f'{depth_view_index:06d}.png'  # 例如: 000000.png
depth_path = os.path.join(scene_dir, 'train_pbr/000000/depth', depth_filename)
rgb_path = os.path.join(scene_dir, 'train_pbr/000000/rgb', depth_filename)
instance_mask_path = os.path.join(scene_dir, 'InstanceMask', depth_filename)
```

**文件路径示例**:
```
{root_dir}/scene_0001/train_pbr/000000/depth/000000.png
{root_dir}/scene_0001/train_pbr/000000/rgb/000000.png
{root_dir}/scene_0001/InstanceMask/000000.png
```

#### 2.2.2 加载图像
```python
# 位置: io_utils.py: 17-73
# 所有加载函数都使用 @lru_cache 装饰器进行缓存

# 1. 深度图
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
# 格式: uint16, shape (H, W), 值需要除以 depth_scale 转换为米

# 2. RGB图
rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
# 格式: uint8, shape (H, W, 3), 值范围 [0, 255]

# 3. 实例mask
instance_mask_image = cv2.imread(instance_mask_path, cv2.IMREAD_UNCHANGED)
# 格式: uint8/uint16, shape (H, W), 每个像素值代表物体ID
```

**返回数据**:
```python
{
    'scene_id': str,
    'scene_dir': str,
    'depth_view_index': int,
    'depth_image': np.ndarray (H, W),          # uint16
    'rgb_image': np.ndarray (H, W, 3),         # uint8, RGB顺序
    'instance_mask_image': np.ndarray (H, W)   # uint8/uint16
}
```

---

### 2.3 步骤2: 处理点云和mask `_process_point_cloud_and_masks()`

#### 2.3.1 深度图 → 3D点云 `create_point_cloud_from_depth_image()`
```python
# 位置: common_utils.py: 270-281
scene_pc_raw_camera_frame = create_point_cloud_from_depth_image(
    depth_image,    # (H, W)
    self.camera,
    organized=False
)
```

**生成3D点云的数学原理**:
```python
# 1. 生成像素坐标网格
xmap, ymap = np.meshgrid(range(width), range(height))
# xmap: (H, W) - 每个像素的x坐标 [0, width-1]
# ymap: (H, W) - 每个像素的y坐标 [0, height-1]

# 2. 深度值转换为米
points_z = depth.astype(np.float32) / camera.depth_scale  # (H, W)

# 3. 相机针孔模型反投影
# 给定像素坐标 (u, v) 和深度 z，计算3D点 (x, y, z):
points_x = (xmap - camera.cx) * points_z / camera.fx
points_y = (ymap - camera.cy) * points_z / camera.fy

# 相机坐标系定义:
# - X轴: 指向右
# - Y轴: 指向下
# - Z轴: 指向前（深度方向）

# 4. 组合为点云
cloud = np.stack([points_x, points_y, points_z], axis=-1)  # (H, W, 3)

# 5. 如果 organized=False，展平为 (H*W, 3)
if not organized:
    cloud = cloud.reshape([-1, 3])
```

**输出**: `scene_pc_raw_camera_frame` - shape (N, 3), 在相机坐标系中的3D点云

#### 2.3.2 为点云添加RGB颜色 `add_rgb_to_pointcloud()`
```python
# 位置: pointcloud_utils.py: 18-68
scene_pc_with_rgb = add_rgb_to_pointcloud(
    scene_pc_raw_camera_frame,  # (N, 3)
    rgb_image,                   # (H, W, 3)
    self.camera
)
```

**处理流程**:
```python
# 1. 过滤有效深度的点
valid_depth_mask = points_z > 1e-6

# 2. 将3D点投影回图像平面
# 使用相机内参矩阵投影:
u = (points_x * camera.fx / points_z + camera.cx).astype(int)
v = (points_y * camera.fy / points_z + camera.cy).astype(int)

# 3. 检查像素边界
pixel_valid_mask = (u >= 0) & (u < camera.width) & 
                   (v >= 0) & (v < camera.height)

# 4. 从RGB图像采样颜色
colors = rgb_image[valid_v, valid_u].astype(np.float32) / 255.0  # 归一化到[0,1]

# 5. 合并xyz和rgb
pointcloud_with_rgb = np.hstack([points_3d, colors])  # (N, 6)
```

**输出**: `scene_pc_with_rgb` - shape (N, 6), [x, y, z, r, g, b]

#### 2.3.3 提取目标物体的2D mask `extract_object_mask()`
```python
# 位置: sceneleappro_dataset.py: 145-147
view_specific_instance_attributes = self.instance_maps.get(scene_id, {}).get(str(depth_view_index), [])
object_mask_2d = extract_object_mask(
    instance_mask_image,                  # (H, W)
    category_id_for_masking,              # target object ID
    view_specific_instance_attributes     # [{category_id: int, ...}, ...]
)
```

**处理流程**:
```python
# 1. 从instance_attribute_maps查找target物体的实例ID
target_instance_id = None
for attr in view_specific_instance_attributes:
    if attr.get('category_id') == category_id_for_masking:
        target_instance_id = attr.get('instance_id')
        break

# 2. 在instance_mask_image中提取该实例
object_mask_2d = (instance_mask_image == target_instance_id)  # (H, W), bool
```

**输出**: `object_mask_2d` - shape (H, W), bool类型的2D mask

#### 2.3.4 将2D mask映射到3D点云 `map_2d_mask_to_3d_pointcloud()`
```python
# 位置: pointcloud_utils.py: 71-122
mask_2d_reshaped = object_mask_2d.reshape(instance_mask_image.shape)  # (H, W)

object_mask_np = map_2d_mask_to_3d_pointcloud(
    scene_pc_with_rgb[:, :3],  # (N, 3) - xyz部分
    mask_2d_reshaped,          # (H, W)
    self.camera
)
```

**映射原理**:
```python
# 对于点云中的每个3D点，将其投影回图像平面，
# 检查对应像素在2D mask中是否为True

for each point (x, y, z) in point_cloud:
    if z > 1e-6:  # 有效深度
        # 投影到像素坐标
        u = int(x * fx / z + cx)
        v = int(y * fy / z + cy)
        
        # 检查边界
        if 0 <= u < width and 0 <= v < height:
            # 检查2D mask
            object_mask_3d[point_idx] = mask_2d[v, u]
```

**输出**: `object_mask_np` - shape (N,), bool类型的3D点云mask

#### 2.3.5 点云裁剪（可选）`crop_point_cloud_to_objects_with_mask()`
```python
# 位置: sceneleappro_dataset.py: 158-161
if self.enable_cropping:
    scene_pc_with_rgb, object_mask_np = crop_point_cloud_to_objects_with_mask(
        scene_pc_with_rgb,      # (N, 6)
        object_mask_np,         # (N,)
        instance_mask_image,    # (H, W)
        self.camera
    )
```

**裁剪流程**:
```python
# 1. 在instance mask中找到所有物体（value > 9）
all_objects_mask = instance_mask_image > 9

# 2. 计算所有物体的边界框（带padding）
min_y, max_y = object_pixels[0].min(), object_pixels[0].max()
min_x, max_x = object_pixels[1].min(), object_pixels[1].max()

crop_padding = 25  # 像素
min_x = max(0, min_x - crop_padding)
max_x = min(width - 1, max_x + crop_padding)
min_y = max(0, min_y - crop_padding)
max_y = min(height - 1, max_y + crop_padding)

# 3. 将点云投影回图像，保留落在边界框内的点
for each point in point_cloud:
    u, v = project_to_image(point, camera)
    if min_x <= u <= max_x and min_y <= v <= max_y:
        keep_point = True

# 4. 同步裁剪object_mask
cropped_object_mask = object_mask[crop_mask]
```

**输出**:
```python
{
    'scene_pc_with_rgb': np.ndarray (M, 6),  # M < N，裁剪后的点云
    'object_mask_np': np.ndarray (M,)        # 对应的mask
}
```

---

### 2.4 步骤3: 加载抓取和物体数据 `_load_grasp_and_object_data_for_fixed_grasps()`

#### 2.4.1 获取固定数量的手部姿态 `_get_fixed_number_hand_poses()`
```python
# 位置: sceneleapplus_dataset.py: 404-418
hand_pose_tensor = self._get_fixed_number_hand_poses(
    scene_id,
    object_code,
    grasp_indices  # 如果使用穷尽采样，则预先指定索引
)
```

**处理流程**:
```python
# 1. 获取该物体的所有无碰撞抓取
all_poses_tensor = self.hand_pose_data[scene_id][object_code]  # (N_available, 23)

# 2. 如果指定了grasp_indices（穷尽采样模式）
if grasp_indices:
    valid_indices = [i for i in grasp_indices if 0 <= i < N_available]
    if len(valid_indices) >= num_grasps:
        return all_poses_tensor[valid_indices[:num_grasps]]  # (num_grasps, 23)

# 3. 否则使用采样策略
return self._sample_grasps_from_available(all_poses_tensor)
```

**采样策略实现**:
```python
def _sample_grasps_from_available(available_grasps):  # (N_available, 23)
    if N_available >= num_grasps:
        # 情况1: 足够的抓取
        if strategy == "random":
            indices = torch.randperm(N_available)[:num_grasps]
        elif strategy == "first_n":
            indices = torch.arange(num_grasps)
        elif strategy == "farthest_point":
            # 最远点采样（FPS）基于位置 [:, :3]
            indices = _farthest_point_sampling(available_grasps[:, :3], num_grasps)
        elif strategy == "nearest_point":
            # 最近点采样（NPS）基于位置
            indices = _nearest_point_sampling(available_grasps[:, :3], num_grasps)
        
        return available_grasps[indices]  # (num_grasps, 23)
    
    else:
        # 情况2: 抓取不足，需要填充
        if strategy == "repeat":
            # 循环索引
            indices = torch.arange(num_grasps) % N_available
        else:
            # 全部使用 + 随机重复
            all_indices = torch.arange(N_available)
            additional_indices = torch.randint(0, N_available, (num_grasps - N_available,))
            indices = torch.cat([all_indices, additional_indices])
        
        return available_grasps[indices]  # (num_grasps, 23)
```

**最远点采样(FPS)算法**:
```python
def _farthest_point_sampling(points, num_samples):  # points: (N, 3)
    """
    贪心算法：每次选择距离已选点集最远的点
    """
    sampled_indices = torch.zeros(num_samples, dtype=torch.long)
    
    # 随机选择第一个点
    sampled_indices[0] = torch.randint(0, N, (1,))
    
    # 初始化距离数组
    distances = torch.full((N,), float('inf'))
    
    for i in range(1, num_samples):
        # 计算所有点到最后一个采样点的距离
        last_point = points[sampled_indices[i-1]]
        current_distances = torch.norm(points - last_point, dim=1)
        
        # 更新最小距离
        distances = torch.min(distances, current_distances)
        
        # 选择距离最大的点
        sampled_indices[i] = torch.argmax(distances)
        distances[sampled_indices[i]] = 0  # 标记已选择
    
    return sampled_indices  # (num_samples,)
```

**输出**: `hand_pose_tensor` - shape (num_grasps, 23), 在物体模型坐标系(OMF)中

#### 2.4.2 获取相机变换矩阵 `get_camera_transform()`
```python
# 位置: transform_utils.py: 337-360
scene_gt_for_view = self.scene_gt[scene_id][str(depth_view_index)]
cam_R, cam_t = get_camera_transform(scene_gt_for_view, category_id_for_masking)
```

**提取流程**:
```python
# 1. 在scene_gt中查找目标物体
for obj_gt in scene_gt_for_view:
    if obj_gt['obj_id'] == category_id_for_masking:
        cam_R_m2c_list = obj_gt['cam_R_m2c']  # 9个元素
        cam_t_m2c_list = obj_gt['cam_t_m2c']  # 3个元素
        break

# 2. 转换为numpy数组
cam_R_m2c = np.array(cam_R_m2c_list).reshape(3, 3)  # 旋转矩阵
cam_t_m2c = np.array(cam_t_m2c_list) / 1000.0       # mm → m
```

**变换矩阵含义**:
```
T_OMF_to_CF = [R_m2c | t_m2c]  (4x4 齐次矩阵)
              [  0   |   1  ]

其中:
- OMF (Object Model Frame): 物体模型坐标系（加载的mesh的原始坐标系）
- CF (Camera Frame): 相机坐标系

变换公式:
P_camera = R_m2c @ P_model + t_m2c
```

**输出**:
- `cam_R_model_to_camera`: np.ndarray (3, 3)
- `cam_t_model_to_camera`: np.ndarray (3,)

#### 2.4.3 加载物体网格 `load_object_mesh()`
```python
# 位置: io_utils.py: 76-96
obj_verts, obj_faces = load_object_mesh(
    self.obj_root_dir,
    object_code,
    self.mesh_scale
)
```

**加载流程**:
```python
# 1. 构建mesh文件路径
mesh_path = os.path.join(obj_root_dir, object_code, 'mesh', 'simplified.obj')
# 例如: {obj_root_dir}/bottle_0a12b34c/mesh/simplified.obj

# 2. 使用pytorch3d加载OBJ文件
from pytorch3d.io import load_obj
verts, faces, _ = load_obj(mesh_path, load_textures=False)
# verts: torch.Tensor (V, 3) - 顶点坐标
# faces: Faces对象，其中 faces.verts_idx 是 (F, 3) 的面片索引

# 3. 应用缩放
verts = verts * mesh_scale  # 通常 mesh_scale = 0.1

# 4. 返回
return verts, faces.verts_idx
```

**输出**:
- `obj_verts`: torch.Tensor (V, 3), float32, 在物体模型坐标系(OMF)中
- `obj_faces`: torch.Tensor (F, 3), int64, 面片顶点索引

**返回数据汇总**:
```python
{
    'hand_pose_tensor': torch.Tensor (num_grasps, 23),  # OMF坐标系
    'scene_gt_for_view': List[Dict],
    'cam_R_model_to_camera': np.ndarray (3, 3),
    'cam_t_model_to_camera': np.ndarray (3,),
    'obj_verts': torch.Tensor (V, 3),                   # OMF坐标系
    'obj_faces': torch.Tensor (F, 3)
}
```

---

### 2.5 步骤4: 坐标转换 `_transform_data_by_mode()`

这是数据处理的核心步骤，根据不同的 `mode` 选择不同的坐标转换策略。

#### 2.5.1 转换策略概览

```python
# 位置: coordinate_transform_strategies.py: 503-525
strategy = create_transform_strategy(self.mode)
transformed_data = strategy.transform(transform_data)
```

**支持的5种模式**:

| Mode | 点云坐标系 | 抓取坐标系 | 物体顶点坐标系 | 是否归一化 |
|------|-----------|-----------|---------------|----------|
| `object_centric` | OMF | OMF | OMF | 否 |
| `camera_centric` | CF | CF | CF | 否 |
| `camera_centric_obj_mean_normalized` | CF | CF | CF | 是（物体中心） |
| `camera_centric_scene_mean_normalized` | CF | CF | CF | 是（场景中心） |

**输入数据（TransformationData）**:
```python
{
    'pc_cam_raw_xyz_rgb': np.ndarray (N, 6),        # 点云在CF中
    'grasps_omf': torch.Tensor (num_grasps, 23),    # 抓取在OMF中
    'obj_verts': torch.Tensor (V, 3),               # 顶点在OMF中
    'R_omf_to_cf_np': np.ndarray (3, 3),            # OMF→CF 旋转
    't_omf_to_cf_np': np.ndarray (3,),              # OMF→CF 平移
    'object_mask_np': np.ndarray (N,)               # 点云物体mask
}
```

#### 2.5.2 模式1: Object-Centric（物体中心坐标系）

```python
# 实现: coordinate_transform_strategies.py: 225-264
class ObjectCentricStrategy(CoordinateTransformStrategy):
    def transform(self, data):
        # 1. 点云: CF → OMF
        pc_omf_xyz = transform_point_cloud(
            data.pc_cam_raw_xyz_rgb[:, :3],  # xyz部分
            data.R_omf_to_cf_np,
            data.t_omf_to_cf_np
        )
        final_pc = np.hstack([pc_omf_xyz, data.pc_cam_raw_xyz_rgb[:, 3:6]])  # 添加RGB
        
        # 2. 抓取: 保持OMF
        final_grasps = data.grasps_omf
        
        # 3. 物体顶点: 保持OMF
        final_obj_verts = data.obj_verts.clone()
        
        return {'pc': final_pc, 'grasps': final_grasps, 'obj_verts': final_obj_verts}
```

**点云变换数学原理**:
```python
def transform_point_cloud(points_cf, R_m2c, t_m2c):
    """
    将点云从相机坐标系(CF)转换到物体模型坐标系(OMF)
    
    给定:
    - P_cf: 点在相机坐标系中 (N, 3)
    - T_omf_to_cf = [R_m2c | t_m2c]: OMF到CF的变换
    
    求: P_omf 点在物体坐标系中
    
    逆变换:
    T_cf_to_omf = inv(T_omf_to_cf)
    R_cm = R_m2c.T
    t_cm = -R_m2c.T @ t_m2c
    
    P_omf = R_cm @ P_cf + t_cm
         = P_cf @ R_cm.T + t_cm  (向量形式)
    """
    R_cm = R_m2c.T
    t_cm = -np.dot(R_m2c.T, t_m2c.reshape(3,1)).flatten()
    
    points_omf = np.dot(points_cf, R_cm.T) + t_cm.reshape(1, 3)
    return points_omf  # (N, 3)
```

#### 2.5.3 模式2: Camera-Centric（相机中心坐标系）

```python
# 实现: coordinate_transform_strategies.py: 267-316
class CameraCentricStrategy(CoordinateTransformStrategy):
    def transform(self, data):
        # 1. 点云: 保持CF
        final_pc = data.pc_cam_raw_xyz_rgb
        
        # 2. 抓取: OMF → CF
        final_grasps = transform_hand_poses_omf_to_cf(
            data.grasps_omf,
            data.R_omf_to_cf_np,
            data.t_omf_to_cf_np
        )
        
        # 3. 物体顶点: OMF → CF
        R_m2c_tensor = torch.from_numpy(data.R_omf_to_cf_np.astype(np.float32)).to(data.obj_verts.device)
        t_m2c_tensor = torch.from_numpy(data.t_omf_to_cf_np.astype(np.float32)).to(data.obj_verts.device)
        final_obj_verts = torch.matmul(data.obj_verts, R_m2c_tensor.T) + t_m2c_tensor.unsqueeze(0)
        
        return {'pc': final_pc, 'grasps': final_grasps, 'obj_verts': final_obj_verts}
```

**手部姿态变换**:
```python
def transform_hand_poses_omf_to_cf(hand_poses_omf, R_m2c, t_m2c):
    """
    将手部姿态从OMF转换到CF
    
    输入: hand_poses_omf (N, 23) = [P_omf, Q_omf_wxyz, Joints]
    输出: hand_poses_cf (N, 23) = [P_cf, Q_cf_wxyz, Joints]
    """
    # 提取组件
    P_omf = hand_poses_omf[:, :3]        # (N, 3)
    Q_omf = hand_poses_omf[:, 3:7]       # (N, 4) wxyz格式
    Joints = hand_poses_omf[:, 7:]       # (N, 16)
    
    # 转换为张量
    R_m2c_tensor = torch.from_numpy(R_m2c.astype(np.float32))  # (3, 3)
    t_m2c_tensor = torch.from_numpy(t_m2c.astype(np.float32))  # (3,)
    
    # 1. 位置变换
    # P_cf = P_omf @ R_m2c.T + t_m2c
    P_cf = torch.matmul(P_omf, R_m2c_tensor.T) + t_m2c_tensor.unsqueeze(0)  # (N, 3)
    
    # 2. 旋转变换（四元数）
    # Q_cf = Q_R_m2c * Q_omf  (四元数左乘)
    Q_R_m2c = matrix_to_quaternion(R_m2c_tensor)  # (4,) wxyz
    Q_R_m2c_batched = Q_R_m2c.unsqueeze(0).expand(N, -1)  # (N, 4)
    Q_cf = quaternion_multiply(Q_R_m2c_batched, Q_omf)  # (N, 4)
    
    # 3. 关节角度不变
    # Joints保持不变
    
    # 4. 重新组合
    hand_poses_cf = torch.cat([P_cf, Q_cf, Joints], dim=-1)  # (N, 23)
    
    return hand_poses_cf
```

**物体顶点变换**:
```python
# V_cf = V_omf @ R_m2c.T + t_m2c
obj_verts_cf = torch.matmul(obj_verts_omf, R_m2c.T) + t_m2c.unsqueeze(0)
# obj_verts_cf: (V, 3)
```

#### 2.5.4 模式3: Camera-Centric + Object Mean Normalized

```python
# 实现: coordinate_transform_strategies.py: 319-410
class CameraCentricObjNormalizedStrategy(CoordinateTransformStrategy):
    def transform(self, data):
        # 1. 计算物体点云的中心（在CF中）
        obj_pts_cf = data.pc_cam_raw_xyz_rgb[data.object_mask_np, :3]  # (N_obj, 3)
        obj_mean_cf = np.mean(obj_pts_cf, axis=0) if obj_pts_cf.shape[0] > 0 else np.zeros(3)  # (3,)
        
        # 2. 点云归一化: CF → CF_normalized
        final_pc_xyz = data.pc_cam_raw_xyz_rgb[:, :3] - obj_mean_cf.reshape(1, 3)
        final_pc = np.hstack([final_pc_xyz, data.pc_cam_raw_xyz_rgb[:, 3:6]])
        
        # 3. 抓取: OMF → CF → CF_normalized
        grasps_cf = transform_hand_poses_omf_to_cf(
            data.grasps_omf, data.R_omf_to_cf_np, data.t_omf_to_cf_np
        )
        final_grasps = self._normalize_grasps_by_mean(grasps_cf, obj_mean_cf)
        
        # 4. 物体顶点: OMF → CF → CF_normalized
        obj_verts_cf = torch.matmul(data.obj_verts, R_m2c_tensor.T) + t_m2c_tensor.unsqueeze(0)
        obj_mean_tensor = torch.from_numpy(obj_mean_cf.astype(np.float32)).to(obj_verts_cf.device)
        final_obj_verts = obj_verts_cf - obj_mean_tensor.unsqueeze(0)
        
        return {'pc': final_pc, 'grasps': final_grasps, 'obj_verts': final_obj_verts}
```

**抓取姿态归一化**:
```python
def _normalize_grasps_by_mean(grasps_cf, mean_vector):
    """
    只对位置部分减去mean，旋转和关节保持不变
    
    输入: grasps_cf (N, 23) = [P_cf, Q_cf, Joints]
    输出: grasps_normalized (N, 23) = [P_cf - mean, Q_cf, Joints]
    """
    P_cf = grasps_cf[:, :3]              # (N, 3)
    Q_cf = grasps_cf[:, 3:7]             # (N, 4)
    Joints = grasps_cf[:, 7:]            # (N, 16)
    
    mean_tensor = torch.from_numpy(mean_vector.astype(np.float32)).to(P_cf.device)
    P_normalized = P_cf - mean_tensor.unsqueeze(0)  # (N, 3)
    
    grasps_normalized = torch.cat([P_normalized, Q_cf, Joints], dim=-1)
    return grasps_normalized  # (N, 23)
```

#### 2.5.5 模式4: Camera-Centric + Scene Mean Normalized

```python
# 实现: coordinate_transform_strategies.py: 412-500
class CameraCentricSceneNormalizedStrategy(CoordinateTransformStrategy):
    def transform(self, data):
        # 1. 计算整个场景点云的中心（在CF中）
        scene_mean_cf = np.mean(data.pc_cam_raw_xyz_rgb[:, :3], axis=0) if data.pc_cam_raw_xyz_rgb.shape[0] > 0 else np.zeros(3)
        
        # 2. 点云归一化
        final_pc_xyz = data.pc_cam_raw_xyz_rgb[:, :3] - scene_mean_cf.reshape(1, 3)
        final_pc = np.hstack([final_pc_xyz, data.pc_cam_raw_xyz_rgb[:, 3:6]])
        
        # 3. 抓取归一化
        grasps_cf = transform_hand_poses_omf_to_cf(...)
        final_grasps = self._normalize_grasps_by_mean(grasps_cf, scene_mean_cf)
        
        # 4. 物体顶点归一化
        obj_verts_cf = ...
        final_obj_verts = obj_verts_cf - scene_mean_tensor.unsqueeze(0)
        
        return {'pc': final_pc, 'grasps': final_grasps, 'obj_verts': final_obj_verts}
```

**归一化策略对比**:

| 策略 | 归一化中心 | 适用场景 |
|------|-----------|---------|
| Object Mean | 目标物体点云的中心 | 物体尺度和位置归一化，关注物体本身 |
| Scene Mean | 整个场景点云的中心 | 场景级归一化，保留物体在场景中的相对位置 |

---

### 2.6 步骤5: 打包最终数据 `_package_data_for_fixed_grasps()`

#### 2.6.1 点云下采样 `downsample_point_cloud_with_mask()`

```python
# 位置: pointcloud_utils.py: 431-511
downsampled_pc_with_rgb_np, downsampled_object_mask_np = downsample_point_cloud_with_mask(
    final_pc_xyz_rgb_np,    # (N, 6)
    object_mask_np,         # (N,)
    max_points=self.max_points  # 默认10000
)
```

**下采样流程**:

```python
def downsample_point_cloud_with_mask(point_cloud, object_mask, max_points=10000):
    N = point_cloud.shape[0]
    
    # 1. 如果点数已经 <= max_points
    if N <= max_points:
        if N < max_points:
            # 填充零点
            padding_size = max_points - N
            pc_padding = np.zeros((padding_size, 6), dtype=point_cloud.dtype)
            mask_padding = np.zeros(padding_size, dtype=object_mask.dtype)
            return (
                np.vstack([point_cloud, pc_padding]),
                np.concatenate([object_mask, mask_padding])
            )
        return point_cloud, object_mask
    
    # 2. 需要下采样: 使用分层采样保持物体/背景比例
    object_indices = np.where(object_mask)[0]      # 物体点索引
    background_indices = np.where(~object_mask)[0]  # 背景点索引
    
    # 计算采样目标
    object_ratio = len(object_indices) / N
    target_object_points = int(max_points * object_ratio)
    target_background_points = max_points - target_object_points
    
    # 3. 采样物体点
    if len(object_indices) > target_object_points:
        selected_object = np.random.choice(object_indices, target_object_points, replace=False)
    else:
        selected_object = object_indices
    
    # 4. 采样背景点
    if len(background_indices) > target_background_points:
        selected_background = np.random.choice(background_indices, target_background_points, replace=False)
    else:
        selected_background = background_indices
    
    # 5. 合并并填充到max_points
    selected_indices = np.concatenate([selected_object, selected_background])
    if len(selected_indices) < max_points:
        # 从剩余点中随机采样补足
        unused_indices = np.setdiff1d(np.arange(N), selected_indices)
        additional = np.random.choice(unused_indices, max_points - len(selected_indices), replace=False)
        selected_indices = np.concatenate([selected_indices, additional])
    
    # 6. 打乱顺序
    np.random.shuffle(selected_indices)
    selected_indices = selected_indices[:max_points]
    
    return point_cloud[selected_indices], object_mask[selected_indices]
```

**输出**: 
- `downsampled_pc_with_rgb_np`: (10000, 6)
- `downsampled_object_mask_np`: (10000,)

#### 2.6.2 生成文本提示

```python
# 正面提示: 目标物体名称
object_name = extract_object_name_from_code(object_code)  # "bottle_0a12b34c" → "bottle"
positive_prompt = object_name.replace('_', ' ')

# 负面提示: 场景中其他物体（最多num_neg_prompts个）
negative_prompts = generate_negative_prompts(
    self.collision_free_grasp_info[scene_id],
    object_name,
    self.num_neg_prompts  # 默认4
)
```

**负面提示生成逻辑**:
```python
def generate_negative_prompts(scene_collision_info, current_obj_name, num_neg_prompts=4):
    # 1. 收集场景中所有其他物体的名称
    other_objects = []
    for obj_info in scene_collision_info:
        obj_name = obj_info.get('object_name')
        if obj_name and obj_name != current_obj_name:
            other_objects.append(obj_name)
    
    # 2. 去重
    unique_objects = list(dict.fromkeys(other_objects))
    
    # 3. 生成num_neg_prompts个负面提示
    if len(unique_objects) == 0:
        return [""] * num_neg_prompts  # 没有其他物体，返回空字符串
    elif len(unique_objects) < num_neg_prompts:
        # 不足，用最后一个物体填充
        return unique_objects + [unique_objects[-1]] * (num_neg_prompts - len(unique_objects))
    else:
        # 足够，取前N个
        return unique_objects[:num_neg_prompts]
```

#### 2.6.3 处理手部姿态: 重排序和创建SE(3)矩阵

```python
# 位置: sceneleappro_dataset.py: 361-406
hand_model_pose_reordered, se3_matrices = self._process_batch_grasps(
    final_grasps,                           # (num_grasps, 23)
    downsampled_pc_with_rgb_tensor.device
)
```

**批量抓取处理流程**:

```python
def _process_batch_grasps(final_grasps, device):
    """
    输入: final_grasps (N, 23) = [P, Q_wxyz, Joints]
    输出:
    - hand_model_pose_reordered (N, 23) = [P, Joints, Q_wxyz]
    - se3_matrices (N, 4, 4)
    """
    N = final_grasps.shape[0]
    
    # 1. 提取组件
    P_all = final_grasps[:, :3]          # (N, 3) 位置
    Q_all_wxyz = final_grasps[:, 3:7]    # (N, 4) 四元数 [w,x,y,z]
    Joints_all = final_grasps[:, 7:]     # (N, 16) 关节角度
    
    # 2. 重排序为 [P, Joints, Q_wxyz] 格式
    # 这是模型期望的输入格式
    hand_model_pose_reordered = torch.cat([P_all, Joints_all, Q_all_wxyz], dim=1)  # (N, 23)
    
    # 3. 处理零四元数（修正为单位四元数）
    Q_corrected_wxyz = Q_all_wxyz.clone()
    norms = torch.norm(Q_corrected_wxyz, dim=1, keepdim=True)  # (N, 1)
    zero_q_mask = norms.squeeze(1) < 1e-6  # 找出模长接近0的四元数
    
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=final_grasps.dtype)  # 单位四元数
    Q_corrected_wxyz[zero_q_mask] = identity_q  # 替换为单位四元数
    
    # 4. 创建SE(3)矩阵
    # 四元数 → 旋转矩阵
    R_all = quaternion_to_matrix(Q_corrected_wxyz)  # (N, 3, 3)
    
    # 组装SE(3)矩阵
    se3_matrices = torch.zeros((N, 4, 4), device=device, dtype=final_grasps.dtype)
    se3_matrices[:, :3, :3] = R_all      # 旋转部分
    se3_matrices[:, :3, 3] = P_all       # 平移部分
    se3_matrices[:, 3, 3] = 1.0          # 齐次坐标
    
    return hand_model_pose_reordered, se3_matrices
```

**SE(3)矩阵格式**:
```
SE3[i] = [R_11  R_12  R_13  t_x]    (4x4)
         [R_21  R_22  R_23  t_y]
         [R_31  R_32  R_33  t_z]
         [ 0     0     0     1 ]

其中:
- R (3x3): 从四元数 Q_wxyz 转换的旋转矩阵
- t (3,): 位置向量 P
```

#### 2.6.4 最终返回字典

```python
# 位置: sceneleappro_dataset.py: 313-326
return {
    # 基本信息
    'obj_code': object_code,                             # str, e.g., "bottle_0a12b34c"
    'scene_id': scene_id,                                # str, e.g., "scene_0001"
    'category_id_from_object_index': category_id,        # int
    'depth_view_index': depth_view_index,                # int
    
    # 点云数据
    'scene_pc': downsampled_pc_with_rgb_tensor,         # torch.Tensor (10000, 6), float32
    'object_mask': downsampled_object_mask_tensor,      # torch.Tensor (10000,), bool
    
    # 抓取数据
    'hand_model_pose': hand_model_pose_reordered,       # torch.Tensor (num_grasps, 23), float32
    'se3': se3_matrices,                                # torch.Tensor (num_grasps, 4, 4), float32
    
    # 物体mesh
    'obj_verts': final_obj_verts,                       # torch.Tensor (V, 3), float32
    'obj_faces': obj_faces,                             # torch.Tensor (F, 3), int64
    
    # 文本提示
    'positive_prompt': positive_prompt,                  # str, e.g., "bottle"
    'negative_prompts': negative_prompts                 # List[str], length=num_neg_prompts
}
```

---

## 三、批量处理 `collate_fn()`

### 3.1 SceneLeapPlus的固定抓取批处理

```python
# 位置: sceneleapplus_dataset.py: 474-516
@staticmethod
def collate_fn(batch):
    """
    输入: batch - List[Dict]，每个Dict是__getitem__返回的数据
    输出: 批处理后的Dict，所有tensor维度+1（batch维度）
    """
```

**处理流程**:

```python
# 1. 过滤空样本
batch = [item for item in batch if isinstance(item, dict)]
if not batch:
    return {}

# 2. 确定hand_model_pose和se3的期望shape
expected_shapes = {}
for item in batch:
    if 'hand_model_pose' in item and isinstance(item['hand_model_pose'], torch.Tensor):
        expected_shapes['hand_model_pose'] = item['hand_model_pose'].shape  # (num_grasps, 23)
    if 'se3' in item and isinstance(item['se3'], torch.Tensor):
        expected_shapes['se3'] = item['se3'].shape  # (num_grasps, 4, 4)
    if both found:
        break

# 3. 对每个key进行collate
collated_output = {}
for key in all_keys:
    items = [d.get(key) for d in batch]
    
    if key in ['hand_model_pose', 'se3']:
        # 固定shape的张量，直接stack
        valid_tensors = []
        expected_shape = expected_shapes[key]
        
        for item in items:
            if isinstance(item, torch.Tensor) and item.shape == expected_shape:
                valid_tensors.append(item)
            else:
                # 填充零张量
                dtype = item.dtype if isinstance(item, torch.Tensor) else torch.float32
                device = item.device if isinstance(item, torch.Tensor) else 'cpu'
                valid_tensors.append(torch.zeros(expected_shape, dtype=dtype, device=device))
        
        # Stack: (batch_size, num_grasps, ...)
        collated_output[key] = torch.stack(valid_tensors)
    
    elif key in ['obj_verts', 'obj_faces', 'positive_prompt', 'negative_prompts', 'error']:
        # 保持为列表
        collated_output[key] = items
    
    else:
        # 其他字段使用默认collate
        try:
            collated_output[key] = torch.utils.data.dataloader.default_collate(items)
        except:
            collated_output[key] = items

return collated_output
```

**批处理后的数据格式**:
```python
{
    'scene_pc': torch.Tensor (batch_size, 10000, 6),
    'object_mask': torch.Tensor (batch_size, 10000),
    'hand_model_pose': torch.Tensor (batch_size, num_grasps, 23),
    'se3': torch.Tensor (batch_size, num_grasps, 4, 4),
    'obj_verts': List[torch.Tensor],  # length=batch_size, each (V_i, 3)
    'obj_faces': List[torch.Tensor],  # length=batch_size, each (F_i, 3)
    'positive_prompt': List[str],     # length=batch_size
    'negative_prompts': List[List[str]],  # shape (batch_size, num_neg_prompts)
    'obj_code': List[str],
    'scene_id': List[str],
    'category_id_from_object_index': torch.Tensor (batch_size,),
    'depth_view_index': torch.Tensor (batch_size,)
}
```

---

## 四、坐标系统总结

### 4.1 涉及的坐标系

| 坐标系 | 缩写 | 定义 | 用途 |
|-------|-----|------|------|
| Grasp World Frame | GW | LEAP生成抓取时的世界坐标系 | 原始grasp数据存储 |
| Object Model Frame | OMF | 物体mesh的原始坐标系 | 与物体相关的数据 |
| Camera Frame | CF | 相机坐标系（Z轴向前，X右，Y下） | 点云和图像坐标系 |

### 4.2 坐标变换链

```
原始数据加载:
grasp_qpos (GW) + obj_pose (GW) ──────────────┐
                                              │
                            transform_to_object_centric
                                              │
                                              ▼
                                    hand_pose_data (OMF)
                                              │
                                              │
                      __getitem__时根据mode选择转换
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
              object_centric           camera_centric          camera_centric_*_normalized
                    │                         │                         │
                    ▼                         ▼                         ▼
             保持在OMF                 转换到CF                    转换到CF后归一化
                                              │
                                        OMF → CF 使用:
                                        R_m2c, t_m2c
```

### 4.3 关键变换矩阵

**T_OMF_to_CF (从scene_gt.json获取)**:
```
T = [R_m2c  |  t_m2c]
    [  0    |    1  ]

其中:
- R_m2c: (3, 3) 旋转矩阵
- t_m2c: (3,) 平移向量 (单位: 米)

正向变换:
P_cf = R_m2c @ P_omf + t_m2c
Q_cf = Q_R_m2c * Q_omf

逆变换(CF→OMF):
R_cm = R_m2c.T
t_cm = -R_m2c.T @ t_m2c
P_omf = R_cm @ P_cf + t_cm
Q_omf = Q_R_cm * Q_cf
```

---

## 五、数据维度速查表

### 5.1 单个样本 `__getitem__` 返回

| 字段 | 类型 | Shape | 说明 |
|-----|------|-------|------|
| `scene_pc` | torch.Tensor | (10000, 6) | 点云 xyz+rgb，float32 |
| `object_mask` | torch.Tensor | (10000,) | 物体mask，bool |
| `hand_model_pose` | torch.Tensor | (num_grasps, 23) | 手部姿态，float32 |
| `se3` | torch.Tensor | (num_grasps, 4, 4) | SE(3)矩阵，float32 |
| `obj_verts` | torch.Tensor | (V, 3) | 物体顶点，float32 |
| `obj_faces` | torch.Tensor | (F, 3) | 物体面片，int64 |
| `positive_prompt` | str | - | 正面文本提示 |
| `negative_prompts` | List[str] | (num_neg_prompts,) | 负面文本提示 |
| `obj_code` | str | - | 物体代码 |
| `scene_id` | str | - | 场景ID |
| `category_id_from_object_index` | int | - | 物体类别ID |
| `depth_view_index` | int | - | 深度视图索引 |

### 5.2 批处理后 `collate_fn` 返回

| 字段 | 类型 | Shape | 说明 |
|-----|------|-------|------|
| `scene_pc` | torch.Tensor | (B, 10000, 6) | 批量点云 |
| `object_mask` | torch.Tensor | (B, 10000) | 批量mask |
| `hand_model_pose` | torch.Tensor | (B, num_grasps, 23) | 批量手部姿态 |
| `se3` | torch.Tensor | (B, num_grasps, 4, 4) | 批量SE(3)矩阵 |
| `obj_verts` | List[Tensor] | [(V_1,3), ..., (V_B,3)] | 变长顶点列表 |
| `obj_faces` | List[Tensor] | [(F_1,3), ..., (F_B,3)] | 变长面片列表 |
| `positive_prompt` | List[str] | (B,) | 批量正面提示 |
| `negative_prompts` | List[List[str]] | (B, num_neg_prompts) | 批量负面提示 |
| 其他字段 | List or Tensor | (B,) | 批量标量字段 |

---

## 六、关键配置参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `num_grasps` | 8 | 每个样本返回的抓取数量 |
| `mode` | "camera_centric" | 坐标系模式 |
| `max_grasps_per_object` | 200 | 每个物体最多保留的抓取数 |
| `mesh_scale` | 0.1 | 物体mesh缩放因子 |
| `num_neg_prompts` | 4 | 负面提示数量 |
| `enable_cropping` | True | 是否启用点云裁剪 |
| `max_points` | 10000 | 下采样后的点数 |
| `grasp_sampling_strategy` | "random" | 抓取采样策略 |
| `use_exhaustive_sampling` | False | 是否使用穷尽采样 |
| `exhaustive_sampling_strategy` | "sequential" | 穷尽采样策略 |

---

## 七、性能优化点

### 7.1 缓存机制
```python
# io_utils.py 中的所有加载函数都使用 @lru_cache
@lru_cache(maxsize=None)
def load_depth_image(path): ...

@lru_cache(maxsize=None)
def load_rgb_image(path): ...

@lru_cache(maxsize=None)
def load_mask_image(path): ...

@lru_cache(maxsize=None)
def load_object_mesh(obj_root_dir, object_code, mesh_scale): ...
```
- 避免重复加载相同文件
- 特别适合多个样本共享同一场景/物体的情况

### 7.2 预处理
- 在初始化时完成所有手部姿态的坐标转换
- 过滤无碰撞抓取，减少运行时判断
- 构建数据索引，加快样本查找

### 7.3 分层采样
- 点云下采样时保持物体/背景比例
- 避免物体点过少导致的训练不平衡

---

## 八、错误处理

### 8.1 各阶段错误返回
每个加载阶段失败时都会返回标准化的错误字典:
```python
{
    'obj_code': object_code,
    'scene_pc': torch.zeros((0, 6)),
    'object_mask': torch.zeros((0), dtype=torch.bool),
    'hand_model_pose': torch.zeros((0, 23), dtype=torch.float32),  # 批量格式
    'se3': torch.zeros((0, 4, 4), dtype=torch.float32),
    'scene_id': scene_id,
    'category_id_from_object_index': -1,
    'depth_view_index': -1,
    'obj_verts': torch.zeros((0, 3), dtype=torch.float32),
    'obj_faces': torch.zeros((0, 3), dtype=torch.long),
    'positive_prompt': object_name,
    'negative_prompts': [...],
    'error': error_message  # 错误描述字符串
}
```

### 8.2 错误场景
- 文件不存在（深度图、RGB图、mask等）
- 物体mesh加载失败
- 无可用的无碰撞抓取
- 坐标转换失败
- 点云处理异常

---

## 九、与其他数据集类的对比

| 特性 | SceneLeapProDataset | ForMatchSceneLeapProDataset | SceneLeapPlusDataset |
|------|-------------------|----------------------------|---------------------|
| 抓取数量 | 1个 | 所有可用抓取 | 固定num_grasps个 |
| 数据索引 | 每个抓取一个item | 每个物体+视图一个item | 每个物体+视图一个item |
| `hand_model_pose` shape | (23,) | (N, 23)变长 | (num_grasps, 23)固定 |
| `se3` shape | (4, 4) | (N, 4, 4)变长 | (num_grasps, 4, 4)固定 |
| collate复杂度 | 简单stack | 需要padding | 简单stack |
| 适用场景 | 单抓取训练 | 批量评估/匹配 | 并行多抓取学习 |
| 采样策略 | 无 | 无（返回全部） | 支持多种策略 |
| 穷尽采样 | 不支持 | 不需要 | 支持 |

---

## 十、使用示例

```python
from datasets.sceneleapplus_dataset import SceneLeapPlusDataset
from torch.utils.data import DataLoader

# 1. 创建数据集
dataset = SceneLeapPlusDataset(
    root_dir="/path/to/scenes",
    succ_grasp_dir="/path/to/grasps",
    obj_root_dir="/path/to/objects",
    num_grasps=8,
    mode="camera_centric",
    max_grasps_per_object=200,
    mesh_scale=0.1,
    num_neg_prompts=4,
    enable_cropping=True,
    max_points=10000,
    grasp_sampling_strategy="farthest_point",
    use_exhaustive_sampling=False
)

# 2. 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collate_fn
)

# 3. 迭代数据
for batch in dataloader:
    scene_pc = batch['scene_pc']            # (16, 10000, 6)
    hand_poses = batch['hand_model_pose']   # (16, 8, 23)
    se3 = batch['se3']                      # (16, 8, 4, 4)
    obj_verts = batch['obj_verts']          # List[Tensor], length=16
    
    # 训练代码...
```

---

## 十一、总结

SceneLeapPlusDataset的数据处理pipeline包含以下核心环节:

1. **初始化阶段**: 加载场景元数据、手部姿态数据（含LEAP反转换和坐标系转换）、过滤无碰撞抓取
2. **图像加载**: 从PNG文件加载深度图、RGB图、实例mask
3. **点云生成**: 使用相机内参将深度图转换为3D点云，添加RGB信息
4. **Mask提取**: 从2D实例mask映射到3D点云mask
5. **数据采样**: 根据策略采样固定数量的抓取姿态
6. **坐标转换**: 根据mode选择物体中心、相机中心或归一化坐标系
7. **数据打包**: 点云下采样、生成文本提示、创建SE(3)矩阵、格式化输出

整个pipeline设计注重:
- **模块化**: 每个处理环节独立可测
- **灵活性**: 支持多种坐标系模式和采样策略
- **高效性**: 缓存机制减少重复加载
- **鲁棒性**: 完善的错误处理机制

生成日期: 2025-10-21

graph TD
    A["原始文件<br/>{obj_root_dir}/{object_name}_{uid}/mesh/simplified.obj"] --> B["load_object_mesh()<br/>io_utils.py"]
    
    B --> C["pytorch3d.io.load_obj()"]
    C --> D["verts: torch.Tensor (V, 3)<br/>faces.verts_idx: torch.Tensor (F, 3)"]
    
    D --> E["应用缩放<br/>verts = verts * mesh_scale<br/>默认 mesh_scale=0.1"]
    
    E --> F["缓存到内存<br/>@lru_cache装饰器"]
    
    F --> G["存储到grasp_object_data<br/>obj_verts: (V, 3) 在OMF坐标系<br/>obj_faces: (F, 3)"]
    
    G --> H{mode选择}
    
    H -->|object_centric| I1["保持OMF坐标系<br/>obj_verts不变"]
    H -->|camera_centric| I2["OMF → CF<br/>V_cf = V_omf @ R_m2c.T + t_m2c"]
    H -->|camera_centric_obj_normalized| I3["OMF → CF → 归一化<br/>V_norm = V_cf - obj_mean"]
    H -->|camera_centric_scene_normalized| I4["OMF → CF → 归一化<br/>V_norm = V_cf - scene_mean"]
    
    I1 --> J["最终返回字典"]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K["obj_verts: torch.Tensor (V, 3)<br/>obj_faces: torch.Tensor (F, 3)"]
    
    K --> L["collate_fn批处理"]
    L --> M["obj_verts: List[Tensor] - 变长列表<br/>obj_faces: List[Tensor] - 变长列表<br/>不做stack，保持为List"]
    
    style A fill:#e1f5ff
    style D fill:#fff4e1
    style K fill:#e8f5e9
    style M fill:#f3e5f5


graph TD
    A["原始文件<br/>{succ_grasp_dir}/{object_name}_{uid}.npy"] --> B["初始化阶段加载<br/>load_and_process_hand_pose_data()"]
    
    B --> C["np.load() 得到字典<br/>grasp_qpos: (N, 23)<br/>obj_pose: (7,)"]
    
    C --> D["数据在抓取世界坐标系GW<br/>grasp_qpos: [P_gw, Q_gw, Joints]<br/>obj_pose: [t_obj_gw, q_obj_gw]"]
    
    D --> E["坐标转换: GW → OMF<br/>transform_hand_poses_to_object_centric_frame()"]
    
    E --> E1["计算物体逆变换<br/>q_inv = quaternion_invert(q_obj_gw)<br/>R_inv = quaternion_to_matrix(q_inv)"]
    E1 --> E2["位置转换<br/>P_omf = R_inv @ (P_gw - t_obj_gw)"]
    E2 --> E3["旋转转换<br/>Q_omf = q_inv * Q_gw"]
    E3 --> E4["输出: (N, 23) 在OMF中"]
    
    E4 --> F["LEAP格式反转换<br/>revert_leap_qpos_static()"]
    
    F --> F1["应用Delta旋转<br/>delta_rot = [0,1,0,0]<br/>R_orig = R_leap @ DeltaR"]
    F1 --> F2["应用局部偏移<br/>offset = [0,0,0.1]<br/>P_orig = P_leap + R_leap @ offset"]
    F2 --> F3["输出: (N, 23) 反转换后"]
    
    F3 --> G["存储到hand_pose_data<br/>{scene_id}{object_code}: Tensor (N, 23)"]
    
    G --> H["过滤无碰撞抓取<br/>_filter_collision_free_poses()"]
    
    H --> H1["从collision_free_grasp_info<br/>获取无碰撞索引列表"]
    H1 --> H2["保留索引对应的姿态<br/>filtered: (N_cf, 23)"]
    H2 --> H3{需要限制数量?}
    H3 -->|是| H4["应用max_grasps_per_object<br/>和采样策略"]
    H3 -->|否| H5["保持所有姿态"]
    H4 --> I
    H5 --> I
    
    I["__getitem__时采样<br/>_get_fixed_number_hand_poses()"]
    
    I --> I1{穷尽采样模式?}
    I1 -->|是| I2["使用预指定的grasp_indices"]
    I1 -->|否| I3["使用采样策略"]
    
    I3 --> I4{抓取数量}
    I4 -->|足够| I5["根据strategy采样<br/>random/first_n/FPS/NPS"]
    I4 -->|不足| I6["全部使用+随机重复填充"]
    
    I2 --> J
    I5 --> J
    I6 --> J
    
    J["得到固定数量<br/>(num_grasps, 23)"]
    
    J --> K{mode选择}
    
    K -->|object_centric| L1["保持OMF坐标系<br/>grasps不变"]
    K -->|camera_centric| L2["OMF → CF<br/>transform_hand_poses_omf_to_cf()"]
    K -->|*_normalized| L3["OMF → CF → 归一化<br/>位置减去mean"]
    
    L2 --> L2a["位置: P_cf = P_omf @ R_m2c.T + t_m2c"]
    L2a --> L2b["旋转: Q_cf = Q_R_m2c * Q_omf"]
    L2b --> L2c["关节: Joints不变"]
    
    L3 --> L3a["先转到CF"]
    L3a --> L3b["计算mean<br/>obj_mean 或 scene_mean"]
    L3b --> L3c["P_norm = P_cf - mean"]
    
    L1 --> M
    L2c --> M
    L3c --> M
    
    M["重排序组件<br/>_process_batch_grasps()"]
    
    M --> M1["原格式: [P, Q, Joints]<br/>目标格式: [P, Joints, Q]"]
    M1 --> M2["hand_model_pose_reordered<br/>(num_grasps, 23)"]
    
    M2 --> N["同时创建SE3矩阵"]
    N --> N1["处理零四元数<br/>替换为单位四元数[1,0,0,0]"]
    N1 --> N2["四元数→旋转矩阵<br/>R = quaternion_to_matrix(Q)"]
    N2 --> N3["组装SE3矩阵<br/>SE3 = [[R | P], [0 | 1]]"]
    N3 --> N4["se3: (num_grasps, 4, 4)"]
    
    M2 --> O["最终返回"]
    N4 --> O
    
    O --> P["hand_model_pose: (num_grasps, 23)<br/>se3: (num_grasps, 4, 4)"]
    
    P --> Q["collate_fn批处理"]
    Q --> Q1["检查每个样本的shape"]
    Q1 --> Q2{shape一致?}
    Q2 -->|是| Q3["直接stack"]
    Q2 -->|否| Q4["用零tensor填充到期望shape"]
    Q3 --> R
    Q4 --> R
    
    R["hand_model_pose: (batch, num_grasps, 23)<br/>se3: (batch, num_grasps, 4, 4)"]
    
    style A fill:#e1f5ff
    style D fill:#fff4e1
    style G fill:#ffe1e1
    style J fill:#e8f5e9
    style P fill:#e8f5e9
    style R fill:#f3e5f5
    style E1 fill:#ffeaa7
    style E2 fill:#ffeaa7
    style E3 fill:#ffeaa7
    style F1 fill:#fab1a0
    style F2 fill:#fab1a0
    style L2a fill:#74b9ff
    style L2b fill:#74b9ff
    style M1 fill:#a29bfe
    style N2 fill:#fd79a8
    style N3 fill:#fd79a8