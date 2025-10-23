# PTv3 导入路径修复

## 问题描述

PTv3 官方代码中包含硬编码的导入路径 `from grasp_gen.models.ptv3...`，导致在我们的项目中无法正常导入。

## 修复的文件

### 1. models/backbone/ptv3/ptv3.py
```python
# 修复前
from grasp_gen.models.ptv3.serialization.default import encode

# 修复后
from .serialization.default import encode
```

### 2. models/backbone/ptv3/serialization/default.py
```python
# 修复前
from grasp_gen.models.ptv3.serialization.hilbert import decode as hilbert_decode_
from grasp_gen.models.ptv3.serialization.hilbert import encode as hilbert_encode_
from grasp_gen.models.ptv3.serialization.z_order import key2xyz as z_order_decode_
from grasp_gen.models.ptv3.serialization.z_order import xyz2key as z_order_encode_

# 修复后
from .hilbert import decode as hilbert_decode_
from .hilbert import encode as hilbert_encode_
from .z_order import key2xyz as z_order_decode_
from .z_order import xyz2key as z_order_encode_
```

## 验证方法

```bash
# 激活环境
source ~/.bashrc && conda activate DexGrasp

# 测试导入
python -c "from models.backbone.ptv3_backbone import PTV3Backbone; print('✓ Import OK')"

# 测试完整模型导入
python -c "from models.decoder.dit import DiTModel; from models.decoder.unet_new import UNetModel; print('✓ All OK')"
```

## 修复状态

✅ **已完成** - 所有硬编码导入路径已修复为相对导入。

现在可以正常使用 PTv3 backbone 了！

