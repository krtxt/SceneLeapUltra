"""
设置grasp_gen导入路径的辅助模块
"""
import os
import sys
import types

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bin_path = os.path.join(project_root, 'bin')

# 添加路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if bin_path not in sys.path:
    sys.path.insert(0, bin_path)

# 创建grasp_gen包结构
def setup_grasp_gen_imports():
    """设置grasp_gen包的导入路径"""
    # 创建grasp_gen模块
    if 'grasp_gen' not in sys.modules:
        grasp_gen_module = types.ModuleType('grasp_gen')
        sys.modules['grasp_gen'] = grasp_gen_module
    
    # 创建grasp_gen.models
    if 'grasp_gen.models' not in sys.modules:
        models_module = types.ModuleType('grasp_gen.models')
        sys.modules['grasp_gen.models'] = models_module
    
    # 创建grasp_gen.models.pointnet
    if 'grasp_gen.models.pointnet' not in sys.modules:
        pointnet_module = types.ModuleType('grasp_gen.models.pointnet')
        sys.modules['grasp_gen.models.pointnet'] = pointnet_module
    
    # 创建grasp_gen.models.ptv3
    if 'grasp_gen.models.ptv3' not in sys.modules:
        ptv3_module = types.ModuleType('grasp_gen.models.ptv3')
        sys.modules['grasp_gen.models.ptv3'] = ptv3_module
    
    # 创建grasp_gen.utils
    if 'grasp_gen.utils' not in sys.modules:
        utils_module = types.ModuleType('grasp_gen.utils')
        sys.modules['grasp_gen.utils'] = utils_module
    
    # 创建grasp_gen.utils.logging_config
    if 'grasp_gen.utils.logging_config' not in sys.modules:
        logging_config_module = types.ModuleType('grasp_gen.utils.logging_config')
        import logging
        logging.basicConfig(level=logging.INFO)
        def get_logger(name):
            return logging.getLogger(name)
        logging_config_module.get_logger = get_logger
        sys.modules['grasp_gen.utils.logging_config'] = logging_config_module
    
    # 导入并映射pointnet2_utils
    try:
        from bin.grasp_gen_models.pointnet import pointnet2_utils
        sys.modules['grasp_gen.models.pointnet.pointnet2_utils'] = pointnet2_utils
    except Exception as e:
        print(f"Warning: Could not import pointnet2_utils: {e}")
    
    # 导入并映射pointnet2_modules
    try:
        from bin.grasp_gen_models.pointnet import pointnet2_modules
        sys.modules['grasp_gen.models.pointnet.pointnet2_modules'] = pointnet2_modules
    except Exception as e:
        print(f"Warning: Could not import pointnet2_modules: {e}")

