"""
测试PTv3默认配置（grid_size=0.003, token_strategy=last_layer）

验证新的默认配置是否工作正常
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import torch
from omegaconf import OmegaConf

from models.backbone.ptv3_sparse_encoder import PTv3SparseEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_default_config():
    """测试使用配置文件的默认参数"""
    logger.info("="*80)
    logger.info("测试1：从配置文件加载默认参数")
    logger.info("="*80)
    
    # 加载配置文件
    config_path = project_root / "config/model/flow_matching/decoder/backbone/ptv3_sparse.yaml"
    cfg = OmegaConf.load(config_path)
    
    logger.info(f"配置文件加载成功:")
    logger.info(f"  grid_size: {cfg.grid_size}")
    logger.info(f"  target_num_tokens: {cfg.target_num_tokens}")
    logger.info(f"  token_strategy: {cfg.token_strategy}")
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PTv3SparseEncoder(cfg)
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ 模型创建成功")
    logger.info(f"  实际 grid_size: {model.grid_size}")
    logger.info(f"  实际 target_num_tokens: {model.target_num_tokens}")
    logger.info(f"  实际 token_strategy: {model.token_strategy}")
    
    # 测试前向传播
    batch_size = 2
    num_points = 8192
    test_input = torch.randn(batch_size, num_points, 3, device=device)
    
    with torch.no_grad():
        xyz_out, feat_out = model(test_input)
    
    logger.info(f"✓ 前向传播成功")
    logger.info(f"  输入形状: {test_input.shape}")
    logger.info(f"  xyz输出: {xyz_out.shape}")
    logger.info(f"  feat输出: {feat_out.shape}")
    
    # 验证有效token数量
    valid_tokens = (xyz_out.abs().sum(dim=-1) > 0).sum(dim=1)
    logger.info(f"  有效token数: {valid_tokens.tolist()}")
    
    # 验证输出形状
    assert xyz_out.shape == (batch_size, cfg.target_num_tokens, 3), "xyz形状不匹配"
    assert feat_out.shape == (batch_size, cfg.out_dim, cfg.target_num_tokens), "feat形状不匹配"
    
    logger.info("✓ 形状验证通过")
    
    return True


def test_code_defaults():
    """测试代码中的默认参数（不使用配置文件）"""
    logger.info("\n" + "="*80)
    logger.info("测试2：使用代码默认参数（无配置文件）")
    logger.info("="*80)
    
    # 创建最小配置
    minimal_cfg = OmegaConf.create({
        'encoder_channels': [32, 64, 128, 256],
        'encoder_depths': [1, 1, 2, 2],
        'encoder_num_head': [2, 4, 8, 16],
        'enc_patch_size': [1024, 1024, 1024, 1024],
        'stride': [2, 2, 2],
        'out_dim': 256,
        'input_feature_dim': 1,
        'mlp_ratio': 2,
        # 不指定 grid_size, target_num_tokens, token_strategy
        # 应该使用代码中的默认值
    })
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PTv3SparseEncoder(minimal_cfg)
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ 模型创建成功（使用代码默认值）")
    logger.info(f"  默认 grid_size: {model.grid_size} (期望: 0.003)")
    logger.info(f"  默认 target_num_tokens: {model.target_num_tokens} (期望: 128)")
    logger.info(f"  默认 token_strategy: {model.token_strategy} (期望: last_layer)")
    
    # 验证默认值
    assert model.grid_size == 0.003, f"grid_size默认值错误: {model.grid_size}"
    assert model.target_num_tokens == 128, f"target_num_tokens默认值错误: {model.target_num_tokens}"
    assert model.token_strategy == 'last_layer', f"token_strategy默认值错误: {model.token_strategy}"
    
    logger.info("✓ 默认值验证通过")
    
    # 测试前向传播
    batch_size = 2
    num_points = 8192
    test_input = torch.randn(batch_size, num_points, 3, device=device)
    
    with torch.no_grad():
        xyz_out, feat_out = model(test_input)
    
    logger.info(f"✓ 前向传播成功")
    logger.info(f"  输出形状: xyz={xyz_out.shape}, feat={feat_out.shape}")
    
    # 验证有效token数量
    valid_tokens = (xyz_out.abs().sum(dim=-1) > 0).sum(dim=1)
    logger.info(f"  有效token数: {valid_tokens.tolist()} / 128")
    
    return True


def main():
    """主测试函数"""
    logger.info("PTv3 默认配置测试")
    logger.info("验证 grid_size=0.003 和 token_strategy=last_layer 的默认配置\n")
    
    try:
        # 测试1：配置文件
        test1_pass = test_default_config()
        
        # 测试2：代码默认值
        test2_pass = test_code_defaults()
        
        logger.info("\n" + "="*80)
        logger.info("✅ 所有测试通过！")
        logger.info("="*80)
        logger.info("默认配置已成功设置为:")
        logger.info("  - grid_size: 0.003")
        logger.info("  - target_num_tokens: 128")
        logger.info("  - token_strategy: last_layer")
        logger.info("\n推荐使用方式:")
        logger.info("  cfg = OmegaConf.load('config/.../ptv3_sparse.yaml')")
        logger.info("  model = PTv3SparseEncoder(cfg)")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

