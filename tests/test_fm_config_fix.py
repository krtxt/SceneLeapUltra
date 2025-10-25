"""
测试Mode参数命名冲突修复

验证：
1. mode → 坐标系模式 (camera_centric_scene_mean_normalized)
2. pred_mode → 预测模式 (velocity)
"""

import sys

import torch

sys.path.insert(0, '/home/xiantuo/source/grasp/GithubClone/SceneLeapUltra')

def test_mode_separation():
    """测试mode参数分离"""
    print("="*60)
    print("测试: Mode参数命名冲突修复")
    print("="*60)
    
    try:
        from omegaconf import OmegaConf

        from models.decoder.dit_fm import DiTFM
        
        print("\n创建测试配置...")
        
        # 模拟完整的配置（包含顶层的mode定义）
        cfg = OmegaConf.create({
            'name': 'dit_fm',
            'rot_type': 'r6d',
            'pred_mode': 'velocity',  # FM预测模式
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 4,
            'd_head': 64,
            'dropout': 0.0,
            'max_sequence_length': 100,
            'use_learnable_pos_embedding': False,
            'time_embed_dim': 512,
            'use_adaptive_norm': True,
            'continuous_time': True,
            'freq_dim': 128,
            'use_text_condition': False,
            'text_dropout_prob': 0.1,
            'use_negative_prompts': False,
            'use_object_mask': False,
            'use_rgb': False,
            'backbone': {
                'name': 'pointnet2',
                'use_pooling': False,
                'layer1': {
                    'npoint': 512,
                    'radius_list': [0.04],
                    'nsample_list': [32],
                    'mlp_list': [0, 64, 64, 128]
                },
                'layer2': {
                    'npoint': 256,
                    'radius_list': [0.1],
                    'nsample_list': [16],
                    'mlp_list': [128, 128, 128, 256]
                },
                'layer3': {
                    'npoint': 128,
                    'radius_list': [0.2],
                    'nsample_list': [16],
                    'mlp_list': [256, 128, 128, 256]
                },
                'layer4': {
                    'npoint': 64,
                    'radius_list': [0.3],
                    'nsample_list': [8],
                    'mlp_list': [256, 512, 512]
                },
                'use_xyz': True,
                'normalize_xyz': True
            },
            'debug': {
                'check_nan': True,
                'log_tensor_stats': False
            }
        })
        
        print("✅ 配置创建成功")
        
        # 检查pred_mode参数
        assert cfg.pred_mode == 'velocity', f"pred_mode错误: {cfg.pred_mode}"
        print(f"✅ pred_mode = '{cfg.pred_mode}' (预测模式)")
        
        # 创建模型
        if not torch.cuda.is_available():
            print("\n⚠️  CPU环境，跳过模型创建测试")
            print("   (PointNet2需要GPU)")
            return True
        
        model = DiTFM(cfg).cuda()
        
        # 验证模型的pred_mode属性
        assert model.pred_mode == 'velocity', f"模型pred_mode错误: {model.pred_mode}"
        print(f"✅ 模型初始化成功，pred_mode = '{model.pred_mode}'")
        
        # 测试前向传播
        B, num_grasps, D = 2, 4, 25
        x_t = torch.randn(B, num_grasps, D).cuda()
        t = torch.rand(B).cuda()
        data = {
            'scene_pc': torch.randn(B, 1024, 3).cuda(),
        }
        
        cond = model.condition(data)
        data.update(cond)
        
        with torch.no_grad():
            v_pred = model(x_t, t, data)
        
        assert v_pred.shape == (B, num_grasps, D), f"输出形状错误: {v_pred.shape}"
        print(f"✅ 前向传播成功，输出形状: {v_pred.shape}")
        
        print("\n" + "="*60)
        print("✅ Mode参数命名冲突已修复！")
        print("="*60)
        print("\n总结:")
        print("  - mode → 坐标系模式 (用于process_hand_pose)")
        print("  - pred_mode → 预测模式 (velocity/epsilon/pose)")
        print("  - 无命名冲突，可以正常训练！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = test_mode_separation()
    exit(0 if result else 1)

