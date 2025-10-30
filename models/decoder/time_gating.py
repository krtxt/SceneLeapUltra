"""
时间门控模块 (Time-aware Gating for Conditioning)

提供时间相关的条件门控机制，用于动态调节 cross-attention 的影响强度。
核心思想：扩散早期施加强约束，后期给模型更多自由度。
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import logging


class CosineSquaredGate:
    """
    余弦平方门控：α(t) = cos²(π/2 * t)
    
    特性：
    - 零参数，无需训练
    - 单调递减，早期强(t=0: α=1.0)，后期弱(t=1: α=0.0)
    - 光滑过渡，稳定性好
    - 符合"前期强约束，后期弱约束"的直觉
    
    Args:
        scale: 缩放因子，用于调整门控强度 (默认1.0)
    """
    def __init__(self, scale: float = 1.0):
        self.scale = scale
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        计算门控因子
        
        Args:
            t: 归一化时间 [0, 1]，shape (B,) or (B, 1)
        
        Returns:
            alpha: 门控因子 [0, 1]，shape (B, 1, 1) 便于广播
        """
        # 确保 t 在 [0, 1] 范围
        t = torch.clamp(t, 0.0, 1.0)
        
        # 计算 cos²(π/2 * t)
        alpha = torch.cos(math.pi / 2.0 * t) ** 2
        
        # 应用缩放因子
        alpha = alpha * self.scale
        # 保持在 [0, 1] 区间，避免缩放>1 时放大跨注意力（与文档一致）
        alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # 扩展维度以便广播: (B,) -> (B, 1, 1)
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        elif alpha.dim() == 2 and alpha.shape[1] == 1:
            alpha = alpha.unsqueeze(-1)
        
        return alpha
    
    def __repr__(self):
        return f"CosineSquaredGate(scale={self.scale})"


class MLPGate(nn.Module):
    """
    可学习的 MLP 门控：通过 MLP 预测门控因子
    
    特性：
    - 可学习，能适应数据分布
    - 输出经 sigmoid 限制在 [0, 1]
    - 支持 warmup：前期输出固定值，后期再学习
    
    Args:
        input_dim: 输入维度（通常是时间嵌入维度）
        hidden_dims: 隐藏层维度列表
        init_value: 初始输出值（warmup期间的固定值）
        warmup_steps: warmup步数，期间输出固定为 init_value
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        init_value: float = 1.0,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        
        # 全局训练步数计数器（用于 warmup）
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        
        # 构建 MLP
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        
        # 输出层：映射到 1 维，经 sigmoid 得到 [0, 1]
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化输出层偏置，使得初始输出接近 init_value
        # sigmoid(x) = init_value => x = log(init_value / (1 - init_value))
        init_logit = math.log(init_value / (1.0 - init_value + 1e-6))
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_logit)
            self.mlp[-1].weight.zero_()
    
    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        """
        计算门控因子
        
        Args:
            time_emb: 时间嵌入，shape (B, input_dim)
        
        Returns:
            alpha: 门控因子 [0, 1]，shape (B, 1, 1) 便于广播
        """
        # Warmup 期间输出固定值
        if self.training and self.global_step < self.warmup_steps:
            batch_size = time_emb.shape[0]
            alpha = torch.full(
                (batch_size, 1, 1),
                self.init_value,
                device=time_emb.device,
                dtype=time_emb.dtype
            )
            return alpha
        
        # 通过 MLP 预测门控
        logits = self.mlp(time_emb)  # (B, 1)
        alpha = torch.sigmoid(logits)  # (B, 1)
        
        # 扩展维度: (B, 1) -> (B, 1, 1)
        alpha = alpha.unsqueeze(-1)
        
        return alpha
    
    def step(self):
        """训练步数递增（需要在训练循环中手动调用）"""
        if self.training:
            self.global_step += 1
    
    def __repr__(self):
        return (f"MLPGate(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
                f"init_value={self.init_value}, warmup_steps={self.warmup_steps}, "
                f"global_step={self.global_step.item()})")


class TimeGate(nn.Module):
    """
    时间门控的统一接口
    
    根据配置创建并管理场景和文本的门控模块。
    支持：
    - 统一门控：场景和文本使用相同的门控函数
    - 分离门控：场景和文本使用独立的门控函数
    - 选择性应用：可以只对场景或文本应用门控
    
    Args:
        gate_config: 门控配置字典
        time_embed_dim: 时间嵌入维度（用于 MLP 门控）
    """
    def __init__(self, gate_config: Dict[str, Any], time_embed_dim: Optional[int] = None):
        super().__init__()
        self.config = gate_config
        self.gate_type = gate_config.get('type', 'cos2')
        self.apply_to = gate_config.get('apply_to', 'both')  # 'both' | 'scene' | 'text'
        self.scene_scale = gate_config.get('scene_scale', 1.0)
        self.text_scale = gate_config.get('text_scale', 1.0)
        self.separate_text_gate = gate_config.get('separate_text_gate', False)
        
        self.logger = logging.getLogger(__name__)
        
        # 创建场景门控
        if self.apply_to in ['both', 'scene']:
            self.scene_gate = self._build_gate(self.gate_type, self.scene_scale, time_embed_dim)
        else:
            self.scene_gate = None
        
        # 创建文本门控
        if self.apply_to in ['both', 'text']:
            if self.separate_text_gate:
                # 独立的文本门控
                self.text_gate = self._build_gate(self.gate_type, self.text_scale, time_embed_dim)
            else:
                # 与场景共享门控，仅缩放不同
                if self.text_scale != self.scene_scale and self.scene_gate is not None:
                    # 需要单独的缩放
                    self.text_gate = self._build_gate(self.gate_type, self.text_scale, time_embed_dim)
                else:
                    # 完全共享
                    self.text_gate = self.scene_gate
        else:
            self.text_gate = None
        
        self.logger.info(
            f"TimeGate initialized: type={self.gate_type}, apply_to={self.apply_to}, "
            f"scene_scale={self.scene_scale}, text_scale={self.text_scale}, "
            f"separate={self.separate_text_gate}"
        )
    
    def _build_gate(self, gate_type: str, scale: float, time_embed_dim: Optional[int]):
        """构建门控模块"""
        if gate_type == 'cos2':
            return CosineSquaredGate(scale=scale)
        
        elif gate_type == 'mlp':
            if time_embed_dim is None:
                raise ValueError("MLP gate requires time_embed_dim")
            
            hidden_dims = self.config.get('mlp_hidden_dims', [256, 128])
            init_value = self.config.get('init_value', 1.0)
            warmup_steps = self.config.get('warmup_steps', 1000)
            
            return MLPGate(
                input_dim=time_embed_dim,
                hidden_dims=hidden_dims,
                init_value=init_value,
                warmup_steps=warmup_steps
            )
        
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def get_scene_gate(self, t: Optional[torch.Tensor] = None, 
                       time_emb: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        获取场景门控因子
        
        Args:
            t: 归一化时间 [0, 1]，用于 cos2 门控
            time_emb: 时间嵌入，用于 MLP 门控
        
        Returns:
            alpha_scene: 场景门控因子，shape (B, 1, 1)，或 None（不应用门控）
        """
        if self.scene_gate is None:
            return None
        
        if self.gate_type == 'cos2':
            if t is None:
                raise ValueError("CosineSquaredGate requires t (normalized time)")
            return self.scene_gate(t)
        
        elif self.gate_type == 'mlp':
            if time_emb is None:
                raise ValueError("MLPGate requires time_emb")
            alpha = self.scene_gate(time_emb)
            # 对 MLP 门控输出应用缩放
            if self.scene_scale != 1.0:
                alpha = alpha * self.scene_scale
            return alpha
        
        return None
    
    def get_text_gate(self, t: Optional[torch.Tensor] = None, 
                      time_emb: Optional[torch.Tensor] = None,
                      text_mask: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        获取文本门控因子
        
        Args:
            t: 归一化时间 [0, 1]，用于 cos2 门控
            time_emb: 时间嵌入，用于 MLP 门控
            text_mask: 文本dropout mask，会与门控因子相乘
        
        Returns:
            alpha_text: 文本门控因子，shape (B, 1, 1)，或 None（不应用门控）
        """
        if self.text_gate is None:
            return None
        
        if self.gate_type == 'cos2':
            if t is None:
                raise ValueError("CosineSquaredGate requires t (normalized time)")
            alpha = self.text_gate(t)
        
        elif self.gate_type == 'mlp':
            if time_emb is None:
                raise ValueError("MLPGate requires time_emb")
            alpha = self.text_gate(time_emb)
            # 对 MLP 门控输出应用缩放
            if self.text_scale != 1.0:
                alpha = alpha * self.text_scale
        
        else:
            return None
        
        # 如果有文本 dropout mask，与门控因子相乘
        if text_mask is not None:
            # text_mask shape: (B, 1) -> (B, 1, 1)
            if text_mask.dim() == 2:
                text_mask = text_mask.unsqueeze(-1)
            alpha = alpha * text_mask
        
        return alpha
    
    def step(self):
        """训练步数递增（用于 MLP 门控的 warmup）"""
        if self.gate_type == 'mlp':
            if self.scene_gate is not None and isinstance(self.scene_gate, MLPGate):
                self.scene_gate.step()
            if (self.text_gate is not None and 
                isinstance(self.text_gate, MLPGate) and 
                self.text_gate is not self.scene_gate):
                self.text_gate.step()
    
    def get_mlp_modules(self):
        """获取所有 MLP 门控模块（用于参数注册）"""
        modules = []
        if isinstance(self.scene_gate, MLPGate):
            modules.append(('scene_gate', self.scene_gate))
        if (isinstance(self.text_gate, MLPGate) and 
            self.text_gate is not self.scene_gate):
            modules.append(('text_gate', self.text_gate))
        return modules


def build_time_gate(
    use_t_aware_conditioning: bool,
    gate_config: Optional[Dict[str, Any]] = None,
    time_embed_dim: Optional[int] = None
) -> Optional[TimeGate]:
    """
    工厂函数：根据配置创建时间门控
    
    Args:
        use_t_aware_conditioning: 是否启用时间门控
        gate_config: 门控配置字典
        time_embed_dim: 时间嵌入维度（用于 MLP 门控）
    
    Returns:
        TimeGate 实例，或 None（未启用）
    """
    if not use_t_aware_conditioning:
        return None
    
    if gate_config is None:
        # 使用默认配置
        gate_config = {
            'type': 'cos2',
            'apply_to': 'both',
            'scene_scale': 1.0,
            'text_scale': 1.0,
            'separate_text_gate': False
        }
    
    return TimeGate(gate_config, time_embed_dim)

