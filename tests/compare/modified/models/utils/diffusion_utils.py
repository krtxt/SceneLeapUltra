from typing import Dict, List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import repeat, rearrange
from inspect import isfunction

def make_schedule_ddpm(timesteps: int, beta: List, beta_schedule: str, s=0.008) -> Dict:
    assert beta[0] < beta[1] < 1.0
    if beta_schedule == 'linear':
        betas = torch.linspace(beta[0], beta[1], timesteps)
    elif beta_schedule == 'cosine':
        x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0, 0.999)
    elif beta_schedule == 'sqrt':
        betas = torch.sqrt(torch.linspace(beta[0], beta[1], timesteps))
    else:
        raise Exception('Unsupport beta schedule.')

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])    
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1 - alphas_cumprod),
        'log_one_minus_alphas_cumprod': torch.log(1 - alphas_cumprod),
        'sqrt_recip_alphas_cumprod': torch.sqrt(1 / alphas_cumprod),
        'sqrt_recipm1_alphas_cumprod': torch.sqrt(1 / alphas_cumprod - 1),
        'posterior_variance': posterior_variance,
        'posterior_log_variance_clipped': torch.log(posterior_variance.clamp(min=1e-20)),
        'posterior_mean_coef1': betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        'posterior_mean_coef2': (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
    }

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    import logging

    # Validate inputs
    if torch.isnan(timesteps).any() or torch.isinf(timesteps).any():
        logging.error(f"[NaN Detection] Invalid timesteps in timestep_embedding")
        logging.error(f"  timesteps: {timesteps}")
        logging.error(f"  NaN count: {torch.isnan(timesteps).sum().item()}, Inf count: {torch.isinf(timesteps).sum().item()}")
        raise RuntimeError("Invalid timesteps in timestep_embedding")

    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)

        # Check freqs for NaN
        if torch.isnan(freqs).any() or torch.isinf(freqs).any():
            logging.error(f"[NaN Detection] NaN/Inf in frequency computation")
            logging.error(f"  freqs: {freqs}")
            logging.error(f"  NaN count: {torch.isnan(freqs).sum().item()}, Inf count: {torch.isinf(freqs).sum().item()}")
            raise RuntimeError("NaN/Inf in timestep embedding frequency computation")

        args = timesteps[:, None].float() * freqs[None]

        # Check args for NaN
        if torch.isnan(args).any() or torch.isinf(args).any():
            logging.error(f"[NaN Detection] NaN/Inf in timestep embedding args")
            logging.error(f"  args shape: {args.shape}")
            logging.error(f"  NaN count: {torch.isnan(args).sum().item()}, Inf count: {torch.isinf(args).sum().item()}")
            logging.error(f"  timesteps: {timesteps}")
            logging.error(f"  freqs: {freqs}")
            raise RuntimeError("NaN/Inf in timestep embedding args")

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Check embedding for NaN
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            logging.error(f"[NaN Detection] NaN/Inf in timestep embedding output")
            logging.error(f"  embedding shape: {embedding.shape}")
            logging.error(f"  NaN count: {torch.isnan(embedding).sum().item()}, Inf count: {torch.isinf(embedding).sum().item()}")
            raise RuntimeError("NaN/Inf in timestep embedding output")

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)

    logging.debug(f"[timestep_embedding] Output: shape={embedding.shape}, min={embedding.min():.6f}, max={embedding.max():.6f}, mean={embedding.mean():.6f}")

    return embedding

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    """

    def __init__(
        self,
        in_channels,
        emb_channels,
        dropout,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = in_channels if out_channels is None else out_channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, self.in_channels),
            nn.SiLU(),
            nn.Conv1d(self.in_channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1)
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(self.in_channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out.unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h



def exists(val):
    return val is not None

def uniq(arr):
    return {el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, mult_ff=2):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, mult=mult_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for sequential data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to sequential data.
    """
    def __init__(self, in_channels, n_heads=8, d_head=64,
                 depth=1, dropout=0., context_dim=None, mult_ff=2):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, mult_ff=mult_ff)
                for d in range(depth)]
        )

        self.proj_out = nn.Conv1d(inner_dim,
                                in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        B, C, L,  = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        x = rearrange(x, 'b c l -> b l c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b l c -> b c l')
        x = self.proj_out(x)
        return x + x_in

class GraspNet(nn.Module):
    """
    抓取姿态编码器，将抓取姿态编码为512维特征向量
    支持单抓取格式 [B, input_dim] 和多抓取格式 [B, num_grasps, input_dim]
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=512):
        super(GraspNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 基础编码网络
        self.grasp_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        # 抓取间交互的自注意力层（仅用于多抓取格式）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, grasp_pose):
        """
        Args:
            grasp_pose: (B, input_dim) 或 (B, num_grasps, input_dim) - 抓取姿态
                       单抓取: quat: 23维, r6d: 25维
                       多抓取: [B, num_grasps, 23/25]
        Returns:
            grasp_embedding: (B, 512) 或 (B, num_grasps, 512) - 抓取姿态的嵌入
        """
        if grasp_pose.dim() == 2:
            # 单抓取格式（向后兼容）
            return self.grasp_encoder(grasp_pose)  # [B, output_dim]

        elif grasp_pose.dim() == 3:
            # 多抓取格式
            B, num_grasps, input_dim = grasp_pose.shape

            # 编码每个抓取: [B*num_grasps, input_dim] -> [B*num_grasps, output_dim] - 确保内存连续
            grasp_flat = grasp_pose.contiguous().view(B * num_grasps, input_dim)
            encoded_flat = self.grasp_encoder(grasp_flat)
            encoded = encoded_flat.contiguous().view(B, num_grasps, self.output_dim)

            # 抓取间自注意力交互
            attended, _ = self.self_attention(encoded, encoded, encoded)

            return attended  # [B, num_grasps, output_dim]

        else:
            raise ValueError(f"Unsupported input dimension: {grasp_pose.dim()}. Expected 2 or 3.")


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦余弦位置编码，用于时间步编码
    """
    def __init__(self, dim=64):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (B,) - 时间步
        Returns:
            time_embedding: (B, dim) - 时间步的位置编码
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块，支持单抓取和多抓取格式
    Query: 抓取姿态 + 文本的融合特征
    Key/Value: 点云特征
    """
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Query投影
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),  # 512 -> 512
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, grasp_text_embedding, scene_features):
        """
        Args:
            grasp_text_embedding: (B, 512) 或 (B, num_grasps, 512) - 抓取姿态+文本的融合特征
            scene_features: (B, N_points, 512) - 点云特征作为Key/Value
        Returns:
            attended_features: (B, 512) 或 (B, num_grasps, 512) - 经过注意力融合的特征
        """
        if grasp_text_embedding.dim() == 2:
            # 单抓取格式（向后兼容）
            return self._forward_single_grasp(grasp_text_embedding, scene_features)

        elif grasp_text_embedding.dim() == 3:
            # 多抓取格式
            return self._forward_multi_grasp(grasp_text_embedding, scene_features)

        else:
            raise ValueError(f"Unsupported grasp_text_embedding dimension: {grasp_text_embedding.dim()}")

    def _forward_single_grasp(self, grasp_text_embedding, scene_features):
        """处理单抓取格式"""
        B = grasp_text_embedding.shape[0]

        # 1. 投影Query并增加序列维度
        query = self.query_projection(grasp_text_embedding).unsqueeze(1)  # (B, 1, 512)

        # 2. 场景特征作为Key和Value
        key = value = scene_features  # (B, N_points, 512)

        # 3. 执行交叉注意力
        attended_output, _ = self.multihead_attn(
            query=query,      # (B, 1, 512)
            key=key,          # (B, N_points, 512)
            value=value       # (B, N_points, 512)
        )

        # 4. 输出投影并压缩序列维度
        output = self.output_proj(attended_output.squeeze(1))  # (B, 512)
        return output

    def _forward_multi_grasp(self, grasp_text_embedding, scene_features):
        """处理多抓取格式"""
        B, num_grasps, embed_dim = grasp_text_embedding.shape

        # 1. 投影Query
        query = self.query_projection(grasp_text_embedding)  # (B, num_grasps, 512)

        # 2. 场景特征作为Key和Value
        key = value = scene_features  # (B, N_points, 512)

        # 3. 批量处理多个抓取的注意力
        attended_list = []
        for i in range(num_grasps):
            query_i = query[:, i:i+1, :]  # (B, 1, 512)

            # 执行交叉注意力
            attended_i, _ = self.multihead_attn(
                query=query_i,    # (B, 1, 512)
                key=key,          # (B, N_points, 512)
                value=value       # (B, N_points, 512)
            )
            attended_list.append(attended_i)

        # 4. 合并所有抓取的注意力结果
        attended_output = torch.cat(attended_list, dim=1)  # (B, num_grasps, 512)

        # 5. 输出投影
        output = self.output_proj(attended_output)  # (B, num_grasps, 512)
        return output


if __name__ == '__main__':
    # 测试原有的SpatialTransformer
    st = SpatialTransformer(256, 8, 64, 6, context_dim=768)
    print(st)
    a = torch.rand(2, 256, 10)
    context = torch.rand(2, 5, 768)
    o = st(a, context=context)
    print(o.shape)

    # 测试新的组件
    print("\n测试新组件:")

    # 测试GraspNet
    grasp_net = GraspNet(input_dim=23, output_dim=512)
    grasp_input = torch.rand(4, 23)
    grasp_output = grasp_net(grasp_input)
    print(f"GraspNet: {grasp_input.shape} -> {grasp_output.shape}")

    # 测试SinusoidalPositionEmbeddings
    time_encoder = SinusoidalPositionEmbeddings(dim=64)
    time_input = torch.randint(0, 1000, (4,))
    time_output = time_encoder(time_input)
    print(f"TimeEncoder: {time_input.shape} -> {time_output.shape}")

    # 测试CrossAttentionFusion
    fusion = CrossAttentionFusion()
    grasp_text = torch.rand(4, 512)
    time_emb = torch.rand(4, 64)
    scene_feat = torch.rand(4, 128, 512)
    fusion_output = fusion(grasp_text, scene_feat)
    print(f"CrossAttentionFusion: grasp_text{grasp_text.shape} + time{time_emb.shape} + scene{scene_feat.shape} -> {fusion_output.shape}")

