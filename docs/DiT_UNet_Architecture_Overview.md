# SceneLeapPlus 模型架构总览：DiT 与 UNet

本文档基于源码整理与可视化，涵盖模型文件：
- [models/decoder/dit.py](cci:7://file:///home/xiantuo/source/grasp/SceneLeapPlus/models/decoder/dit.py:0:0-0:0)
- [models/decoder/unet_new.py](cci:7://file:///home/xiantuo/source/grasp/SceneLeapPlus/models/decoder/unet_new.py:0:0-0:0)
- [models/utils/diffusion_utils.py](cci:7://file:///home/xiantuo/source/grasp/SceneLeapPlus/models/utils/diffusion_utils.py:0:0-0:0)

并给出两类结构的整体流程与Block级细节（竖版），以及关键差异对比。

---

## 1. DiT 模型整体结构

```mermaid
graph TB
    subgraph Input["输入数据"]
        NoisyInput["Noisy Input (x_t)<br/>(B, num_grasps, 23/25)"]
        ScenePC["Scene PC<br/>(B, N_points, 3/4)"]
        TextPrompt["Text Prompt<br/>(List[str])"]
        Timestep["Timestep (ts)<br/>(B,)"]
    end
    
    subgraph TextProcessing["文本处理分支"]
        TextPrompt --> CLIP["PosNegTextEncoder<br/>(CLIP-based)"]
        CLIP --> TextProcessor["TextConditionProcessor<br/>(MLP + Dropout)"]
        TextProcessor --> TextEmbed["Text Embedding<br/>(B, 512) → (B, d_model)"]
    end
    
    subgraph TimestepProcessing["时间步处理"]
        Timestep --> TSEmbed["timestep_embedding<br/>(sinusoidal)"]
        TSEmbed --> TSMLP["TimestepEmbedding<br/>(Linear + SiLU + Linear)"]
        TSMLP --> TimeEmbed["Time Embedding<br/>(B, time_embed_dim)"]
    end
    
    subgraph SceneProcessing["场景处理分支"]
        ScenePC --> Backbone["Backbone<br/>(PointNet++/etc)"]
        Backbone --> SceneProj["SceneProjection<br/>(Linear: 512 → d_model)"]
        SceneProj --> SceneContext["Scene Context<br/>(B, N_points, d_model)"]
    end
    
    subgraph MainPipeline["主处理流程"]
        NoisyInput --> Tokenizer["GraspTokenizer<br/>(Linear + LayerNorm)"]
        Tokenizer --> Tokens["Grasp Tokens<br/>(B, num_grasps, d_model)"]
        Tokens --> PosEmb["PositionalEmbedding<br/>(可选, learnable)"]
        PosEmb --> TokensPos["Tokens + Pos<br/>(B, num_grasps, d_model)"]
        
        TokensPos --> DiTBlock1["DiT Block 1"]
        DiTBlock1 --> DiTBlock2["DiT Block 2"]
        DiTBlock2 --> DiTBlockN["...<br/>DiT Block N"]
        
        DiTBlockN --> OutProj["OutputProjection<br/>(LayerNorm + Linear)"]
        OutProj --> Output["Output<br/>(B, num_grasps, 23/25)"]
    end
    
    TimeEmbed -.时间条件.-> DiTBlock1
    TimeEmbed -.时间条件.-> DiTBlock2
    TimeEmbed -.时间条件.-> DiTBlockN
    
    SceneContext -.场景条件.-> DiTBlock1
    SceneContext -.场景条件.-> DiTBlock2
    SceneContext -.场景条件.-> DiTBlockN
    
    TextEmbed -.文本条件.-> DiTBlock1
    TextEmbed -.文本条件.-> DiTBlock2
    TextEmbed -.文本条件.-> DiTBlockN

    