# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

使用简体中文回答

## Project Overview

SceneLeapUltra is a PyTorch Lightning-based project for generating grasp poses using diffusion models. The project supports two main model architectures:
- **GraspDiffuser**: Diffusion-based grasp generation with DiT (Diffusion Transformer) or UNet decoders
- **GraspCVAE**: Conditional VAE-based grasp generation

The system generates grasp poses conditioned on:
- 3D scene point clouds (with optional RGB and object masks)
- Text prompts (using CLIP or T5 encoders)
- Negative prompts for classifier-free guidance

## Environment Setup

**Required conda environment**: DexGrasp

Before running any commands, activate the environment:
```bash
source ~/.bashrc && conda activate DexGrasp
```

## Common Commands

### Training

**Recommended**: Use the distributed training script (supports single and multi-GPU):
```bash
# Auto-detect all GPUs and train
./train_distributed.sh

# Specify GPU count
./train_distributed.sh --gpus 4

# Customize hyperparameters via Hydra overrides
./train_distributed.sh --gpus 4 batch_size=128 model.optimizer.lr=0.002

# Specify save location
./train_distributed.sh --gpus 4 save_root="./experiments/my_experiment"

# Resume from checkpoint
./train_distributed.sh --gpus 4 resume=true checkpoint_path="path/to/checkpoint.ckpt"
```

**Legacy**: The `train_lightning.sh` script exists for backward compatibility but `train_distributed.sh` is preferred.

### Testing

Test a single checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +"checkpoint_path='experiments/my_exp/checkpoints/epoch=100-val_loss=11.50.ckpt'" \
    data.test.batch_size=64
```

Test all checkpoints in an experiment directory:
```bash
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +train_root='experiments/my_exp' \
    data.test.batch_size=64
```

Force retest (ignore cached results):
```bash
CUDA_VISIBLE_DEVICES=0 python test_lightning.py \
    +train_root='experiments/my_exp' \
    +force_retest=true
```

### Direct Python Training

For more control, use the Python scripts directly:
```bash
# Single GPU
python train_lightning.py

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 train_lightning.py distributed.enabled=true
```

## Architecture Overview

### High-Level Structure

```
SceneLeapUltra/
├── config/              # Hydra configuration files (YAML)
│   ├── data_cfg/        # Dataset configurations
│   ├── model/           # Model architectures
│   │   ├── diffuser/    # Diffusion model configs
│   │   └── cvae/        # CVAE model configs
│   ├── distributed/     # Distributed training settings
│   └── wandb/           # Weights & Biases logging
├── models/              # Model implementations
│   ├── backbone/        # Point cloud encoders (PointNet2, PTv3)
│   ├── decoder/         # Decoders (DiT, UNet)
│   ├── loss/            # Loss functions and matchers
│   └── utils/           # Model utilities (diffusion, text encoding)
├── datasets/            # PyTorch datasets and dataloaders
├── utils/               # General utilities (hand model, evaluation, etc.)
├── train_lightning.py   # Main training script
├── test_lightning.py    # Testing/evaluation script
└── train_distributed.sh # Recommended training wrapper
```

### Key Architectural Components

#### 1. Diffusion Pipeline (`models/diffuser_lightning.py`)

The `DDPMLightning` class implements the complete diffusion training and inference pipeline:
- **Training**: Adds noise at random timesteps, predicts noise with the decoder
- **Inference**: Iterative denoising from pure noise to grasp poses
- **Conditioning**: Scene point clouds + text embeddings → decoder
- **Guidance**: Supports classifier-free guidance with negative prompts

#### 2. Decoder Architectures

**DiT (Diffusion Transformer)** (`models/decoder/dit.py`):
- Transformer-based architecture inspired by Hunyuan-DiT
- Self-attention over grasp tokens + cross-attention to scene/text
- AdaptiveLayerNorm for time-conditional processing
- See `docs/dit_architecture.md` for detailed architecture diagrams

**UNet** (`models/decoder/unet_new.py`):
- 1D temporal UNet architecture
- Transformer blocks for cross-attention to scene/text
- ResNet blocks with time embeddings

#### 3. Scene Encoders (`models/backbone/`)

- **PointNet2** (`pointnet2.py`): Set abstraction layers for hierarchical point cloud encoding
- **PTv3** (`ptv3/`): Point Transformer V3 with optional Flash Attention
- Both output per-point features used for cross-attention in decoders

#### 4. Text Conditioning (`models/utils/text_encoder.py`)

- CLIP or T5 text encoders for prompt embedding
- MLP projection to model hidden dimension
- Optional text dropout during training for classifier-free guidance
- Negative prompt support for guided generation

#### 5. Data Pipeline (`datasets/`)

- `SceneLeapDataModule`: PyTorch Lightning DataModule
- Supports multiple dataset types: SceneLeap, SceneLeapPlus, SceneLeapPro
- Caching system for preprocessed data (`*_cached.py`)
- On-the-fly point cloud cropping and sampling

### Configuration System (Hydra)

The project uses Hydra for hierarchical configuration management:

- **Main config**: `config/config.yaml` (imports other configs via defaults)
- **Command-line overrides**: Use dot notation, e.g., `model.optimizer.lr=0.001`
- **Configuration groups**: Swap entire config sections, e.g., `model/diffuser/decoder=dit`
- **Variable interpolation**: Use `${var}` to reference other config values

Important config variables:
- `${mode}`: Training mode (e.g., "multi" for multi-grasp)
- `${rot_type}`: Rotation representation (e.g., "rot6d")
- `${use_text_condition}`: Enable/disable text conditioning
- `${use_negative_prompts}`: Enable negative prompts
- `${use_rgb}`: Include RGB features in point cloud
- `${use_object_mask}`: Include object segmentation mask
- `${target_num_grasps}`: Number of grasps per scene

## Distributed Training

The project uses PyTorch Lightning with DDP (Distributed Data Parallel):

- **Auto-scaling**: Batch size and learning rate automatically adjust based on GPU count
- **Synchronization**: BatchNorm syncing, gradient syncing handled by Lightning
- **Port management**: Automatic port detection to avoid conflicts
- **Multi-node**: Supports SLURM and manual multi-node setups

Key distributed parameters in config:
```yaml
distributed:
  enabled: true
  devices: 4
  num_nodes: 1
  strategy: ddp
  lr_scaling: sqrt  # How to scale LR with GPU count
```

## Experiment Management

Each training run creates a structured experiment directory:

```
experiments/my_exp/
├── checkpoints/          # Model checkpoints
│   ├── epoch=X-val_loss=Y.ckpt
│   └── last.ckpt
├── config/               # Saved configuration
│   └── whole_config.yaml
├── lightning_logs/       # TensorBoard logs
├── test_results/         # Test outputs (per checkpoint)
│   └── epoch=X-val_loss=Y/
│       ├── test_results.json
│       └── config/
└── backups/              # Code snapshots
```

**Important**: The saved `whole_config.yaml` is used when resuming training or testing to ensure reproducibility.

## Loss Functions and Metrics

### Training Losses (`models/loss/grasp_loss_pose.py`)

The system computes multiple loss components:
- **Translation loss**: L1 or L2 distance between predicted and GT grasp positions
- **Rotation loss**: Geodesic distance on SO(3) or rotation matrix Frobenius norm
- **Joint angle loss**: For hand articulation (if applicable)
- **Q1 loss**: Quality metric-based loss

### Evaluation Metrics

During testing, the following metrics are computed per grasp:
- **Q1**: Grasp quality score (higher is better)
- **Penetration (pen)**: Object penetration depth (lower is better)
- **Valid Q1**: Q1 for grasps with penetration below threshold

Test results include:
- Overall statistics (mean, std, min, max)
- Success rate (% of grasps below penetration threshold)
- Best-grasp-per-scene statistics

## WandB Logging

Weights & Biases integration is configured via `config/wandb/default.yaml`:

```yaml
wandb:
  enabled: true
  project: "scene-leap-plus-diffusion-grasp"
  name: null  # Auto-generated if null
  tags: []
  save_model: false

  optimization:
    enable_visualization: false  # Disable to save bandwidth
    log_histograms: false        # Disable to save bandwidth
    visualization_freq: 20
    histogram_freq: 50
```

Logged metrics are organized with prefixes:
- `train/*`: Training losses and learning rate
- `val/*`: Validation metrics
- `system/*`: GPU memory, utilization (if enabled)

## Model Selection

### When to use DiT vs UNet

**DiT (Diffusion Transformer)**:
- Better for long-range dependencies and multi-grasp scenarios
- More parameter-efficient with large sequence lengths
- Requires more memory per forward pass
- Config: `model/diffuser/decoder=dit`

**UNet**:
- Faster inference and training
- Lower memory footprint
- Good for single or few grasps
- Config: `model/diffuser/decoder=unet`

### Backbone Selection

**PointNet2**:
- Lightweight, faster training
- Good for scenes with moderate complexity
- Config: `model/diffuser/decoder/backbone=pointnet2`

**PTv3 (Point Transformer V3)**:
- State-of-the-art point cloud encoding
- Better scene understanding but slower
- Optional Flash Attention for speed
- Config: `model/diffuser/decoder/backbone=ptv3`

## Common Development Patterns

### Adding a New Model Component

1. Implement the module in the appropriate directory (`models/decoder/`, `models/backbone/`, etc.)
2. Add a build function or register it in `models/decoder/__init__.py`
3. Create a YAML config file in `config/model/diffuser/decoder/my_component.yaml`
4. Test with: `python train_lightning.py model/diffuser/decoder=my_component`

### Modifying the Diffusion Process

Core diffusion logic is split across:
- `models/utils/diffusion_core.py`: Mixin with forward/reverse diffusion methods
- `models/utils/diffusion_utils.py`: Noise schedule creation
- `models/diffuser_lightning.py`: Lightning integration

### Custom Loss Functions

1. Implement in `models/loss/` following the pattern in `grasp_loss_pose.py`
2. Update `config/model/diffuser/criterion/loss.yaml` with new loss weights
3. Register in the criterion builder if needed

## Important Implementation Details

### Rotation Representations

The system supports multiple rotation parameterizations via `rot_type`:
- `"rot6d"`: 6D continuous rotation (recommended, from Zhou et al.)
- `"quat"`: Quaternions
- `"euler"`: Euler angles (can have discontinuities)
- `"rotmat"`: 3×3 rotation matrices (9D)

Conversion utilities are in `utils/rot6d.py`.

### Hand Model Integration

The project includes a detailed hand model (`utils/hand_model.py`) for:
- Forward kinematics (joint angles → fingertip positions)
- Grasp feasibility checking
- Penetration and quality computation

Hand pose processing utilities (`utils/hand_helper.py`):
- `process_hand_pose()`: Normalize poses for training
- `denorm_hand_pose_robust()`: Denormalize predicted poses
- `process_hand_pose_test()`: Prepare poses for evaluation

### Grasp Sampling Strategies

Configured via `grasp_sampling_strategy` in data config:
- `"farthest_point"`: Maximize diversity in grasp positions
- `"random"`: Uniform random sampling
- `"exhaustive"`: Use all available grasps (controlled by `max_grasps_per_object`)

## Cursor Rules Integration

From `.cursor/rules/scene-leap-ultra.mdc`:
- Always activate the DexGrasp conda environment before running tests or training

## Debugging Tips

### Check GPU Memory Issues
- Reduce `batch_size` or `max_points` in data config
- Enable gradient checkpointing: `model.decoder.gradient_checkpointing=true`
- Reduce model size: `model.decoder.d_model=256` or `model.decoder.num_layers=6`

### Distributed Training Failures
- Check port availability (auto-detected but can conflict)
- Verify `CUDA_VISIBLE_DEVICES` is not set (script handles it)
- Ensure all nodes can communicate (multi-node only)

### Config Override Not Working
- Use `+key=value` to add new keys not in base config
- Use `++key=value` to force override even if key doesn't exist
- Check Hydra output logs for "Overriding" messages

### Checkpoint Loading Errors
- Ensure config compatibility (major architecture changes break checkpoints)
- Use `resume=true` for continuing training (loads optimizer state)
- Use `checkpoint_path` alone for fine-tuning (fresh optimizer)
