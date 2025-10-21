使用简体中文回答

# Repository Guidelines

## Project Structure & Module Organization
Entry scripts (`train_lightning.py`, `train_distributed.py`, `test_lightning.py`) live in the repo root. Models and diffusion modules sit under `models/`, data modules in `datasets/`, and shared helpers in `utils/`. Hydra configs are grouped in `config/` with subfolders for `model/`, `data_cfg/`, and `distributed/`. Generated artifacts (`experiments/`, `lightning_logs/`, `outputs/`) should stay untracked or Git LFS backed; notebooks and reports live in `notebooks/` and `docs/`.

## Build, Test, and Development Commands
Run single-GPU training with `python train_lightning.py`, applying overrides like `python train_lightning.py model.optimizer.lr=1e-4 data.train.batch_size=32`. For multi-GPU jobs use `bash train_distributed.sh --gpus 4 save_root=./experiments/run_xyz`. Batch experiment presets are scripted in `bash run.sh`. Evaluate a checkpoint via `python test_lightning.py +checkpoint_path=experiments/<run>/checkpoints/epoch=XXX.ckpt`. Dataset visualizers in `tests/` and `scripts/` are invoked as standalone Python modules when inspecting samples.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation, `snake_case` functions, `PascalCase` classes. Prefer explicit type hints and keep Lightning modules modular by moving utilities into `utils/`. Configuration keys mirror their directories; keep lowercase with underscores (`data_cfg.sceneleappro`). Format code with `python -m black .` and `python -m isort .` before sending a PR, and lint imports when touching shared utilities.

## Testing Guidelines
PyTorch Lightning smoke tests live in `test_lightning.py`; gate long runs behind smaller batch sizes and Hydra overrides such as `data.val.limit_batches=2`. Add unit tests with `pytest` under `tests/`, naming files `test_<feature>.py` and functions `test_<behavior>`. Ensure metric aggregation writes to `experiments/<run>/test_results.json` and confirm W&B logging still succeeds when callback logic changes.

## Commit & Pull Request Guidelines
Commits use imperative summaries (e.g., `Refactor configuration files`) capped at 72 characters, followed by an empty line and optional bullet points. Reference configs or scripts touched, and include Hydra overrides used for reproducing results. Pull requests should link issues when relevant, summarize model or data impacts, and paste key metrics or screenshots. Request review only after local `pytest` and targeted Lightning smoke tests pass.
