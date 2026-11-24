# Wiskers Project Guidelines

## Package Structure & Imports
- **Keep `__init__.py` empty**: Do not add re-exports, side effects, or `__all__` to `__init__.py` files (especially across `wiskers/nn/`, `wiskers/systems/`, and any new packages). Keeping `__init__.py` empty avoids circular imports, symbol collisions, `__all__` maintenance, and eager imports of heavy optional dependencies while keeping a 1-to-1 mapping from import paths to files.
- **Use explicit submodule imports**: Import from the defining file path (e.g., `from wiskers.nn.conv.bam import BAM`, `from wiskers.systems.world_models import WorldModels`) and use full module `_target_` paths in Hydra configs under `configs/`.

## Repository & `wiskers/` Architecture
- **`wiskers/controller.py` & `wiskers/metrics.py`**: Core protocols and standalone utilities — `Controller` protocol (`action_spec: ActionSpec`, `obs: TensorDict, step -> action Tensor`) + `validate_controller(controller, action_spec)` for policies/planners, and VQ metrics (`codebook_usage_metrics`).
- **`wiskers/envs/`**: Torch-first Gymnasium environment wrappers (`GymEnv` in `gym_env.py`, `MujocoEnv` in `mujoco_env.py`, `ActionSpec` in `action_spec.py`, `render_to_tensor` in `utils.py`) and per-environment packages (`adroit_hand/`, `breakout/`, `car_racing/`, `franka_kitchen/`, `inverted_pendulum/`, `shadow_hand/`) with env implementations + env-specific policies. Environments emit flat `TensorDict` observations (e.g., `"render"`, `"ram"`, `"state:*"`).
- **`wiskers/policy/`**: Base rollout policies (`Policy`, `DiscretePolicy`, `ContinuousPolicy`, `RandomPolicy` in `base.py`) parameterized by `action_spec: ActionSpec`, mapping `obs: TensorDict` to 1D action `torch.Tensor`s and conforming to `Controller`.
- **`wiskers/planner/`**: Model-based receding-horizon planners (`Planner`, `DynamicsModel` in `base.py`, `RandomShootingPlanner` in `random_shooting.py`) conforming to `Controller` for MPC planning in imagination.
- **`wiskers/data/`**: Trajectory storage and dataloading pipeline:
  - `episode.py`: `Episode(TensorDict)` schema (`obs`, `action`, `reward`, `done`).
  - `buffer.py`: `Buffer`, `ReplayBuffer`, `RolloutBuffer`.
  - `sampler.py`: Chunk sampling strategies (`TrajectorySampler`, `UniformChunkSampler`, `RecencyWeightedSampler`).
  - `transforms.py`: Episode preprocessing (`EpisodeTransform`, `EpisodeToTensor`).
  - `collector.py`: `Collector` rolling out a `GymEnv` + `Controller` into a `Buffer`.
  - `dataset.py` & `datamodule.py`: PyTorch `TrajectoryDataset` (`IterableDataset`) and Lightning `TrajectoryDataModule`.
- **`wiskers/nn/` (Pure PyTorch only)**:
  - Composable `torch.nn.Module` building blocks (`attention/`, `conv/`, `heads.py`, `losses.py`, `norm.py`, `quantizer.py`) and paper-specific world-model modules (`wm/planet/`, `wm/world_models/`).
  - Never import `lightning`, optimizers, or training-loop logic inside `wiskers/nn/`.
- **`wiskers/systems/` (PyTorch Lightning)**:
  - `lightning.LightningModule` training systems (`planet.py`, `world_models.py`) that assemble components from `wiskers/nn/`.
- **`wiskers/pipeline/`**: Hydra/OmegaConf config loading (`config.py`), training/evaluation/dataset CLIs (`train.py`, `predict.py`, `inspect_dataset.py`), and export/inference backends (`backends.py`: ONNX, SafeTensors, Checkpoint).
- **Supporting root folders**:
  - `configs/`: Hydra/OmegaConf YAML configs (`dataset/`, `module/`, `training/`).
  - `tests/`: Unit/integration tests mirroring the `wiskers/` directory layout.
  - `webui/`: Streamlit interactive visualization app for inspecting datasets and rollouts.

## Coding Best Practices
- **Avoid mutable default arguments**: Never use mutable objects (`list`, `dict`, or instantiated `nn.Module` objects like `nn.ReLU()`) as default argument values — Python evaluates defaults once at import time and shares them across instances.
  - Use immutable defaults for sequence parameters (e.g., `block_channels: Sequence[int] = (32, 64, 128)`).
  - For configurable activations or repeated submodules, pass a class/factory (`activation: Callable[[], nn.Module] = nn.ReLU`) and call `activation()` inside `__init__` so every layer owns a fresh instance; show class examples without `()` in docstrings (e.g., `nn.ReLU`, `nn.GELU`).
  - When a single optional module instance is expected, default to `None` and instantiate inside `__init__` (e.g., `reconstruction: nn.Module | None = None`).
- **No backward-compatibility shims**: Prefer clean, direct refactors over keeping legacy aliases, unused wrappers, or fallback paths.
- **Close environments explicitly**: Release Gymnasium/MuJoCo environments via `with env:` or `try ... finally: collector.close()`, and monkeypatch `MujocoRenderer.render` in headless MuJoCo unit tests.

## Tooling & Verification
- Environment & dependencies are managed with `uv` (Python 3.13).
- Formatting/linting follows `pyproject.toml` (Ruff: line length 120, double quotes, 2 blank lines after imports) and static typing follows `pyrefly.toml` (`Pyrefly`).
- Do not write unit tests for `webui/`.
- After code changes, run:
  ```bash
  uv run --no-sync ruff check .
  uv run --no-sync pyrefly check
  uv run --no-sync pytest
  ```

## Git Hygiene
- Do not stage or commit `TODO.md` unless explicitly asked.
