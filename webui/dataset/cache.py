"""
Preview rollout collection and disk caching utilities for the WebUI Dataset Inspector.
"""

from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate

from webui.dataset.action_sweep import SWEEP_NUM_STEPS, collect_action_sweep
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.pipeline.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache" / "webui"

NUM_EPISODES = 10
MAX_STEPS = 10
MIN_STEPS = 5
SAMPLER_NUM_STEPS = 5
IMAGE_SIZE = None


def get_config_files() -> list[str]:
    """List all dataset config YAML files in ``configs/dataset``."""
    return [str(f) for f in sorted((PROJECT_ROOT / "configs" / "dataset").glob("*.yaml"))]


def get_buffer_cache_path(config_path: str) -> Path:
    """Return the cached ReplayBuffer file path for a given dataset config."""
    stem = Path(config_path).stem.lower()
    return CACHE_DIR / f"{stem}_buffer.pt"


def get_sweep_cache_path(config_path: str) -> Path:
    """Return the cached single-action sweep file path for a given dataset config."""
    stem = Path(config_path).stem.lower()
    return CACHE_DIR / f"{stem}_action_sweep.pt"


def apply_ui_overrides(config: Any) -> Any:
    """Apply fast WebUI preview overrides (10 episodes, 10 max steps, 5 min steps, native image_size)."""
    config.data_module.collector.num_episodes = NUM_EPISODES
    config.data_module.collector.max_steps = MAX_STEPS
    config.data_module.collector.min_steps = MIN_STEPS
    config.data_module.collector.env.image_size = IMAGE_SIZE
    config.data_module.sampler.num_steps = SAMPLER_NUM_STEPS
    config.data_module.num_workers = 0
    return config


def _save_action_sweep(env: TorchGymEnv, config_path: str, sweep_path: Path, seed: int) -> None:
    """Collect single-action sweep episodes for ``env`` and persist them to ``sweep_path``."""
    print(f"[prepare_ui] Collecting action sweep for {Path(config_path).name}...")
    episodes = collect_action_sweep(env, num_steps=SWEEP_NUM_STEPS, seed=seed)
    torch.save(episodes, sweep_path)
    print(f"[prepare_ui] Saved {len(episodes)} action sweep episodes to {sweep_path}")


def prepare_config(config_path: str, force: bool = False) -> tuple[Path, Path]:
    """
    Collect preview dataset rollouts and single-action sweep episodes for ``config_path``
    and save them under ``CACHE_DIR``.
    """
    buffer_path = get_buffer_cache_path(config_path)
    sweep_path = get_sweep_cache_path(config_path)
    need_buffer = force or not buffer_path.exists()
    need_sweep = force or not sweep_path.exists()

    if not need_buffer and not need_sweep:
        print(f"[prepare_ui] Using cached preview data for {Path(config_path).name}")
        return buffer_path, sweep_path

    config = apply_ui_overrides(load_config(config_path))
    seed = int(config.get("seed", 0))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if need_buffer:
        print(f"[prepare_ui] Collecting preview rollouts for {Path(config_path).name}...")
        data_module = instantiate(config.data_module, _convert_="all")
        try:
            data_module.setup("fit")
            torch.save(data_module.replay_buffer, buffer_path)
            print(f"[prepare_ui] Saved {len(data_module.replay_buffer)} episodes to {buffer_path}")

            if need_sweep:
                _save_action_sweep(data_module.collector.env, config_path, sweep_path, seed)
        finally:
            data_module.collector.close()
    elif need_sweep:
        with instantiate(config.data_module.collector.env, _convert_="all") as env:
            _save_action_sweep(env, config_path, sweep_path, seed)

    return buffer_path, sweep_path
