"""
Pre-collect short rollout trajectories on the main thread for the Streamlit WebUI.

Usage:
    python webui/prepare_ui.py
    python webui/prepare_ui.py --config configs/dataset/franka_kitchen.yaml --force
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List


# Ensure project root is on sys.path when invoked as `python webui/prepare_ui.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from wiskers.cli.utils import load_config  # noqa: E402


NUM_EPISODES = 10
MAX_STEPS = 10
MIN_STEPS = 5
SAMPLER_NUM_STEPS = 5
IMAGE_SIZE = None
CACHE_DIR = PROJECT_ROOT / ".cache" / "webui"


def get_config_files() -> List[str]:
    """List all dataset config YAML files in configs/dataset."""
    config_dir = Path("configs/dataset")
    if not config_dir.is_dir():
        config_dir = PROJECT_ROOT / "configs" / "dataset"

    files = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    return [str(f) for f in files]


def get_cache_path(config_path: str) -> Path:
    """Return the cached ReplayBuffer file path for a given dataset config."""
    stem = Path(config_path).stem.lower()
    return CACHE_DIR / f"{stem}_buffer.pt"


def apply_ui_overrides(config: Any) -> Any:
    """Apply fast WebUI preview overrides (10 episodes, 10 max steps, 5 min steps, native image_size)."""
    config.data_module.collector.num_episodes = NUM_EPISODES
    config.data_module.collector.max_steps = MAX_STEPS
    config.data_module.collector.min_steps = MIN_STEPS
    config.data_module.collector.env.env.image_size = IMAGE_SIZE
    config.data_module.sampler.num_steps = SAMPLER_NUM_STEPS
    return config


def prepare_dataset_cache(config_path: str, force: bool = False) -> Path:
    """
    Collect rollouts on the main thread and save the ReplayBuffer to disk.
    """
    cache_path = get_cache_path(config_path)
    if cache_path.exists() and not force:
        print(f"[prepare_ui] Using cached rollouts for {Path(config_path).name}: {cache_path}")
        return cache_path

    print(f"[prepare_ui] Collecting preview rollouts for {Path(config_path).name}...")
    config = apply_ui_overrides(load_config(config_path))
    data_module = instantiate(config.data_module, _convert_="all")
    data_module.prepare_data()
    data_module.setup("fit")
    data_module.collector.close()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(data_module.replay_buffer, cache_path)
    print(f"[prepare_ui] Saved {len(data_module.replay_buffer)} episodes to {cache_path}")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-collect rollout trajectories for the Streamlit Dataset Inspector."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional single dataset config YAML to prepare (defaults to all in configs/dataset/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-collect rollouts even if a cached buffer already exists.",
    )
    args = parser.parse_args()

    configs = [args.config] if args.config else get_config_files()
    for cfg in configs:
        prepare_dataset_cache(cfg, force=args.force)


if __name__ == "__main__":
    main()
