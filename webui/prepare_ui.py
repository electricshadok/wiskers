"""
Pre-collect short rollout trajectories on the main thread for the WebUI.

Usage:
    python webui/prepare_ui.py
    python webui/prepare_ui.py --config configs/dataset/franka_kitchen.yaml --force
"""

import argparse
import sys
from pathlib import Path


# Ensure project root is on sys.path when invoked as `python webui/prepare_ui.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webui.dataset.cache import get_config_files, prepare_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-collect rollout trajectories for the WebUI Dataset Inspector.")
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
        prepare_config(cfg, force=args.force)


if __name__ == "__main__":
    main()
