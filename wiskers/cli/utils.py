import glob
import os

import omegaconf
from omegaconf import DictConfig


def load_config(config_file: str) -> DictConfig:
    """
    Load a configuration files
    """
    try:
        config = omegaconf.OmegaConf.load(config_file)
        return config
    except Exception as e:
        raise RuntimeError(f"Error reading the config file '{config_file}': {e}")


def get_latest_checkpoint(experiment_dir: str, run_name: str, ext: str = "ckpt") -> str:
    checkpoint_dir = os.path.join(experiment_dir, run_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"{checkpoint_dir} doesn't exist")

    files = glob.glob(os.path.join(checkpoint_dir, f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"No .{ext} files found in {checkpoint_dir}")

    return max(files, key=os.path.getmtime)  # most recent
