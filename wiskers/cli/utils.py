import glob
import os
from typing import List

import omegaconf
from omegaconf import DictConfig, OmegaConf


def _resolve_config_path(base_file: str, path: str) -> str:
    """
    Resolve a config file path either relative to the base file or the current working directory.
    """
    if os.path.isabs(path):
        return path
    candidate_rel = os.path.normpath(os.path.join(os.path.dirname(base_file), path))
    if os.path.exists(candidate_rel):
        return candidate_rel
    if os.path.exists(path):
        return path
    return candidate_rel


def load_config(config_file: str) -> DictConfig:
    """
    Load a configuration file, resolving any 'defaults' inclusions.
    """
    try:
        raw_cfg = omegaconf.OmegaConf.load(config_file)
    except Exception as e:
        raise RuntimeError(f"Error reading the config file '{config_file}': {e}")

    if not isinstance(raw_cfg, DictConfig) or "defaults" not in raw_cfg:
        return raw_cfg

    defaults = raw_cfg.pop("defaults")
    if isinstance(defaults, str):
        defaults = [defaults]

    configs_to_merge: List[DictConfig] = []
    self_merged = False

    for item in defaults:
        if item == "_self_":
            configs_to_merge.append(raw_cfg)
            self_merged = True
        elif isinstance(item, str):
            resolved = _resolve_config_path(config_file, item)
            base_cfg = load_config(resolved)
            configs_to_merge.append(base_cfg)

    if not self_merged:
        configs_to_merge.append(raw_cfg)

    return OmegaConf.merge(*configs_to_merge)


def get_latest_checkpoint(experiment_dir: str, run_name: str, ext: str = "ckpt") -> str:
    checkpoint_dir = os.path.join(experiment_dir, run_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"{checkpoint_dir} doesn't exist")

    files = glob.glob(os.path.join(checkpoint_dir, f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"No .{ext} files found in {checkpoint_dir}")

    return max(files, key=os.path.getmtime)  # most recent
