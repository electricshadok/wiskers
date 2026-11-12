import glob
import os
from typing import Dict, List, Optional

import omegaconf
from omegaconf import DictConfig, OmegaConf


def resolve_config_file(config_name: str, config_path: str = "./configs") -> str:
    """
    Resolve a config file given config_name and config_path root.

    Examples:
        resolve_config_file("training/carracing_world_models", "./configs")
        -> "configs/training/carracing_world_models.yaml"

        resolve_config_file("./configs/training/carracing_world_models.yaml")
        -> "configs/training/carracing_world_models.yaml"
    """
    if os.path.exists(config_name):
        return os.path.normpath(config_name)

    for ext in ["", ".yaml", ".yml"]:
        candidate = os.path.normpath(os.path.join(config_path, f"{config_name}{ext}"))
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find config '{config_name}' in config root '{config_path}'"
    )


def resolve_group_path(config_path: str, group: str, name: str) -> str:
    """
    Resolve a Hydra config group path: <config_path>/<group>/<name>.yaml.
    """
    for ext in [".yaml", ".yml"]:
        candidate = os.path.normpath(os.path.join(config_path, group, f"{name}{ext}"))
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find config for group '{group}' with name '{name}' under '{config_path}/{group}/'"
    )


def load_config(
    config_name: str,
    config_path: str = "./configs",
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load a configuration file, resolving Hydra config groups and CLI overrides.

    Args:
        config_name (str): Configuration name (e.g. 'training/carracing_world_models')
                           or path to YAML file.
        config_path (str): Root directory for configs (default: 'configs').
        overrides (Optional[List[str]]): CLI overrides (e.g. ['dataset=breakout', 'trainer.max_epochs=10']).

    Returns:
        DictConfig: Merged OmegaConf configuration.
    """
    config_file = resolve_config_file(config_name, config_path)
    try:
        raw_cfg = omegaconf.OmegaConf.load(config_file)
    except Exception as e:
        raise RuntimeError(f"Error reading config file '{config_file}': {e}")

    # Parse overrides into group overrides (e.g. dataset=breakout) vs dotlist (e.g. trainer.max_epochs=10)
    group_overrides: Dict[str, str] = {}
    dotlist_overrides: List[str] = []

    if overrides:
        for item in overrides:
            if "=" in item:
                k, v = item.split("=", 1)
                k = k.strip()
                v = v.strip()
                if "." not in k:
                    group_overrides[k] = v
                else:
                    dotlist_overrides.append(f"{k}={v}")

    if not isinstance(raw_cfg, DictConfig) or "defaults" not in raw_cfg:
        if dotlist_overrides:
            cli_cfg = OmegaConf.from_dotlist(dotlist_overrides)
            return OmegaConf.merge(raw_cfg, cli_cfg)
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
        elif isinstance(item, (dict, DictConfig)):
            for group, name in item.items():
                group_str = str(group)
                name_str = str(name)
                # Check for CLI group override (e.g. dataset=breakout)
                if group_str in group_overrides:
                    name_str = group_overrides[group_str]
                group_file = resolve_group_path(config_path, group_str, name_str)
                base_cfg = load_config(group_file, config_path=config_path)
                configs_to_merge.append(base_cfg)

    if not self_merged:
        configs_to_merge.append(raw_cfg)

    merged = OmegaConf.merge(*configs_to_merge)

    if dotlist_overrides:
        cli_cfg = OmegaConf.from_dotlist(dotlist_overrides)
        merged = OmegaConf.merge(merged, cli_cfg)

    return merged


def get_latest_checkpoint(experiment_dir: str, run_name: str, ext: str = "ckpt") -> str:
    checkpoint_dir = os.path.join(experiment_dir, run_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"{checkpoint_dir} doesn't exist")

    files = glob.glob(os.path.join(checkpoint_dir, f"*.{ext}"))
    if not files:
        raise FileNotFoundError(f"No .{ext} files found in {checkpoint_dir}")

    return max(files, key=os.path.getmtime)  # most recent
