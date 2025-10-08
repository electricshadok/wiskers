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
