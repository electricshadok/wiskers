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
        print(f"Error reading the config file: {e}")
        return None
