import os

import omegaconf
from omegaconf import DictConfig


def load_config(config_file: str, template_config_dir: str) -> DictConfig:
    """
    Load and merge a configuration file with a template configuration.
    The template configuration file should have the same name as the Python script calling this function
    """
    template_config_file = os.path.join(template_config_dir, os.path.basename(config_file))

    if not os.path.exists(template_config_file):
        raise FileNotFoundError(f"{template_config_file} doesn't exist")

    try:
        template_config = omegaconf.OmegaConf.load(template_config_file)
        config = omegaconf.OmegaConf.load(config_file)
        config = omegaconf.OmegaConf.merge(template_config, config)
        return config
    except Exception as e:
        print(f"Error reading the config file: {e}")
        return None
