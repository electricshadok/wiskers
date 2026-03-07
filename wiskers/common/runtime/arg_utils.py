from typing import Any, Dict, Union

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig


def instantiate(target: Union[str, Dict[str, Any], DictConfig, Any], **kwargs):
    """
    Instantiate an object using Hydra. Strings are converted into a Hydra dict.
    If target is already an instantiated object (not a string or dict-like config),
    it is returned as-is.

    Args:
        target: Dotted path string (e.g., 'torch.nn.GELU'), Hydra config mapping
            (dict/DictConfig), or an already instantiated object.
        **kwargs: Optional keyword arguments forwarded to the instantiation call.

    Returns:
        The instantiated object.
    """
    if isinstance(target, str):
        target = {"_target_": target}

    if isinstance(target, (dict, DictConfig)):
        return hydra_instantiate(target, **kwargs)

    return target


