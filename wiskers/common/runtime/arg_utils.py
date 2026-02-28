from typing import Any, Dict, List, Tuple, Union

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


def format_image_size(
    image_size: Union[int, Tuple[int, int], List[int]],
) -> Tuple[int, int]:
    """
    Converts an image size input into a standardized (H, W) tuple.

    Args:
        image_size (int, tuple, or list): The input image size. Can be:
            - a single int (assumes square image),
            - a tuple or list of two ints (height, width).

    Returns:
        tuple: A (height, width) tuple representing image dimensions.
    """
    if isinstance(image_size, int):
        return (image_size, image_size)
    elif isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError("image_size must be an int or a sequence of two integers.")
        return tuple(image_size)
    else:
        raise TypeError("image_size must be an int, tuple, or list of two integers.")
