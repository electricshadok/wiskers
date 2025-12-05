import importlib
from typing import Any, Dict, List, Tuple, Union

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig


def instantiate(target: Union[str, Dict[str, Any], DictConfig], **kwargs):
    """
    Instantiate an object either from a dotted string path or a Hydra config dict.

    Args:
        target: Dotted path string (e.g., 'torch.nn.GELU') or Hydra config
            mapping (dict/DictConfig).
        **kwargs: Optional keyword arguments forwarded to the instantiation call.

    Returns:
        The instantiated object.
    """
    if isinstance(target, str):
        try:
            module_path, attr_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            attr = getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown import target: '{target}'") from e

        try:
            return attr(**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Cannot instantiate '{target}' with provided arguments"
            ) from e

    if isinstance(target, (dict, DictConfig)):
        return hydra_instantiate(target, **kwargs)

    raise TypeError("instantiate expects a string import path or a Hydra config dict.")


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
