import importlib
from typing import List, Tuple, Union


def instantiate(name: str):
    """
    Import an object by its string path and instantiate/call it.

    Args:
        name: Dotted path string (e.g., 'torch.nn.GELU').

    Returns:
        The instantiated object.
    """
    try:
        module_path, attr_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unknown import target: '{name}'") from e

    try:
        return attr()
    except TypeError as e:
        raise ValueError(f"Cannot instantiate '{name}' without arguments") from e


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
