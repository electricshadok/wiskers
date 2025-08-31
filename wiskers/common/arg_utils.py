from typing import Callable, Dict, List, Tuple, Union

import torch.nn as nn


_ACTIVATION_MAP: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def get_activation(name: str) -> nn.Module:
    """Returns an instantiated activation function by name."""
    name = name.lower()
    if name not in _ACTIVATION_MAP:
        raise ValueError(f"Unknown activation function: '{name}'")
    return _ACTIVATION_MAP[name]()


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
