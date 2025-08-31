from typing import List, Tuple, Union


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
