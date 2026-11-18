"""
Shared utilities for environment adapters.
"""

import numpy as np
import torch
import torch.nn.functional as F


def render_to_tensor(
    frame: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Convert a raw NumPy image frame into a contiguous ``torch.uint8`` tensor.

    Args:
        frame (np.ndarray): Input image array with shape ``(H, W, C)`` (e.g. RGB)
            or ``(H, W)`` (grayscale). Non-contiguous strides (such as flipped
            OpenGL views) are copied cleanly.
        image_size (tuple[int, int] | None): Optional target ``(height, width)`` to
            downsample/resize frames to via area interpolation. If ``None``,
            preserves the native resolution.

    Returns:
        torch.Tensor: ``uint8`` tensor with shape ``(H, W, C)`` or ``(H, W)``
        (or ``(image_size[0], image_size[1], C)`` / ``image_size`` when resized).
    """
    obs = torch.from_numpy(np.ascontiguousarray(frame))
    if image_size is not None and obs.ndim >= 2 and obs.shape[:2] != image_size:
        if obs.ndim == 3:
            t = obs.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(t, size=image_size, mode="area")
            obs = resized.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        elif obs.ndim == 2:
            t = obs.unsqueeze(0).unsqueeze(0).float()
            resized = F.interpolate(t, size=image_size, mode="area")
            obs = resized.squeeze(0).squeeze(0).to(torch.uint8)
    return obs
