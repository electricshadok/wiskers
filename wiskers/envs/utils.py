"""
Shared utilities for environment adapters.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


class ResizeUint8:
    """
    Resize an image tensor ``(H, W, C)`` or ``(H, W)`` to ``size`` ``(height, width)``
    via area interpolation and convert to ``torch.uint8``.

    Args:
        size: Target ``(height, width)`` spatial resolution.
    """

    def __init__(self, size: Sequence[int]) -> None:
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"size must be a 2-element sequence of positive ints (H, W), got {size}.")
        self.size: tuple[int, int] = (int(size[0]), int(size[1]))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim >= 2 and image.shape[:2] != self.size:
            if image.ndim == 3:
                t = image.permute(2, 0, 1).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.size, mode="area")
                return resized.squeeze(0).permute(1, 2, 0).to(torch.uint8)
            if image.ndim == 2:
                t = image.unsqueeze(0).unsqueeze(0).float()
                resized = F.interpolate(t, size=self.size, mode="area")
                return resized.squeeze(0).squeeze(0).to(torch.uint8)
        return image.to(torch.uint8)


def patch_mujoco_joint_eq() -> None:
    """Allow ``mujoco.mjtJoint`` enum values to compare equal with Python/NumPy ints."""
    import mujoco

    mujoco.mjtJoint.__eq__ = lambda self, other: (
        isinstance(other, (int, np.integer, mujoco.mjtJoint)) and int(self) == int(other)
    )
