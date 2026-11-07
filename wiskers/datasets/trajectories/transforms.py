"""
Episode-to-tensor transforms for trajectory datasets.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from wiskers.datasets.trajectories.episode import Episode


class EpisodeTransform(ABC):
    """
    Abstract base class for transforms that convert an Episode chunk into a dict of PyTorch tensors.
    """

    @abstractmethod
    def __call__(self, episode: Episode) -> dict[str, torch.Tensor]:
        """
        Convert one Episode chunk into a dictionary of PyTorch tensors.

        Args:
            episode (Episode): Trajectory chunk of length ``T``.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping field names to tensors with
            leading time dimension ``T`` (collated by DataLoader into ``(B, T, ...)``).
        """
        ...


class EpisodeToTensor(EpisodeTransform):
    """
    Convert an Episode chunk (numpy arrays) into a dictionary of PyTorch tensors,
    normalizing observations to float32 in ``obs_range`` and converting to target
    ``image_size`` ``(C, H, W)``.

    Args:
        image_size: Target 3-tuple ``(C, H, W)`` where ``C`` is 1 (grayscale) or 3 (RGB),
                    and ``(H, W)`` is the spatial resolution.
        obs_range: Target 2-tuple ``(low, high)`` specifying the output normalization
                   range for observations (e.g., ``(0.0, 1.0)`` or ``(-1.0, 1.0)``).
    """

    def __init__(self, image_size: Sequence[int], obs_range: Sequence[float]) -> None:
        if len(image_size) != 3:
            raise ValueError(
                f"image_size must be a 3-element sequence (C, H, W), got {image_size}."
            )
        c, h, w = int(image_size[0]), int(image_size[1]), int(image_size[2])
        if c not in (1, 3):
            raise ValueError(f"image_size channel count C must be 1 or 3, got {c}.")
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size spatial dimensions (H, W) must be positive, got ({h}, {w}).")

        if len(obs_range) != 2:
            raise ValueError(
                f"obs_range must be a 2-element sequence (low, high), got {obs_range}."
            )
        low, high = float(obs_range[0]), float(obs_range[1])
        if low >= high:
            raise ValueError(
                f"obs_range must satisfy low < high, got ({low}, {high})."
            )

        self.image_size: tuple[int, int, int] = (c, h, w)
        self.obs_range: tuple[float, float] = (low, high)

    def __call__(self, episode: Episode) -> dict[str, torch.Tensor]:
        """
        Convert one Episode chunk into a dictionary of float32 PyTorch tensors.

        Args:
            episode (Episode): Trajectory chunk of ``T`` transitions containing:
                - ``obs``: ``np.ndarray`` of shape ``(T + 1, H_in, W_in, C_in)``, dtype ``uint8``
                - ``actions``: ``np.ndarray`` of shape ``(T, action_dim)``, dtype ``float32``
                - ``rewards``: ``np.ndarray`` of shape ``(T,)``, dtype ``float32``
                - ``dones``: ``np.ndarray`` of shape ``(T,)``, dtype ``bool``

        Returns:
            dict[str, torch.Tensor]: Dictionary of tensors (collated by DataLoader into ``(B, ...)``):
                - ``"obs"``: ``torch.Tensor`` of shape ``(T + 1, C, H, W)``, dtype ``torch.float32`` in ``obs_range``
                - ``"actions"``: ``torch.Tensor`` of shape ``(T, action_dim)``, dtype ``torch.float32``
                - ``"rewards"``: ``torch.Tensor`` of shape ``(T,)``, dtype ``torch.float32``
                - ``"dones"``: ``torch.Tensor`` of shape ``(T,)``, dtype ``torch.float32`` (``0.0`` or ``1.0``)
        """
        obs = torch.from_numpy(episode.obs)
        if obs.ndim == 4:
            # (T, H, W, C) -> (T, C, H, W)
            obs = obs.permute(0, 3, 1, 2)
        elif obs.ndim == 3:
            # (T, H, W) -> (T, 1, H, W)
            obs = obs.unsqueeze(1)

        obs = obs.float() / 255.0

        target_c, target_h, target_w = self.image_size
        in_c, in_h, in_w = obs.shape[1], obs.shape[2], obs.shape[3]

        if target_c == 1 and in_c == 3:
            obs = TF.rgb_to_grayscale(obs, num_output_channels=1)
        elif target_c == 3 and in_c == 1:
            obs = obs.repeat(1, 3, 1, 1)

        if (in_h, in_w) != (target_h, target_w):
            obs = F.interpolate(
                obs,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        low, high = self.obs_range
        if (low, high) != (0.0, 1.0):
            obs = obs * (high - low) + low

        return {
            "obs": obs,
            "actions": torch.from_numpy(episode.actions),
            "rewards": torch.from_numpy(episode.rewards),
            "dones": torch.from_numpy(episode.dones.astype(np.float32)),
        }
