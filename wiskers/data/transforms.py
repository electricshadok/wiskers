"""
Episode-to-tensor transforms for trajectory datasets.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tensordict import TensorDict

from wiskers.data.episode import Episode


class EpisodeTransform(ABC):
    """
    Abstract base class for transforms that convert an Episode chunk into a dict of PyTorch tensors.
    """

    @abstractmethod
    def transform_obs(self, obs: TensorDict) -> TensorDict:
        """
        Transform a raw observation ``TensorDict`` (single step ``batch_size=[]`` or sequence
        ``batch_size=[T]``) into a normalized ``TensorDict`` ready for models/controllers.
        """
        ...

    @abstractmethod
    def __call__(self, episode: Episode) -> dict[str, TensorDict | torch.Tensor]:
        """
        Convert one Episode chunk into a dictionary of PyTorch tensors / TensorDicts.

        Args:
            episode (Episode): Trajectory chunk of length ``T``.

        Returns:
            dict[str, TensorDict | torch.Tensor]: Dictionary mapping field names to tensors or
            TensorDicts with leading time dimension ``T`` (collated by DataLoader into ``(B, T, ...)``).
        """
        ...


class EpisodeToTensor(EpisodeTransform):
    """
    Convert an Episode chunk into a dictionary of normalized PyTorch tensors,
    normalizing observations to float32 in ``obs_range`` and converting to target
    ``image_size`` ``(C, H, W)``.

    Args:
        image_size: Target 3-tuple ``(C, H, W)`` where ``C`` is 1 (grayscale) or 3 (RGB),
                    and ``(H, W)`` is the spatial resolution.
        obs_range: Target 2-tuple ``(low, high)`` specifying the output normalization
                   range for observations (e.g., ``(0.0, 1.0)`` or ``(-1.0, 1.0)``).
    """

    def __init__(
        self,
        image_size: Sequence[int],
        obs_range: Sequence[float],
        obs_key: str = "render",
    ) -> None:
        if len(image_size) != 3:
            raise ValueError(f"image_size must be a 3-element sequence (C, H, W), got {image_size}.")
        c, h, w = image_size[0], image_size[1], image_size[2]
        if c not in (1, 3):
            raise ValueError(f"image_size channel count C must be 1 or 3, got {c}.")
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size spatial dimensions (H, W) must be positive, got ({h}, {w}).")

        if len(obs_range) != 2:
            raise ValueError(f"obs_range must be a 2-element sequence (low, high), got {obs_range}.")
        low, high = obs_range[0], obs_range[1]
        if low >= high:
            raise ValueError(f"obs_range must satisfy low < high, got ({low}, {high}).")

        self.image_size: tuple[int, int, int] = (c, h, w)
        self.obs_range: tuple[float, float] = (low, high)
        self.obs_key = obs_key

    def transform_obs(self, obs: TensorDict) -> TensorDict:
        """
        Convert raw uint8 image observations under ``self.obs_key`` into normalized
        ``float32`` tensors in ``obs_range`` while preserving all other observation keys.
        If ``self.obs_key`` is absent from ``obs``, returns ``obs`` unchanged.

        Supports both a single observation ``batch_size=[]`` (``(H, W, C) -> (C, H, W)``)
        and a sequence ``batch_size=[T]`` (``(T, H, W, C) -> (T, C, H, W)``).
        """
        if self.obs_key not in obs.keys():
            return obs
        img = obs[self.obs_key]
        single_frame = img.ndim == 3
        obs_tensor = img.unsqueeze(0) if single_frame else img

        # (T, H, W, C) -> (T, C, H, W)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2).float() / 255.0

        target_c, target_h, target_w = self.image_size
        in_c, in_h, in_w = obs_tensor.shape[1], obs_tensor.shape[2], obs_tensor.shape[3]

        if target_c == 1 and in_c == 3:
            obs_tensor = TF.rgb_to_grayscale(obs_tensor, num_output_channels=1)
        elif target_c == 3 and in_c == 1:
            obs_tensor = obs_tensor.repeat(1, 3, 1, 1)

        if (in_h, in_w) != (target_h, target_w):
            obs_tensor = F.interpolate(
                obs_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        low, high = self.obs_range
        if (low, high) != (0.0, 1.0):
            obs_tensor = obs_tensor * (high - low) + low

        if single_frame:
            obs_tensor = obs_tensor.squeeze(0)

        out = {k: v for k, v in obs.items() if k != self.obs_key}
        out[self.obs_key] = obs_tensor
        return TensorDict(out, batch_size=obs.batch_size)

    def __call__(self, episode: Episode) -> dict[str, TensorDict | torch.Tensor]:
        """
        Convert one Episode chunk into a dictionary of PyTorch tensors / TensorDicts.

        Args:
            episode (Episode): Trajectory chunk of ``T`` transitions.

        Returns:
            dict[str, TensorDict | torch.Tensor]: Dictionary containing:
                - ``"obs"``: ``TensorDict`` with ``batch_size=[T + 1]`` (``obs[obs_key]`` of shape
                  ``(T + 1, C, H, W)``, dtype ``torch.float32`` in ``obs_range``)
                - ``"action"``: ``torch.Tensor`` of shape ``(T, action_dim)``
                - ``"reward"``: ``torch.Tensor`` of shape ``(T, 1)``, dtype ``torch.float32``
                - ``"done"``: ``torch.Tensor`` of shape ``(T, 1)``, dtype ``torch.float32`` (``0.0`` or ``1.0``)
                - ``"terminated"``: ``torch.Tensor`` of shape ``(T, 1)``, dtype ``torch.float32`` (``0.0`` or ``1.0``)
                - ``"truncated"``: ``torch.Tensor`` of shape ``(T, 1)``, dtype ``torch.float32`` (``0.0`` or ``1.0``)
        """
        obs_curr = episode.obs
        obs_next = episode["next", "obs"]
        obs_seq = obs_curr if len(episode) == 0 else torch.cat([obs_curr[:1], obs_next], dim=0)
        return {
            "obs": self.transform_obs(obs_seq),
            "action": episode.action,
            "reward": episode.reward,
            "done": episode.done.float(),
            "terminated": episode.terminated.float(),
            "truncated": episode.truncated.float(),
        }
