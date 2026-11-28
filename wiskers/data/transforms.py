"""
Episode-to-tensor transforms for trajectory datasets.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tensordict import TensorDict

from wiskers.data.episode import Episode


class EpisodeToTensor:
    """
    Convert an Episode chunk into a dictionary of PyTorch tensors.

    Only the image field named ``render_key`` is converted from ``uint8``
    ``(..., H, W, C)`` to normalized ``float32`` ``(..., C, H, W)`` in
    ``render_range`` (and optionally resized/channel-converted to
    ``render_shape``). All non-render observation fields pass straight through
    from the ``Episode`` unchanged.

    Args:
        render_key: Observation key to convert as an image tensor (default: ``"render"``).
            If ``None`` or absent from ``obs``, observations are returned unchanged.
        render_shape: Optional target 3-tuple ``(C, H, W)`` where ``C`` is ``1``
            (grayscale) or ``3`` (RGB), and ``(H, W)`` is spatial resolution.
            If ``None``, preserves the native channel count and spatial dimensions.
        render_range: Target 2-tuple ``(low, high)`` specifying the float32 output
            normalization range for ``render_key`` (e.g., ``(0.0, 1.0)`` or ``(-1.0, 1.0)``).
    """

    def __init__(
        self,
        render_key: str | None = "render",
        render_shape: Sequence[int] | None = (3, 64, 64),
        render_range: Sequence[float] = (0.0, 1.0),
    ) -> None:
        shape: tuple[int, int, int] | None = None
        if render_shape is not None:
            if len(render_shape) != 3:
                raise ValueError(f"render_shape must be a 3-element sequence (C, H, W), got {render_shape}.")
            c, h, w = int(render_shape[0]), int(render_shape[1]), int(render_shape[2])
            if c not in (1, 3):
                raise ValueError(f"render_shape channel count C must be 1 or 3, got {c}.")
            if h <= 0 or w <= 0:
                raise ValueError(f"render_shape spatial dimensions (H, W) must be positive, got ({h}, {w}).")
            shape = (c, h, w)

        if len(render_range) != 2:
            raise ValueError(f"render_range must be a 2-element sequence (low, high), got {render_range}.")
        low, high = float(render_range[0]), float(render_range[1])
        if low >= high:
            raise ValueError(f"render_range must satisfy low < high, got ({low}, {high}).")

        self.render_key = render_key
        self.render_shape = shape
        self.render_range: tuple[float, float] = (low, high)

    def transform_obs(self, obs: TensorDict) -> TensorDict:
        """
        Convert raw ``uint8`` image observations under ``self.render_key`` into
        normalized ``float32`` tensors while preserving all other observation keys
        unchanged.

        Supports both a single observation ``batch_size=[]`` (``(H, W, C) -> (C, H, W)``)
        and a sequence ``batch_size=[T]`` (``(T, H, W, C) -> (T, C, H, W)``).
        """
        if self.render_key is None or self.render_key not in obs.keys():
            return obs
        img = obs[self.render_key]
        single_frame = img.ndim == 3
        obs_tensor = img.unsqueeze(0) if single_frame else img

        # (T, H, W, C) -> (T, C, H, W)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2).float() / 255.0

        if self.render_shape is not None:
            target_c, target_h, target_w = self.render_shape
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

        low, high = self.render_range
        if (low, high) != (0.0, 1.0):
            obs_tensor = obs_tensor * (high - low) + low

        if single_frame:
            obs_tensor = obs_tensor.squeeze(0)

        out = {k: v for k, v in obs.items() if k != self.render_key}
        out[self.render_key] = obs_tensor
        return TensorDict(out, batch_size=obs.batch_size)

    def __call__(self, episode: Episode) -> dict[str, TensorDict | torch.Tensor]:
        """
        Convert one Episode chunk into a dictionary of PyTorch tensors / TensorDicts.

        Args:
            episode (Episode): Trajectory chunk of ``T`` transitions.

        Returns:
            dict[str, TensorDict | torch.Tensor]: Dictionary containing:
                - ``"obs"``: ``TensorDict`` with ``batch_size=[T + 1]`` (``obs[render_key]``
                  as ``torch.float32`` ``(T + 1, C, H, W)`` in ``render_range`` when present;
                  all other observation fields passed through unchanged)
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
