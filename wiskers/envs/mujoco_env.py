"""
Shared base wrapper for MuJoCo Gymnasium environments.
"""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from wiskers.envs.gym_env import GymEnv
from wiskers.envs.utils import render_to_tensor


class MujocoEnv(GymEnv):
    """
    Base adapter for MuJoCo-backed Gymnasium environments.

    Centralizes visual frame resizing via ``_render_frame()`` and exposes
    YAML-configurable physics overrides (starting with ``gravity``).

    Args:
        env: Underlying MuJoCo Gymnasium environment.
        image_size: Optional target ``(height, width)`` for rendered RGB frames.
            If ``None``, preserves the environment's native resolution.
        gravity: Optional 3-element ``(gx, gy, gz)`` gravity vector override in m/s²
            written to ``model.opt.gravity``. Defaults to ``None`` (keep environment's XML gravity).
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] | None = None,
        gravity: Sequence[float] | None = None,
    ) -> None:
        super().__init__(env)
        self.image_size: tuple[int, int] | None = (image_size[0], image_size[1]) if image_size is not None else None
        if gravity is not None:
            self.set_gravity(gravity)

    @property
    def model(self) -> Any:
        """Return the underlying MuJoCo ``MjModel``."""
        mj_model = getattr(self._gym_env.unwrapped, "model", None)
        if mj_model is None:
            raise RuntimeError(f"{type(self).__name__} underlying env has no MuJoCo model.")
        return mj_model

    @property
    def gravity(self) -> tuple[float, float, float]:
        """Return the current ``(gx, gy, gz)`` gravity vector in m/s²."""
        g = self.model.opt.gravity
        return (float(g[0]), float(g[1]), float(g[2]))

    def set_gravity(self, gravity: Sequence[float]) -> None:
        """Override ``model.opt.gravity`` with a 3-element ``(gx, gy, gz)`` vector in m/s²."""
        if len(gravity) != 3:
            raise ValueError(f"gravity must have length 3, got {len(gravity)}.")
        self.model.opt.gravity[:] = [float(v) for v in gravity]

    def _render_frame(self) -> torch.Tensor:
        """Render an RGB frame from the underlying MuJoCo env as a ``uint8`` tensor."""
        frame = self._gym_env.render()
        if frame is None:
            raise RuntimeError(f"{type(self).__name__} render() returned None. Ensure render_mode='rgb_array'.")
        return render_to_tensor(np.asarray(frame, dtype=np.uint8), self.image_size)
