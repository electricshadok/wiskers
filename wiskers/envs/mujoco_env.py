"""
Shared base wrapper for MuJoCo Gymnasium environments.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import torch

from wiskers.envs.gym_env import GymEnv
from wiskers.envs.utils import render_to_tensor


@dataclass(frozen=True)
class MujocoPhysics:
    """
    Deterministic MuJoCo physics overrides applied relative to nominal XML values.

    Any field left as ``None`` keeps the environment's nominal XML value (or restores it
    when passed to ``MujocoEnv.apply_physics``).

    Args:
        gravity: Optional 3-element ``(gx, gy, gz)`` gravity vector in m/s² written to
            ``model.opt.gravity``.
        timestep: Optional simulation timestep in seconds (> 0) written to ``model.opt.timestep``.
        friction_scale: Optional non-negative multiplier applied to nominal ``model.geom_friction``.
        mass_scale: Optional positive multiplier applied to nominal ``model.body_mass`` and
            ``model.body_inertia``.
        damping_scale: Optional non-negative multiplier applied to nominal ``model.dof_damping``.
    """

    gravity: Sequence[float] | None = None
    timestep: float | None = None
    friction_scale: float | None = None
    mass_scale: float | None = None
    damping_scale: float | None = None

    def __post_init__(self) -> None:
        if self.gravity is not None:
            if len(self.gravity) != 3:
                raise ValueError(f"gravity must have length 3, got {len(self.gravity)}.")
            object.__setattr__(self, "gravity", tuple(float(v) for v in self.gravity))
        if self.timestep is not None and self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}.")
        if self.friction_scale is not None and self.friction_scale < 0:
            raise ValueError(f"friction_scale must be non-negative, got {self.friction_scale}.")
        if self.mass_scale is not None and self.mass_scale <= 0:
            raise ValueError(f"mass_scale must be positive, got {self.mass_scale}.")
        if self.damping_scale is not None and self.damping_scale < 0:
            raise ValueError(f"damping_scale must be non-negative, got {self.damping_scale}.")


class MujocoEnv(GymEnv):
    """
    Base adapter for MuJoCo-backed Gymnasium environments.

    Centralizes visual frame resizing via ``_render_frame()`` and exposes
    deterministic YAML-configurable physics overrides via :class:`MujocoPhysics`.

    Args:
        env: Underlying MuJoCo Gymnasium environment.
        image_size: Optional target ``(height, width)`` for rendered RGB frames.
            If ``None``, preserves the environment's native resolution.
        physics: Optional :class:`MujocoPhysics` configuration applied on top of
            the environment's nominal XML physics. Defaults to ``None``.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] | None = None,
        physics: MujocoPhysics | None = None,
    ) -> None:
        super().__init__(env)
        self.image_size: tuple[int, int] | None = (image_size[0], image_size[1]) if image_size is not None else None

        mj_model = self.model
        self._nominal_gravity: np.ndarray = mj_model.opt.gravity.copy()
        self._nominal_timestep: float = float(mj_model.opt.timestep)
        self._nominal_geom_friction: np.ndarray = mj_model.geom_friction.copy()
        self._nominal_body_mass: np.ndarray = mj_model.body_mass.copy()
        self._nominal_body_inertia: np.ndarray = mj_model.body_inertia.copy()
        self._nominal_dof_damping: np.ndarray = mj_model.dof_damping.copy()

        self._physics: MujocoPhysics = MujocoPhysics()
        if physics is not None:
            self.apply_physics(physics)

    @property
    def model(self) -> Any:
        """Return the underlying MuJoCo ``MjModel``."""
        mj_model = getattr(self._gym_env.unwrapped, "model", None)
        if mj_model is None:
            raise RuntimeError(f"{type(self).__name__} underlying env has no MuJoCo model.")
        return mj_model

    @property
    def data(self) -> Any:
        """Return the underlying MuJoCo ``MjData``."""
        mj_data = getattr(self._gym_env.unwrapped, "data", None)
        if mj_data is None:
            raise RuntimeError(f"{type(self).__name__} underlying env has no MuJoCo data.")
        return mj_data

    @property
    def physics(self) -> MujocoPhysics:
        """Return the active :class:`MujocoPhysics` configuration."""
        return self._physics

    @property
    def gravity(self) -> tuple[float, float, float]:
        """Return the current ``(gx, gy, gz)`` gravity vector in m/s²."""
        g = self.model.opt.gravity
        return (float(g[0]), float(g[1]), float(g[2]))

    @property
    def timestep(self) -> float:
        """Return the current simulation timestep in seconds."""
        return float(self.model.opt.timestep)

    def apply_physics(self, physics: MujocoPhysics) -> None:
        """Apply ``physics`` overrides relative to the snapshotted nominal XML values."""
        mj_model = self.model
        mj_model.opt.gravity[:] = physics.gravity if physics.gravity is not None else self._nominal_gravity
        mj_model.opt.timestep = physics.timestep if physics.timestep is not None else self._nominal_timestep
        mj_model.geom_friction[:] = self._nominal_geom_friction * (
            physics.friction_scale if physics.friction_scale is not None else 1.0
        )
        mass_scale = physics.mass_scale if physics.mass_scale is not None else 1.0
        mj_model.body_mass[:] = self._nominal_body_mass * mass_scale
        mj_model.body_inertia[:] = self._nominal_body_inertia * mass_scale
        mj_model.dof_damping[:] = self._nominal_dof_damping * (
            physics.damping_scale if physics.damping_scale is not None else 1.0
        )
        mujoco.mj_setConst(mj_model, self.data)
        self._physics = physics

    def _render_frame(self) -> torch.Tensor:
        """Render an RGB frame from the underlying MuJoCo env as a ``uint8`` tensor."""
        frame = self._gym_env.render()
        if frame is None:
            raise RuntimeError(f"{type(self).__name__} render() returned None. Ensure render_mode='rgb_array'.")
        return render_to_tensor(np.asarray(frame, dtype=np.uint8), self.image_size)
