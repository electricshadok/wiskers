"""
CarRacing-v3 environment.
"""

import numpy as np
import torch
from gymnasium.envs.box2d.car_racing import CarRacing
from tensordict import TensorDict

from wiskers.envs.gym_env import GymEnv
from wiskers.envs.utils import render_to_tensor


class NoZoomCarRacing(CarRacing):
    """
    Subclass of CarRacing that locks the camera zoom immediately to gameplay scale (t=1.0),
    bypassing the initial zooming-in animation on reset, and optionally hides the bottom HUD.
    """

    def __init__(self, show_indicators: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.show_indicators = show_indicators
        self._clean_surf = None

    def _render_indicators(self, W, H):
        if self.show_indicators:
            super()._render_indicators(W, H)
        elif self.surf is not None:
            self._clean_surf = self.surf.copy()

    def _create_image_array(self, screen, size):
        if not self.show_indicators and self._clean_surf is not None:
            screen = self._clean_surf
        return super()._create_image_array(screen, size)

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)

        # Lock camera zoom immediately to standard gameplay scale (bypass zoom-in animation)
        self.t = 1.0
        obs = self._render("state_pixels")
        return obs, info


class CarRacingEnv(GymEnv):
    """
    CarRacing-v3 environment (with locked gameplay zoom) emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"state:car"`` (float32[6]): ``[x, y, angle, vx, vy, angular_velocity]``
          from the Box2D chassis body.
        - ``"state:wheels"`` (float32[5]): ``[w0_omega, w1_omega, w2_omega, w3_omega, steering_angle]``
          (4 wheel ABS angular speeds + front wheel steering joint angle).

    Reference:
        https://gymnasium.farama.org/environments/box2d/car_racing/

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(96, 96)`` resolution.
        show_indicators: Whether to render the bottom black HUD bar and score text.
    """

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        show_indicators: bool = False,
    ) -> None:
        self.image_size = tuple(image_size) if image_size is not None else None
        env = NoZoomCarRacing(render_mode="rgb_array", show_indicators=show_indicators)
        super().__init__(env)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        car = self._gym_env.car
        assert car is not None
        hull = car.hull
        return TensorDict(
            {
                "render": render_to_tensor(raw_obs, self.image_size),
                "state:car": torch.tensor(
                    [
                        hull.position[0],
                        hull.position[1],
                        hull.angle,
                        hull.linearVelocity[0],
                        hull.linearVelocity[1],
                        hull.angularVelocity,
                    ],
                    dtype=torch.float32,
                ),
                "state:wheels": torch.tensor(
                    [*(w.omega for w in car.wheels), car.wheels[0].joint.angle],
                    dtype=torch.float32,
                ),
            },
            batch_size=[],
        )
