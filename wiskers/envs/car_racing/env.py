"""
CarRacing-v3 environment factory and wrappers.
"""

from typing import Optional

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing


class NoZoomCarRacing(CarRacing):
    """
    Subclass of CarRacing that locks the camera zoom immediately to gameplay scale (t=1.0),
    bypassing the initial zooming-in animation on reset.
    """

    def reset(self, **kwargs):
        obs_info = super().reset(**kwargs)
        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            obs, info = obs_info
        else:
            obs, info = obs_info, {}

        # Lock camera zoom immediately to standard gameplay scale (bypass zoom-in animation)
        self.t = 1.0
        obs = self._render("state_pixels")

        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            return obs, info
        return obs


def make_car_racing_env(image_size: Optional[tuple[int, int]] = (64, 64)) -> gym.Env:
    """
    Construct a NoZoomCarRacing environment with optional observation resizing.

    Args:
        image_size: Target (height, width) to resize frames to. If None, native
                    resolution (96, 96) is preserved.

    Returns:
        Gymnasium environment instance.
    """
    env = NoZoomCarRacing(render_mode="rgb_array")
    if image_size is not None:
        env = gym.wrappers.ResizeObservation(env, tuple(image_size))
    return env

