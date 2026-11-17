"""
CarRacing-v3 environment.
"""

from gymnasium.envs.box2d.car_racing import CarRacing

from wiskers.envs.torch_env import TorchGymEnv


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


class CarRacingEnv(TorchGymEnv):
    """
    CarRacing-v3 environment (with locked gameplay zoom) emitting PyTorch uint8 RGB observations.

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
        env = NoZoomCarRacing(render_mode="rgb_array", show_indicators=show_indicators)
        super().__init__(env, image_size=image_size)
