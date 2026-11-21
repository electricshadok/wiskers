"""
ALE/Breakout environment.
"""

import ale_py
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.torch_env import TorchGymEnv
from wiskers.envs.utils import render_to_tensor


class BreakoutEnv(TorchGymEnv):
    """
    ALE/Breakout-v5 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"ram"`` (uint8[128]): Atari 2600 console RAM state.

    Reference:
        https://ale.farama.org/environments/breakout/

    Args:
        image_size: Target ``(height, width)`` to resize frames to. If ``None``,
            preserves the native ``(210, 160)`` resolution.
    """

    def __init__(self, image_size: tuple[int, int] | None = None) -> None:
        self.image_size = tuple(image_size) if image_size is not None else None
        gym.register_envs(ale_py)
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        super().__init__(env)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        ram = self._gym_env.unwrapped.ale.getRAM()
        return TensorDict(
            {
                "render": render_to_tensor(raw_obs, self.image_size),
                "ram": torch.from_numpy(np.ascontiguousarray(ram, dtype=np.uint8)),
            },
            batch_size=[],
        )
