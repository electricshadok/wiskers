"""
ALE/Breakout environment.
"""

from collections.abc import Mapping

import ale_py
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from wiskers.envs.gym_env import GymEnv, ObsFieldTransform


class BreakoutEnv(GymEnv):
    """
    ALE/Breakout-v5 environment emitting multi-modal TensorDict observations.

    Observation keys (batch_size=[]):
        - ``"render"`` (uint8[H, W, 3]): RGB camera frame.
        - ``"ram"`` (uint8[128]): Atari 2600 console RAM state.

    Reference:
        https://ale.farama.org/environments/breakout/

    Args:
        obs_fields: Optional mapping from observation field name to per-field transform
            callable (or ``None`` to keep unchanged). Defaults to ``None``.
    """

    obs_keys: tuple[str, ...] = ("render", "ram")

    def __init__(
        self,
        obs_fields: Mapping[str, ObsFieldTransform | None] | None = None,
    ) -> None:
        gym.register_envs(ale_py)
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        super().__init__(env, obs_fields=obs_fields)

    def _process_obs(self, raw_obs: np.ndarray) -> TensorDict:
        ram = self._gym_env.unwrapped.ale.getRAM()
        return TensorDict(
            {
                "render": torch.from_numpy(np.ascontiguousarray(raw_obs, dtype=np.uint8)),
                "ram": torch.from_numpy(np.ascontiguousarray(ram, dtype=np.uint8)),
            },
            batch_size=[],
        )
