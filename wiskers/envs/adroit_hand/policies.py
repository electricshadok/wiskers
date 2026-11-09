"""
Policies for AdroitHand dexterous manipulation environments.
"""

from typing import Optional, Sequence

import gymnasium as gym
import torch

from wiskers.common.heads import ContinuousActor
from wiskers.common.policy import ContinuousPolicy
from wiskers.envs.torch_env import TorchGymEnv


class AdroitHandPolicy(ContinuousPolicy):
    """Base policy for AdroitHand. Continuous action space (24..30 DoF depending on task)."""

    action_dim: int = 28

    def _validate_env(self, env: TorchGymEnv) -> None:
        if not isinstance(env.action_space, gym.spaces.Box) or not env.action_space.shape:
            raise ValueError(
                f"{self.__class__.__name__} requires a continuous Box action_space, "
                f"got {env.action_space}."
            )
        self.action_dim = env.action_space.shape[0]


class AdroitHandRandomPolicy(AdroitHandPolicy):
    """Uniform random exploration for AdroitHand."""

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        return self.sample_random()


class AdroitHandHeuristicPolicy(AdroitHandPolicy):
    """
    Smooth exploration policy using momentum / autoregressive noise for AdroitHand control.

    Generates coherent reaching and finger-articulation trajectories rather than high-frequency jitter.

    Args:
        momentum: Autoregressive decay factor in [0, 1). Higher values yield smoother trajectories.
        noise_scale: Standard deviation of injected action noise.
        env: Optional TorchGymEnv environment instance.
    """

    def __init__(
        self,
        momentum: float = 0.85,
        noise_scale: float = 0.25,
        env: Optional[TorchGymEnv] = None,
    ):
        super().__init__(env=env)
        self.momentum = momentum
        self.noise_scale = noise_scale
        self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)

    def reset(self) -> None:
        self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        if self.current_action.shape[0] != self.action_dim:
            self.current_action = torch.zeros(self.action_dim, dtype=torch.float32)
        noise = torch.randn(self.action_dim, dtype=torch.float32) * self.noise_scale
        action = self.momentum * self.current_action + (1.0 - self.momentum) * noise
        self.current_action = torch.clamp(action, -1.0, 1.0)
        return self.current_action.clone()


class AdroitHandNeuralPolicy(AdroitHandPolicy):
    """
    Minimal trainable policy for AdroitHand: flattened pixels → MLP → squashed Gaussian in (-1, 1).

    Samples in train mode and uses the mean in eval mode. Unlike the random/heuristic
    policies, the network size is fixed at construction, so ``action_dim`` must match
    the task's action space (validated in ``set_env``).

    Args:
        image_size: Transformed observation shape ``(C, H, W)``.
        action_dim: Action dimensionality of the target task (e.g. 28 for ``door``).
        hidden_size: MLP hidden width.
        num_layers: Number of MLP hidden layers.
        env: Optional TorchGymEnv environment instance.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        action_dim: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        env: Optional[TorchGymEnv] = None,
    ):
        self.action_dim = action_dim
        super().__init__(env=env)
        c, h, w = image_size
        self.actor = ContinuousActor(
            feat_size=c * h * w,
            action_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def _validate_env(self, env: TorchGymEnv) -> None:
        if env.action_space.shape != (self.action_dim,):
            raise ValueError(
                f"{self.__class__.__name__} expects action_dim={self.action_dim}, "
                f"got action_space shape {env.action_space.shape}."
            )
        super()._validate_env(env)

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        feat = obs.flatten(start_dim=-3)
        return self.actor.sample(feat)[0] if self.training else self.actor(feat)[0]
