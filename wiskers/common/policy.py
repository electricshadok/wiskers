"""
Common base policy interfaces for discrete and continuous environment rollouts.
"""

from typing import Optional

import gymnasium as gym
import torch
import torch.nn as nn

from wiskers.envs.torch_env import TorchGymEnv


class Policy(nn.Module):
    """
    Base class for all rollout policies (random, heuristic, or learned).

    Attributes:
        action_dim (int): Dimensionality of the stored action vector in Episode.actions
            (1 for discrete scalar actions, or D for D-dimensional continuous actions).
    """

    action_dim: int

    def __init__(self, env: Optional[TorchGymEnv] = None) -> None:
        super().__init__()
        self.env: Optional[TorchGymEnv] = None
        if env is not None:
            self.set_env(env)

    def set_env(self, env: TorchGymEnv) -> None:
        """Validate and attach the TorchGymEnv environment to this policy."""
        self._validate_env(env)
        self.env = env

    def _validate_env(self, env: TorchGymEnv) -> None:
        """Optional hook to verify that ``env`` is compatible with this policy."""
        pass

    def sample_random(self) -> torch.Tensor:
        """Sample a uniform random action tensor from ``self.env``."""
        if self.env is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.env is not set. Pass env to __init__ or call set_env(env)."
            )
        return self.env.sample_action()

    def reset(self) -> None:
        """Reset any internal recurrent or episodic state at the start of an episode."""
        pass

    def forward(
        self,
        obs: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Compute the environment action tensor for the current step.

        Args:
            obs (torch.Tensor): Transformed observation tensor for the current step.
            step (int): Current step index within the episode.

        Returns:
            torch.Tensor: Action tensor (scalar ``int64`` tensor for discrete actions,
            or ``float32`` tensor of shape ``(action_dim,)`` for continuous actions).
        """
        raise NotImplementedError


class DiscretePolicy(Policy):
    """Policy for discrete action spaces returning a scalar ``int64`` ``torch.Tensor``."""

    action_dim: int = 1

    def _validate_env(self, env: TorchGymEnv) -> None:
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"{self.__class__.__name__} requires a Discrete action_space, "
                f"got {env.action_space}."
            )


class ContinuousPolicy(Policy):
    """Policy for continuous action spaces returning a ``float32`` ``torch.Tensor``."""

    def _validate_env(self, env: TorchGymEnv) -> None:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError(
                f"{self.__class__.__name__} requires a continuous Box action_space, "
                f"got {env.action_space}."
            )
        if env.action_space.shape != (self.action_dim,):
            raise ValueError(
                f"{self.__class__.__name__} expects action_dim={self.action_dim}, "
                f"got action_space shape {env.action_space.shape}."
            )
