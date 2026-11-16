"""
Base policy interfaces for discrete and continuous environment rollouts.
"""

import gymnasium as gym
import torch
import torch.nn as nn


class Policy(nn.Module):
    """
    Base class for all rollout policies (random, heuristic, or learned).

    Attributes:
        action_dim (int): Dimensionality of the stored action vector in Episode.action
            (1 for discrete scalar actions, or D for D-dimensional continuous actions).
    """

    action_dim: int

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


class ContinuousPolicy(Policy):
    """Policy for continuous action spaces returning a ``float32`` ``torch.Tensor`` of shape ``(action_dim,)``."""

    action_dim: int


class RandomPolicy(Policy):
    """Uniform random exploration across a Gymnasium action space (Discrete or Box)."""

    def __init__(self, action_space: gym.Space) -> None:
        super().__init__()
        self.action_space = action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = 1
        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = int(action_space.shape[0])
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")

    def forward(self, obs: torch.Tensor, step: int = 0) -> torch.Tensor:
        sample = self.action_space.sample()
        if isinstance(self.action_space, gym.spaces.Discrete):
            return torch.tensor(sample, dtype=torch.int64)
        return torch.from_numpy(sample).float()
