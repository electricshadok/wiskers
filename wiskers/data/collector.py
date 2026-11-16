"""
Modular rollout collector for trajectory-based RL and world models.
"""

import gymnasium as gym
import torch
from tensordict import TensorDict

from wiskers.data.buffer import Buffer
from wiskers.data.episode import Episode
from wiskers.data.transforms import EpisodeTransform
from wiskers.envs.torch_env import TorchGymEnv
from wiskers.policies import Policy


class Collector:
    """
    Rollout collector that gathers episodes from a TorchGymEnv using a policy
    and stores them in a Buffer as CPU PyTorch tensors.

    Guarantees:
        - Collects ``num_episodes`` valid episodes.
        - Enforces ``min_steps``: any episode ending earlier than ``min_steps`` is discarded.
        - Enforces ``max_steps``: clamps episodes from exceeding ``max_steps``.
        - Stores observations as compact ``uint8`` CPU tensors while passing transformed
          PyTorch tensors to the policy.

    Args:
        env (TorchGymEnv): The PyTorch-wrapped Gymnasium environment instance.
        policy (Policy): Policy instance implementing ``Policy``.
        transform (EpisodeTransform): Transform used to convert raw ``uint8``
            observations into normalized ``torch.Tensor`` before passing them to ``policy``.
        num_episodes (int): Target number of valid episodes to collect.
        min_steps (int): Minimum step cutoff; episodes with fewer steps are rejected.
        max_steps (int): Maximum step cutoff per episode (clamped).
        seed (int): Base RNG seed for environment resets (must be an int for reproducibility).
        verbose (bool): Whether to log collection progress.
    """

    def __init__(
        self,
        env: TorchGymEnv,
        policy: Policy,
        transform: EpisodeTransform,
        num_episodes: int = 10,
        min_steps: int = 0,
        max_steps: int = 500,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        # Ensure neural policies run in eval mode during rollout to disable Dropout
        # and freeze BatchNorm statistics, while supporting non-nn.Module heuristic policies.
        if isinstance(self.policy, torch.nn.Module):
            self.policy.eval()
        self._validate_policy_action_dim()
        self.transform = transform
        self.num_episodes = num_episodes
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.seed = seed
        self.verbose = verbose
        self._episode_idx = 0

    def _validate_policy_action_dim(self) -> None:
        """Verify that policy action_dim is compatible with env action_space."""
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            if self.policy.action_dim != 1:
                raise ValueError(
                    f"Env has Discrete action space, but policy has action_dim={self.policy.action_dim} (expected 1)."
                )
        elif isinstance(self.env.action_space, gym.spaces.Box):
            expected_dim = self.env.action_space.shape[0] if self.env.action_space.shape else 1
            if self.policy.action_dim != expected_dim:
                raise ValueError(
                    f"Env has Box action space with shape {self.env.action_space.shape}, "
                    f"but policy has action_dim={self.policy.action_dim} (expected {expected_dim})."
                )

    @torch.inference_mode()
    def collect_one_episode(self, episode_idx: int = 0) -> Episode:
        """
        Run a single environment rollout using self.env and self.policy up to self.max_steps.

        Args:
            episode_idx (int): Sequence index of the episode to offset the base seed (seed + episode_idx).

        Returns:
            Episode containing collected transitions as CPU tensors.
        """
        self.policy.reset()

        obs, _ = self.env.reset(seed=self.seed + episode_idx)

        observations: list[torch.Tensor] = [obs]
        actions: list[torch.Tensor] = []
        rewards: list[float] = []
        terminated_list: list[bool] = []
        truncated_list: list[bool] = []

        done = False
        for step in range(self.max_steps):
            if done:
                break

            obs_tensor = self.transform.transform_obs(obs)
            action = self.policy(obs_tensor, step)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            obs = next_obs
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)

        return Episode(
            obs=TensorDict({"render": torch.stack(observations)}, batch_size=[len(observations)]),
            action=torch.stack(actions).to(device="cpu"),
            reward=torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            terminated=torch.tensor(terminated_list, dtype=torch.bool).unsqueeze(-1),
            truncated=torch.tensor(truncated_list, dtype=torch.bool).unsqueeze(-1),
        )

    def collect(
        self,
        buffer: Buffer,
    ) -> int:
        """
        Collect ``self.num_episodes`` valid episodes into ``buffer``.
        Episodes shorter than ``self.min_steps`` are discarded and re-collected.

        Args:
            buffer: Target Buffer (ReplayBuffer, RolloutBuffer) to store episodes.

        Returns:
            Total environment steps collected across all accepted episodes.
        """
        collected = 0
        total_steps = 0

        while collected < self.num_episodes:
            episode = self.collect_one_episode(episode_idx=self._episode_idx)
            self._episode_idx += 1
            if len(episode) < self.min_steps:
                continue

            buffer.add(episode)
            collected += 1
            total_steps += len(episode)

            if self.verbose:
                print(
                    f"[Collector] episode {collected}/{self.num_episodes} | "
                    f"T={len(episode)} steps | "
                    f"reward={episode.reward.sum().item():.1f} | "
                    f"buffer={buffer}"
                )

        return total_steps

    def close(self) -> None:
        self.env.close()
