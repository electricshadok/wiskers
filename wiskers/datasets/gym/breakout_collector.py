"""
ALE/Breakout → ReplayBuffer collector.

Provides helper functions to roll out episodes of ALE/Breakout-v5 with
a given BreakoutPolicy, pushing resulting Episodes into a ReplayBuffer.
"""

from typing import Optional

import ale_py
import gymnasium as gym
import numpy as np

from wiskers.datasets.gym.breakout_policies import BreakoutPolicy
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


# Register all ALE (Atari) environments
gym.register_envs(ale_py)


def make_breakout_env(image_size: Optional[tuple[int, int]] = (64, 64)) -> gym.Env:
    """
    Construct an ALE/Breakout-v5 environment with optional observation resizing.

    Args:
        image_size: Target (height, width) to resize frames to. If None, native
                    resolution (210, 160) is preserved.

    Returns:
        Gymnasium environment instance.
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    if image_size is not None:
        env = gym.wrappers.ResizeObservation(env, image_size)
    return env


def _collect_one_episode(
    env: gym.Env,
    max_steps: int,
    seed: int,
    policy: BreakoutPolicy,
) -> Episode:
    """
    Run a single Breakout episode using the provided policy.

    Args:
        env      : An already-constructed Breakout environment instance.
        max_steps: Maximum number of environment steps per episode.
        seed     : Seed passed to env.reset() for reproducibility.
        policy   : Policy used to select actions.

    Returns:
        Episode with arrays of shape (T, H, W, C), (T, 1), (T,), (T,).
    """
    reset_res = env.reset(seed=seed)
    obs, _ = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
    assert isinstance(obs, np.ndarray), f"Expected np.ndarray observation from reset, got {type(obs)}"

    observations: list[np.ndarray] = [obs]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []

    policy.reset()
    done = False
    for step in range(max_steps):
        if done:
            break

        action_idx = policy(obs, env, step)
        action = np.array([float(action_idx)], dtype=np.float32)

        step_res = env.step(action_idx)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_res

        assert isinstance(next_obs, np.ndarray), f"Expected np.ndarray observation from step, got {type(next_obs)}"
        observations.append(next_obs)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)

    T = len(actions)
    return Episode(
        obs=np.array(observations[:T], dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def collect_breakout_episodes(
    replay_buffer: ReplayBuffer,
    policy: BreakoutPolicy,
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 0,
    image_size: Optional[tuple[int, int]] = (64, 64),
    verbose: bool = True,
) -> None:
    """
    Collect ``n_episodes`` Breakout rollouts and push them into ``replay_buffer``.

    Episodes shorter than ``replay_buffer.chunk_len`` are silently skipped by the
    buffer itself.

    Args:
        replay_buffer : Target ReplayBuffer to populate.
        policy        : Policy instance to collect rollouts with.
        n_episodes    : Number of episodes to collect.
        max_steps     : Maximum steps per episode.
        seed          : Base RNG seed; each episode gets seed + episode_index.
        image_size    : Target image resolution (H, W). Default (64, 64).
        verbose       : Print progress to stdout.
    """
    env = make_breakout_env(image_size=image_size)
    try:
        for i in range(n_episodes):
            episode = _collect_one_episode(
                env=env,
                max_steps=max_steps,
                seed=seed + i,
                policy=policy,
            )
            replay_buffer.add(episode)
            if verbose:
                print(
                    f"[Breakout] episode {i + 1}/{n_episodes} | "
                    f"T={len(episode)} steps | "
                    f"total_reward={episode.rewards.sum():.1f} | "
                    f"buffer={replay_buffer}"
                )
    finally:
        env.close()
