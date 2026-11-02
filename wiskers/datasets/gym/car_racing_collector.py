"""
CarRacing → ReplayBuffer collector.

Provides helper functions to roll out episodes of CarRacing-v3 with
a given CarRacingPolicy, pushing resulting Episodes into a ReplayBuffer.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

from wiskers.datasets.gym.car_racing_policies import CarRacingPolicy
from wiskers.datasets.trajectories.collector import Collector
from wiskers.datasets.trajectories.episode import Episode
from wiskers.datasets.trajectories.replay_buffer import ReplayBuffer


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


def _collect_one_episode(
    env: gym.Env,
    max_steps: int,
    seed: int,
    policy: CarRacingPolicy,
) -> Episode:
    """
    Run a single CarRacing episode with the specified policy.

    Args:
        env      : An already-constructed NoZoomCarRacing instance.
        max_steps: Maximum number of environment steps per episode.
        seed     : Seed passed to env.reset() for reproducibility.
        policy   : Policy used to select actions.

    Returns:
        Episode with arrays of shape (T, H, W, C), (T, 3), (T,), (T,).
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

        action = policy(obs, env, step)

        step_res = env.step(action)
        if len(step_res) == 5:
            next_obs, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_res

        assert isinstance(next_obs, np.ndarray), f"Expected np.ndarray observation from step, got {type(next_obs)}"
        obs = next_obs
        observations.append(next_obs)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)

    # observations has one extra frame at the front; trim to match action length
    T = len(actions)
    return Episode(
        obs=np.array(observations[:T], dtype=np.uint8),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def collect_car_racing_episodes(
    replay_buffer: ReplayBuffer,
    policy: CarRacingPolicy,
    n_episodes: int = 10,
    min_steps: int = 0,
    max_steps: int = 500,
    seed: int = 0,
    image_size: Optional[tuple[int, int]] = (64, 64),
    verbose: bool = True,
) -> int:
    """
    Collect CarRacing rollouts and push them into ``replay_buffer`` using Collector.

    Args:
        replay_buffer : Target ReplayBuffer to populate.
        policy        : Policy instance to collect rollouts with.
        n_episodes    : Number of episodes to collect.
        min_steps     : Minimum steps per episode (shorter episodes are discarded).
        max_steps     : Maximum steps per episode (clamped).
        seed          : Base RNG seed; each episode gets seed + episode_index.
        image_size    : Target image resolution (H, W). Default (64, 64).
        verbose       : Print progress to stdout.

    Returns:
        Total steps collected.
    """
    env = make_car_racing_env(image_size=image_size)
    try:
        collector = Collector(
            env=env,
            policy=policy,
            num_episodes=n_episodes,
            min_steps=min_steps,
            max_steps=max_steps,
            seed=seed,
            verbose=verbose,
        )
        return collector.collect(replay_buffer)
    finally:
        env.close()
