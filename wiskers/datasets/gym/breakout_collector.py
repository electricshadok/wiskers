"""
ALE/Breakout → ReplayBuffer collector.

Provides helper functions to roll out episodes of ALE/Breakout-v5 with
a heuristic ball-tracking policy or random exploration, pushing resulting
Episodes into a ReplayBuffer.
"""

import random
from typing import Callable, Optional, Union

import ale_py
import gymnasium as gym
import numpy as np

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


def _get_heuristic_action(env: gym.Env, step: int, epsilon: float = 0.1) -> int:
    """
    Heuristic policy using Atari RAM state to track the ball with the paddle.

    Atari Breakout RAM addresses:
      - RAM[72]:  paddle x-position
      - RAM[99]:  ball x-position
      - RAM[101]: ball y-position (0 when ball is waiting to be served)
    """
    if random.random() < epsilon:
        # Periodic FIRE during random exploration to avoid stuck ball
        if step % 30 == 0:
            return 1  # FIRE
        return int(env.action_space.sample())

    try:
        ale = getattr(env.unwrapped, "ale", None)
        if ale is None:
            return int(env.action_space.sample())
        ram = ale.getRAM()
        paddle_x = int(ram[72])
        ball_x = int(ram[99])
        ball_y = int(ram[101])

        # If ball is waiting to be served or just reset, press FIRE
        if ball_y == 0 or step == 0:
            return 1  # FIRE

        # Track the ball with the paddle
        if paddle_x < ball_x - 2:
            return 2  # RIGHT
        elif paddle_x > ball_x + 2:
            return 3  # LEFT
        else:
            return random.choice([0, 2, 3])  # small jitter / NOOP near center
    except Exception:
        return int(env.action_space.sample())


def _collect_one_episode(
    env: gym.Env,
    max_steps: int,
    seed: int,
    policy: Union[str, Callable[[np.ndarray, gym.Env, int], int]] = "heuristic",
    epsilon: float = 0.1,
) -> Episode:
    """
    Run a single Breakout episode.

    Args:
        env      : An already-constructed Breakout environment instance.
        max_steps: Maximum number of environment steps per episode.
        seed     : Seed passed to env.reset() for reproducibility.
        policy   : Policy type: 'heuristic', 'random', or a callable(obs, env, step) -> action.
        epsilon  : Random exploration probability when using 'heuristic'.

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

    done = False
    for step in range(max_steps):
        if done:
            break

        # Select action
        if callable(policy):
            action_idx = int(policy(obs, env, step))
        elif policy == "heuristic":
            action_idx = _get_heuristic_action(env, step, epsilon=epsilon)
        elif policy == "random":
            action_idx = 1 if step % 30 == 0 else int(env.action_space.sample())
        else:
            raise ValueError(f"Unknown policy '{policy}'. Expected 'heuristic', 'random', or a Callable.")

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
    buffer: ReplayBuffer,
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = 0,
    image_size: Optional[tuple[int, int]] = (64, 64),
    policy: Union[str, Callable[[np.ndarray, gym.Env, int], int]] = "heuristic",
    epsilon: float = 0.1,
    verbose: bool = True,
) -> None:
    """
    Collect ``n_episodes`` Breakout rollouts and push them into ``buffer``.

    Episodes shorter than ``buffer.chunk_len`` are silently skipped by the
    buffer itself.

    Args:
        buffer     : Target ReplayBuffer to populate.
        n_episodes : Number of episodes to collect.
        max_steps  : Maximum steps per episode.
        seed       : Base RNG seed; each episode gets seed + episode_index.
        image_size : Target image resolution (H, W). Default (64, 64).
        policy     : 'heuristic', 'random', or custom callable policy.
        epsilon    : Exploration rate for heuristic policy.
        verbose    : Print progress to stdout.
    """
    env = make_breakout_env(image_size=image_size)
    try:
        for i in range(n_episodes):
            episode = _collect_one_episode(
                env=env,
                max_steps=max_steps,
                seed=seed + i,
                policy=policy,
                epsilon=epsilon,
            )
            buffer.add(episode)
            if verbose:
                print(
                    f"[Breakout] episode {i + 1}/{n_episodes} | "
                    f"T={len(episode)} steps | "
                    f"total_reward={episode.rewards.sum():.1f} | "
                    f"buffer={buffer}"
                )
    finally:
        env.close()
