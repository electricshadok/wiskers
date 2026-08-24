import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ale_py
import gymnasium as gym
import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


# Register all ALE (Atari) environments so the ALE/ namespace is available
gym.register_envs(ale_py)


@dataclass
class PreprocessingConfig:
    num_train_rollouts: int = 20
    num_val_rollouts: int = 4
    num_test_rollouts: int = 4
    max_steps_per_rollout: int = 500


@dataclass
class TransformConfig:
    image_size: List[int] = field(default_factory=lambda: [64, 64])
    grayscale: bool = True


class BreakoutDataset(Dataset):
    """
    Yields transition step dictionaries for world model training:
    (image, action) -> next_image

    Raw RGB frames are stored on disk at native resolution (210x160).
    Grayscale conversion and resizing are applied on-the-fly in __getitem__,
    consistent with CarRacingDataset.
    """

    def __init__(
        self,
        split_dir: str,
        image_size: Tuple[int, int],
        grayscale: bool = True,
    ):
        super().__init__()
        self.split_dir = split_dir
        self.image_size = image_size
        self.grayscale = grayscale
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        if not self.file_paths:
            raise FileNotFoundError(f"No rollout files (.npz) found in {split_dir}")

        self.transitions = []

        # Build flat index of all transition steps across all rollout files
        for file_path in self.file_paths:
            with np.load(file_path) as data:
                seq_len = len(data["observations"])
            for step_idx in range(seq_len - 1):
                self.transitions.append((file_path, step_idx))

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, step_idx = self.transitions[idx]
        with np.load(file_path) as data:
            frame_t = data["observations"][step_idx]        # (H, W, C) RGB
            frame_next = data["observations"][step_idx + 1] # (H, W, C) RGB
            action_t = data["actions"][step_idx]            # scalar int
            reward_t = data["rewards"][step_idx]            # scalar float
            done_t = data["dones"][step_idx]                # scalar bool

        def process_frame(frame: np.ndarray) -> torch.Tensor:
            # RGB (H, W, C) -> (C, H, W), normalized to [0, 1]
            t = torch.from_numpy(frame).float() / 255.0
            t = t.permute(2, 0, 1)
            if self.grayscale:
                # On-the-fly grayscale: (3, H, W) -> (1, H, W) using luminance weights
                t = 0.2989 * t[0:1] + 0.5870 * t[1:2] + 0.1140 * t[2:3]
            t = TF.resize(t, list(self.image_size), antialias=True)
            return t

        media = process_frame(frame_t)
        media_next = process_frame(frame_next)
        action = torch.tensor([float(action_t)]).float()
        reward = torch.tensor([float(reward_t)]).float()
        done = torch.tensor([float(done_t)]).float()

        return {
            "media": media,
            "action": action,
            "reward": reward,
            "media_next": media_next,
            "done": done,
        }


class BreakoutDataModule(L.LightningDataModule):
    """
    DataModule for ALE/Breakout-v5 Gym dataset preparation and loading.

    Rollouts are collected using a uniformly random discrete policy.
    Each rollout file contains max_steps_per_rollout frames, spanning
    multiple internal lives/episodes via auto-reset.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        preprocessing: Any,
        transform: Any,
        splits: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_root = os.path.join(data_dir, "breakout")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits or ["train", "val", "test"]

        if isinstance(preprocessing, dict):
            self.preprocessing = PreprocessingConfig(**preprocessing)
        else:
            self.preprocessing = preprocessing

        if isinstance(transform, dict):
            self.transform = TransformConfig(**transform)
        else:
            self.transform = transform

    def prepare_data(self) -> None:
        """
        Check if rollout files exist. If not, generate them via the simulator.
        Uses a uniformly random discrete policy — sufficient for Breakout.
        Each rollout file spans multiple internal lives via auto-reset until
        max_steps_per_rollout frames are collected.
        """
        os.makedirs(self.data_root, exist_ok=True)

        split_counts = {
            "train": self.preprocessing.num_train_rollouts,
            "val": self.preprocessing.num_val_rollouts,
            "test": self.preprocessing.num_test_rollouts,
        }

        for split in self.splits:
            split_dir = os.path.join(self.data_root, split)
            os.makedirs(split_dir, exist_ok=True)

            existing = len(glob.glob(os.path.join(split_dir, "*.npz")))
            target = split_counts.get(split, 0)

            if existing >= target:
                print(
                    f"Breakout dataset split '{split}' already has {existing} "
                    f"rollouts (target: {target}). Skipping."
                )
                continue

            print(f"Collecting {target - existing} rollouts for split '{split}' using Breakout...")

            env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

            # Unique base seed per split so train/val/test never overlap
            split_seed_offset = {"train": 0, "val": 100_000, "test": 200_000}.get(split, 0)

            for idx in range(existing, target):
                observations = []
                actions = []
                rewards = []
                dones = []

                # Unique seed per rollout -> different initial random state
                rollout_seed = split_seed_offset + idx
                obs, _ = env.reset(seed=rollout_seed)

                # FIRE once to launch the ball after reset
                obs, _, _, _, _ = env.step(1)

                # Save raw RGB — grayscale/resize applied on-the-fly in __getitem__
                observations.append(obs)

                step_count = 0

                while step_count < self.preprocessing.max_steps_per_rollout:
                    action = env.action_space.sample()

                    step_res = env.step(action)
                    if len(step_res) == 5:
                        next_obs, reward, terminated, truncated, info = step_res
                        episode_done = terminated or truncated
                    else:
                        next_obs, reward, episode_done, info = step_res

                    observations.append(next_obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(episode_done)

                    step_count += 1

                    # Auto-reset: when a life/episode ends, reset and keep collecting
                    # into the same rollout file until max_steps_per_rollout is reached
                    if episode_done and step_count < self.preprocessing.max_steps_per_rollout:
                        obs, _ = env.reset()
                        # FIRE to launch ball on new life
                        obs, _, _, _, _ = env.step(1)
                        # Replace last obs with post-reset frame for continuity
                        observations[-1] = obs

                # Trim last observation to match action/reward/done length
                observations = np.array(observations[:-1])
                actions = np.array(actions, dtype=np.int32)
                rewards = np.array(rewards, dtype=np.float32)
                dones = np.array(dones, dtype=bool)

                save_path = os.path.join(split_dir, f"rollout_{idx}.npz")
                np.savez_compressed(
                    save_path,
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                )
                print(f"Saved {save_path} ({len(observations)} frames)")

            env.close()

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert (H, W, 3) RGB frame to (H, W) grayscale using luminance weights."""
        return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    def setup(self, stage: Optional[str] = None) -> None:
        self.datasets = {}
        for split in self.splits:
            split_dir = os.path.join(self.data_root, split)
            image_size = tuple(self.transform.image_size)
            self.datasets[split] = BreakoutDataset(
                split_dir=split_dir,
                image_size=image_size,
                grayscale=self.transform.grayscale,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        split_key = "val" if "val" in self.datasets else "train"
        return DataLoader(
            self.datasets[split_key],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        split_key = "test" if "test" in self.datasets else "train"
        return DataLoader(
            self.datasets[split_key],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
