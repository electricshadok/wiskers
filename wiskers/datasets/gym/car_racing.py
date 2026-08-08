import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset


try:
    import gymnasium as gym
except ImportError:
    import gym


@dataclass
class PreprocessingConfig:
    num_train_rollouts: int = 10
    num_val_rollouts: int = 2
    num_test_rollouts: int = 2
    max_steps_per_rollout: int = 200


@dataclass
class TransformConfig:
    image_size: List[int] = field(default_factory=lambda: [64, 64])


class CarRacingDataset(Dataset):
    """
    Yields transition step dictionaries for world model training:
    (image, action) -> next_image
    """
    def __init__(
        self,
        split_dir: str,
        image_size: Tuple[int, int],
    ):
        super().__init__()
        self.split_dir = split_dir
        self.image_size = image_size
        self.file_paths = sorted(glob.glob(os.path.join(split_dir, "*.npz")))

        if not self.file_paths:
            raise FileNotFoundError(f"No rollout files (.npz) found in {split_dir}")

        self.transitions = []

        # Build index of all transition steps across all rollout files
        for file_path in self.file_paths:
            with np.load(file_path) as data:
                seq_len = len(data["observations"])
            # With seq_len frames, we can form seq_len - 1 transition steps:
            # (frame_t, action_t) -> frame_t+1
            for step_idx in range(seq_len - 1):
                self.transitions.append((file_path, step_idx))

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_path, step_idx = self.transitions[idx]
        with np.load(file_path) as data:
            frame_t = data["observations"][step_idx]       # (H, W, C)
            frame_next = data["observations"][step_idx + 1] # (H, W, C)
            action_t = data["actions"][step_idx]           # (3,)
            done_t = data["dones"][step_idx]               # scalar bool

        # Helper to convert a single frame to (C, H, W) normalized [0, 1] tensor resized to target size
        def process_frame(frame: np.ndarray) -> torch.Tensor:
            frame_t = torch.from_numpy(frame).float() / 255.0  # (H, W, C)
            frame_t = frame_t.permute(2, 0, 1)                  # (C, H, W)
            frame_t = TF.resize(frame_t, self.image_size, antialias=True)
            return frame_t

        media = process_frame(frame_t)
        media_next = process_frame(frame_next)
        action = torch.from_numpy(action_t).float()
        done = torch.tensor([float(done_t)]).float()

        return {
            "media": media,
            "action": action,
            "media_next": media_next,
            "done": done,
        }


class CarRacingDataModule(L.LightningDataModule):
    """
    DataModule for CarRacing-v3 Gym dataset preparation and loading.
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
        self.data_root = os.path.join(data_dir, "carracing")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits or ["train", "val", "test"]

        # Parse configs
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
        Check if rollout files exist. If not, generate them by running the simulator.
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
                    f"CarRacing dataset split '{split}' already has {existing} "
                    f"rollouts (target: {target}). Skipping."
                )
                continue

            print(f"Collecting {target - existing} rollouts for split '{split}' using CarRacing...")

            # Initialize environment
            try:
                env = gym.make("CarRacing-v3", render_mode="rgb_array")
            except Exception as e:
                print(f"Error creating Gym CarRacing-v3 environment: {e}")
                raise e

            for idx in range(existing, target):
                observations = []
                actions = []
                rewards = []
                dones = []

                # Reset environment
                reset_res = env.reset()
                if isinstance(reset_res, tuple) and len(reset_res) == 2:
                    obs, info = reset_res
                else:
                    obs = reset_res

                observations.append(obs)

                step_count = 0
                done = False

                while not done and step_count < self.preprocessing.max_steps_per_rollout:
                    action = env.action_space.sample()  # Continuous actions: (steering, gas, brake)

                    step_res = env.step(action)
                    if len(step_res) == 5:
                        next_obs, reward, terminated, truncated, info = step_res
                        done = terminated or truncated
                    else:
                        next_obs, reward, done, info = step_res

                    observations.append(next_obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)

                    step_count += 1

                # Trim last observation to match action/reward/done sequence length
                observations = np.array(observations[:-1], dtype=np.uint8)
                actions = np.array(actions, dtype=np.float32)
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

    def setup(self, stage: Optional[str] = None) -> None:
        self.datasets = {}
        for split in self.splits:
            split_dir = os.path.join(self.data_root, split)
            image_size = tuple(self.transform.image_size)
            self.datasets[split] = CarRacingDataset(
                split_dir=split_dir,
                image_size=image_size,
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
        split_key = "val" if "val" in self.datasets else ("valid" if "valid" in self.datasets else "train")
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
