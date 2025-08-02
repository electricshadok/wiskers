import json
import os

import lightning as L
from tqdm import tqdm

import wiskers.datasets.clevrer_dataset as data
from wiskers.datasets.clevrer_utils import frame_count_from_video, get_all_videos


class CLEVRER(L.LightningDataModule):
    """
    LightningDataModule for CLEVRER dataset.
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/

    Args:
        data_dir (str): Path to the directory containing the CIFAR-10 dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """

    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.cleverer = data.Clevrer(self.data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.cleverer.download_all()

        splits = ["train", "valid", "test"]
        video_root = os.path.join(self.data_dir, "videos")

        for split in splits:
            video_dir = os.path.join(video_root, split)
            json_path = os.path.join(video_root, f"{split}.json")

            if os.path.exists(json_path):
                print(f"Skipping {split}, already processed: {json_path}")
                continue

            video_paths = get_all_videos(video_dir)
            print(f"Num videos: {len(video_paths)}")
            print(f"Example video: {video_paths[0]}")

            # Track progress of frame counting
            frame_map = {}
            for path in tqdm(video_paths, desc="Counting frames"):
                try:
                    count = frame_count_from_video(path)
                    frame_map[path] = count
                except Exception as e:
                    print(f"Error reading {path}: {e}")

            with open(json_path, "w") as f:
                json.dump(frame_map, f, indent=2)

            total_frames = sum(frame_map.values())
            print(
                f"Saved frame count mapping to: {json_path} ({len(frame_map)} videos, {total_frames} frames)"
            )
