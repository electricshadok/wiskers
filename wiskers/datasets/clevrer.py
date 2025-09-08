import os
from dataclasses import dataclass, field
from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader

import wiskers.datasets.clevrer_utils.clevrer_datasets as datasets
from wiskers.datasets.clevrer_utils.prepare_clevrer import (
    download_qa,
    download_videos,
    prepare_and_extract_clevrer_videos,
)


@dataclass
class PreprocessingConfig:
    chunk_size: int = 16
    stride: int = 16
    resize: List[int] = field(default_factory=lambda: [160, 240])
    limit: Optional[int] = None  # Optional limit on number of videos processed


class ClevrerBase(L.LightningDataModule):
    """
    Base LightningDataModule for CLEVRER dataset variants (video/image).

    Handles shared logic like downloading, extraction, and QA/video index resolution.

    Args:
        data_dir (str): Path to the root data directory.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of subprocesses for data loading.
        chunk_size (int): Number of frames per video chunk.
        stride (int): Step size between chunks (can be < chunk_size for overlap).
        resize (tuple or None): Resize dimensions (H, W) for raw video preprocessing.
        image_size (tuple or None): Resize dimensions for model input (used at dataloader level).
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        image_size: tuple[int, int] | None,
        preprocessing: PreprocessingConfig,
        splits: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing = preprocessing
        self.image_size = image_size
        self.qa_index_paths = {}
        self.video_index_paths = {}
        self.splits = splits or ["train", "valid", "test"]

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for split in self.splits:
            video_raw_root = os.path.join(self.data_dir, "video_raw")
            qa_root = os.path.join(self.data_dir, "question_answer")
            processed_video_dir = os.path.join(self.data_dir, "video", split)

            # Download QA JSON
            qa_index_path = download_qa(qa_root, split)
            self.qa_index_paths[split] = qa_index_path
            print(f"CLEVRER QA Index ({split}) {qa_index_path} available")

            # Download & extract videos
            raw_video_dir = download_videos(video_raw_root, split)

            video_index_path = prepare_and_extract_clevrer_videos(
                raw_video_dir=raw_video_dir,
                processed_video_dir=processed_video_dir,
                chunk_size=self.preprocessing.chunk_size,
                stride=self.preprocessing.stride,
                resize=self.preprocessing.resize,
                limit=self.preprocessing.limit,
                index_filename="index.json",
            )
            self.video_index_paths[split] = video_index_path
            print(f"CLEVRER Video Index ({split}) {video_index_path} available")

    def _make_dataloader(self, split: str, dataset_cls):
        dataset = dataset_cls(
            self.video_index_paths[split],
            self.qa_index_paths[split],
            self.image_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class ClevrerVideo(ClevrerBase):
    """
    LightningDataModule for loading CLEVRER video chunks as 4D tensors (T, C, H, W).
    """

    def train_dataloader(self):
        return self._make_dataloader("train", datasets.ClevrerVideo)

    def val_dataloader(self):
        return self._make_dataloader("valid", datasets.ClevrerVideo)

    def test_dataloader(self):
        return self._make_dataloader("test", datasets.ClevrerVideo)


class ClevrerImage(ClevrerBase):
    """
    LightningDataModule for loading individual frames from CLEVRER video chunks.
    Each frame is returned as a 3D tensor (C, H, W).
    """

    def train_dataloader(self):
        return self._make_dataloader("train", datasets.ClevrerImage)

    def val_dataloader(self):
        return self._make_dataloader("valid", datasets.ClevrerImage)

    def test_dataloader(self):
        return self._make_dataloader("test", datasets.ClevrerImage)
