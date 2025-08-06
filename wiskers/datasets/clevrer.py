import os

import lightning as L
from torch.utils.data import DataLoader

import wiskers.datasets.clevrer_dataset as data

from wiskers.datasets.clevrer_utils import (
    extract_video_chunks_to_numpy,
    build_frame_count_json,
    download_qa,
    download_videos,
)


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
        self.train_dataset = data.Clevrer(self.data_dir, "train", 4)
        self.test_dataset = data.Clevrer(self.data_dir, "test", 4)
        self.val_dataset = data.Clevrer(self.data_dir, "valid", 4)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for split in ["train", "test", "valid"]:
            video_raw_root = os.path.join(self.data_dir, "video_raw")
            qa_root = os.path.join(self.data_dir, "question_answer")
            video_dir = os.path.join(video_raw_root, split)
            json_path = os.path.join(video_raw_root, f"{split}.json")

            # Download JSON Question_Answer
            download_qa(qa_root, split)

            # Downlod Zip video and Unzip them
            download_videos(video_raw_root, split)

            # prepare video
            # extract_video_chunks_to_numpy

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
