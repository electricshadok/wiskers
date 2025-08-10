import os

import lightning as L
from torch.utils.data import DataLoader

import wiskers.datasets.clevrer_dataset as data
from wiskers.datasets.clevrer_utils import (
    download_qa,
    download_videos,
    prepare_and_extract_clevrer_videos,
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

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        chunk_size: int,
        stride: int,
        resize: tuple[int, int] | None,
    ):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.stride = stride
        self.resize = resize
        self.index_paths = {}

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for split in ["train", "test", "valid"]:
            video_raw_root = os.path.join(self.data_dir, "video_raw")
            qa_root = os.path.join(self.data_dir, "question_answer")
            raw_video_dir = os.path.join(video_raw_root, split)
            processed_video_dir = os.path.join(self.data_dir, "video", split)

            # Download JSON Question_Answer
            download_qa(qa_root, split)

            # Downlod Zip video and Unzip them
            download_videos(video_raw_root, split)

            index_path = prepare_and_extract_clevrer_videos(
                raw_video_dir=raw_video_dir,
                processed_video_dir=processed_video_dir,
                chunk_size=self.chunk_size,
                stride=self.stride,
                resize=self.resize,
                limit=None,
                index_filename="index.json",
            )
            print(f"CLEVRER Index ({split}) {index_path} available")

            self.index_paths[split] = index_path

    def train_dataloader(self):
        train_dataset = data.Clevrer(self.index_paths["train"])
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        val_dataset = data.Clevrer(self.index_paths["valid"])
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        test_dataset = data.Clevrer(self.index_paths["test"])
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
