import os

import lightning as L
from torch.utils.data import DataLoader

import wiskers.datasets.clevrer_dataset as data


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
        self.train_dataset.download_all()
        self.train_dataset.prepare_data()

        self.test_dataset.download_all()
        self.test_dataset.prepare_data()

        self.val_dataset.download_all()
        self.val_dataset.prepare_data()

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
