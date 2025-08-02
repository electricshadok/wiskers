import os

import lightning as L

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
        self.train = data.Clevrer(self.data_dir, "train", 4)
        self.test = data.Clevrer(self.data_dir, "test", 4)
        self.valid = data.Clevrer(self.data_dir, "valid", 4)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.train.download_all()
        self.train.prepare_data()

        self.test.download_all()
        self.test.prepare_data()

        self.valid.download_all()
        self.valid.prepare_data()
