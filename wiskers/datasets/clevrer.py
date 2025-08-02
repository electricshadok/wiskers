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
        self.cleverer = data.Clevrer(self.data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.cleverer.download_all()
