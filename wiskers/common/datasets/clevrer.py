import os
from urllib.request import urlretrieve

import lightning as L


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

    QA_URLS = {
        "train": (None, None),
        "valid": ("http://data.csail.mit.edu/clevrer/questions/validation.json", "validation.json"),
        "test": (None, None),
    }

    VIDEO_URLS = {"train": (None, None), "valid": (None, None), "test": (None, None)}

    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Downloading Question-Answer
        print("Downloading CLEVRER Question-Answer...")
        qa_dir = os.path.join(self.data_dir, "question_answer")
        os.makedirs(qa_dir, exist_ok=True)
        for setname, url_n_local in self.QA_URLS.items():
            print(f"Downloading CLEVRER Question-Answer{setname}...")
            url_path, local_path = url_n_local
            if url_path and local_path and not os.path.exists(local_path):
                local_path = os.path.join(qa_dir, "validation.json")
                urlretrieve(url_path, local_path)


        """
        # Download and extract CLEVRER if not already present
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            zip_path = os.path.join(self.data_dir, "clevrer.zip")
            print("Downloading CLEVRER...")
            urlretrieve(CLEVRER_URL, zip_path)
            print("Extracting...")
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            os.remove(zip_path)
        """
