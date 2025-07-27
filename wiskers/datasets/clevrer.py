import os
import zipfile
from urllib.request import Request, urlopen, urlretrieve

import lightning as L


def get_file_size(url):
    request = Request(url, method="HEAD")
    with urlopen(request) as response:
        size = response.getheader("Content-Length")
        if size is not None:
            return int(size)
        else:
            return None


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
        "train": (
            "http://data.csail.mit.edu/clevrer/questions/train.json",
            "train.json",
        ),
        "valid": (
            "http://data.csail.mit.edu/clevrer/questions/validation.json",
            "validation.json",
        ),
        "test": ("http://data.csail.mit.edu/clevrer/questions/test.json", "test.json"),
    }

    VIDEO_URLS = {
        "train": (
            "http://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
            "video_train.zip",
        ),
        "valid": (
            "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
            "video_validation.zip",
        ),
        "test": (
            "http://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
            "video_test.zip",
        ),
    }

    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Download JSON Question_Answer
        qa_dir = os.path.join(self.data_dir, "question_answer")
        os.makedirs(qa_dir, exist_ok=True)
        for setname, url_n_local in self.QA_URLS.items():
            url_path, local_path = url_n_local
            if url_path and local_path:
                local_path = os.path.join(qa_dir, local_path)
                if not os.path.exists(local_path):
                    size_mb = get_file_size(url_path) / (1024 * 1024)
                    print(
                        f"Downloading CLEVRER Question-Answer ({setname}) of size {size_mb:.2f} MB to {local_path}..."
                    )
                    urlretrieve(url_path, local_path)
                else:
                    print(f"CLEVRER Question-Answer ({setname}) already downloaded.")
            else:
                print(f"CLEVRER Question-Answer ({setname}) not specified...")

        # Downlod Zip video and Unzip them
        video_dir = os.path.join(self.data_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        for setname, url_n_local in self.VIDEO_URLS.items():
            url_path, local_path = url_n_local
            if url_path and local_path:
                local_path = os.path.join(video_dir, local_path)
                if not os.path.exists(local_path):
                    size_mb = get_file_size(url_path) / (1024 * 1024)
                    print(
                        f"Downloading CLEVRER Video ({setname}) of size {size_mb:.2f} MB to {local_path}..."
                    )
                    urlretrieve(url_path, local_path)
                else:
                    print(f"CLEVRER Video ({setname}) already downloaded.")
            else:
                print(f"CLEVRER Video ({setname}) not specified...")

            raw_video_dir = os.path.join(video_dir, setname)
            if not os.path.exists(raw_video_dir):
                os.makedirs(raw_video_dir, exist_ok=True)
                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    print(f"CLEVRER Unzip Video ({setname}).")
                    zip_ref.extractall(raw_video_dir)
            else:
                print(f"CLEVRER Video ({setname}) already unzipped.")
