import os

from torch.utils.data import Dataset

from wiskers.datasets.clevrer_utils import (
    build_frame_count_json,
    download_qa,
    download_videos,
)


class Clevrer(Dataset):
    """
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/
    """

    def __init__(self, data_dir: str, split: str, chunk_len: int):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.chunk_len = chunk_len
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.video_root = os.path.join(self.data_dir, "videos")
        self.qa_root = os.path.join(self.data_dir, "question_answer")
        self.video_dir = os.path.join(self.video_root, self.split)
        self.json_path = os.path.join(self.video_root, f"{self.split}.json")

    def prepare_data(self):
        if os.path.exists(self.json_path):
            print(f"Skipping {self.split}, already processed: {self.json_path}")
            return

        build_frame_count_json(self.video_dir, self.json_path)

    def download_all(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Download JSON Question_Answer
        download_qa(self.qa_root, self.split)

        # Downlod Zip video and Unzip them
        download_videos(self.video_root, self.split)
