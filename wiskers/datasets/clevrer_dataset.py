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

    def prepare_data(self):
        splits = ["train", "valid", "test"]
        video_root = os.path.join(self.data_dir, "videos")

        for split in splits:
            video_dir = os.path.join(video_root, split)
            json_path = os.path.join(video_root, f"{split}.json")

            if os.path.exists(json_path):
                print(f"Skipping {split}, already processed: {json_path}")
                continue

            build_frame_count_json(video_dir, json_path)

    def download_all(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Download JSON Question_Answer
        qa_dir = os.path.join(self.data_dir, "question_answer")
        download_qa(qa_dir)

        # Downlod Zip video and Unzip them
        video_dir = os.path.join(self.data_dir, "videos")
        download_videos(video_dir)
