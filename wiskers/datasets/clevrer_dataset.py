import json
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
        self.cache_dir = os.path.join(self.data_dir, "cache", self.split)
        self.video_root = os.path.join(self.data_dir, "videos")
        self.qa_root = os.path.join(self.data_dir, "question_answer")
        self.video_dir = os.path.join(self.video_root, self.split)
        self.json_path = os.path.join(self.video_root, f"{self.split}.json")

        # Store list of relative video paths
        self.video_list = []

        # Build index: (video_idx, start_frame)
        self.index = []

    def __len__(self):
        return len(self.index)

    def prepare_data(self):
        # Create cached json
        if not os.path.exists(self.json_path):
            build_frame_count_json(self.video_dir, self.json_path)
        else:
            print(f"Skipping {self.split}, already processed: {self.json_path}")

        # Prepare indices for samples
        with open(self.json_path, "r") as f:
            frame_counts = json.load(f)

        self.video_list = list(frame_counts.keys())
        self.index = []
        for video_idx, rel_path in enumerate(self.video_list):
            num_frames = frame_counts[rel_path]
            num_chunks = num_frames // self.chunk_len
            for i in range(num_chunks):
                self.index.append((video_idx, i * self.chunk_len))

        print(f"CLEVRER {len(self.index)} samples loaded from {self.json_path}")

    def download_all(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Download JSON Question_Answer
        download_qa(self.qa_root, self.split)

        # Downlod Zip video and Unzip them
        download_videos(self.video_root, self.split)
