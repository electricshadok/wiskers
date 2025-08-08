import os

from torch.utils.data import Dataset


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
