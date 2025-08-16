import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Clevrer(Dataset):
    """
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/

    Simple CLEVRER Dataset loader.
    Expects an index.json with a "samples" list of relative .npy paths.
    Returns: (tensor, rel_path)
    """

    def __init__(self, video_index_path: str, qa_path: str):
        super().__init__()
        # Prepare video samples
        if not os.path.exists(video_index_path):
            raise FileNotFoundError(f"Index file not found: {video_index_path}")

        with open(video_index_path, "r") as f:
            data = json.load(f)

        self.samples = data.get("samples", [])
        if not self.samples:
            raise ValueError(f"No samples found in {video_index_path}")

        self.video_root_dir = os.path.dirname(video_index_path)

        # Prepare question-answer data
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"Index file not found: {qa_path}")

        with open(qa_path, "r") as f:
            qa_json = json.load(f)

        self.scene_index_2_qa_mapping = {}
        for qa_data in qa_json:
            scene_index = qa_data["scene_index"]
            self.scene_index_2_qa_mapping[scene_index] = qa_data["questions"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Get the vieo sample
        rel_path = self.samples[idx]  # e.g. video_03903/video_03903_chunk0002.npy
        abs_path = os.path.join(self.video_root_dir, rel_path)

        arr = np.load(abs_path)  # shape: (T, H, W, C)
        tensor = torch.from_numpy(arr).float() / 255.0  # normalize to [0,1]
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        # Get scene_index
        # folder_name = rel_path.split("/")[0]
        # scene_index = int(folder_name.split("_")[1])  # 3903

        # Get the question_answers
        # TODO: qa_data = self.scene_index_2_qa_mapping[scene_index]

        return tensor, rel_path
