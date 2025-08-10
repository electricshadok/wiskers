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

    def __init__(self, index_path: str):
        super().__init__()
        self.index_path = index_path
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, "r") as f:
            data = json.load(f)

        self.samples = data.get("samples", [])
        if not self.samples:
            raise ValueError(f"No samples found in {index_path}")

        self.root_dir = os.path.dirname(index_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path = self.samples[idx]
        abs_path = os.path.join(self.root_dir, rel_path)

        arr = np.load(abs_path)  # shape: (T, H, W, C)
        tensor = torch.from_numpy(arr).float() / 255.0  # normalize to [0,1]
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        return tensor, rel_path
