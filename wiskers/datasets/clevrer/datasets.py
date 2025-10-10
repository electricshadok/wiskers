import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ClevrerMedia(Dataset):
    """
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/


    Prepared CLEVRER video data is stored as sequences of frames in `.npy` files, where each file represents
    a chunk of consecutive frames (with length defined by `chunk_size`). For example:
        video_03903/video_03903_chunk0002.npy  # Shape: (chunk_size, H, W, 3)

    Simple CLEVRER Dataset loader.
    Expects an index.json with a "samples" list of relative .npy paths.
    Returns:
        - video: Tensor of shape (T, C, H, W) if using `ClevrerVideo`
        - image: Tensor of shape (C, H, W) if using `ClevrerImage`
    """

    def __init__(
        self,
        video_index_path: str,
        qa_path: str,
        resize: tuple[int, int] | None,
    ):
        super().__init__()

        # Load video index
        if not os.path.exists(video_index_path):
            raise FileNotFoundError(f"Index file not found: {video_index_path}")

        with open(video_index_path, "r") as f:
            data = json.load(f)

        self.samples = data["samples"]
        self.chunk_size = data["chunk_size"]  # number of frames in each .npy chunk
        self.video_root_dir = os.path.dirname(video_index_path)

        if not self.samples:
            raise ValueError(f"No samples found in {video_index_path}")

        # Load QA mapping
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA file not found: {qa_path}")

        with open(qa_path, "r") as f:
            qa_json = json.load(f)

        self.scene_index_2_qa_mapping = {
            qa["scene_index"]: qa["questions"] for qa in qa_json
        }

        self.resize = resize

    def load_and_resize_video(self, rel_path: str):
        """
        Loads a video chunk from disk and optionally resizes it.
        Returns:
            torch.Tensor: Tensor of shape (T, C, H, W)
        """
        abs_path = os.path.join(self.video_root_dir, rel_path)
        arr = np.load(abs_path)  # (T, H, W, C)
        video = torch.from_numpy(arr).float() / 255.0  # normalize to [0,1]
        video = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        if self.resize:
            video = F.interpolate(
                video, size=self.resize, mode="bilinear", align_corners=False
            )

        return video

    def extract_scene_index(self, rel_path: str) -> int:
        # rel_path example: video_03903/video_03903_chunk0002.npy
        folder_name = rel_path.split("/")[0]
        return int(folder_name.split("_")[1])


class ClevrerVideo(ClevrerMedia):
    """
    Loads full video chunks as tensors.

    Returns:
        - video: torch.Tensor of shape (T, C, H, W)
        - rel_path: str
    """

    def __init__(
        self,
        video_index_path: str,
        qa_path: str,
        resize: tuple[int, int] | None,
    ):
        super().__init__(video_index_path, qa_path, resize)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path = self.samples[idx]  # e.g. video_03903/video_03903_chunk0002.npy
        video = self.load_and_resize_video(rel_path)

        # Get the question_answers
        # scene_index = self.extract_scene_index(rel_path)
        # TODO: qa_data = self.scene_index_2_qa_mapping[scene_index]

        return {"media": video, "media_path": rel_path}


class ClevrerImage(ClevrerMedia):
    """
    Loads individual frames (as images) from CLEVRER chunks.

    Returns:
        - image: torch.Tensor of shape (C, H, W)
        - rel_path: str (of the chunk the image came from)
    """

    def __init__(
        self,
        video_index_path: str,
        qa_path: str,
        resize: tuple[int, int] | None,
    ):
        super().__init__(video_index_path, qa_path, resize)

    def __getitem__(self, idx: int):
        chunk_idx = idx // self.chunk_size
        frame_idx = idx % self.chunk_size

        rel_path = self.samples[chunk_idx]
        video = self.load_and_resize_video(rel_path)

        if frame_idx >= video.shape[0]:
            raise IndexError(f"Frame {frame_idx} out of bounds in video {rel_path}")

        image = video[frame_idx]  # (C, H, W)
        return {"media": image, "media_path": rel_path}

    def __len__(self):
        return len(self.samples) * self.chunk_size
