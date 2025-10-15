import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ClevrerQAHelper:
    """
    Helper for Question-Answer data from CLEVRER
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/
    """

    def __init__(self, qa_path: str):
        self.qa_path = qa_path
        if not os.path.exists(qa_path):
            raise FileNotFoundError(f"QA file not found: {qa_path}")

        with open(qa_path, "r") as f:
            qa_json = json.load(f)

        self.scene_index_2_qa = {qa["scene_index"]: qa["questions"] for qa in qa_json}

    def get_scene_indices(self) -> list[int]:
        return list(self.scene_index_2_qa.keys())

    def get_questions(self, scene_index: int) -> list[dict]:
        return self.scene_index_2_qa[scene_index]

    def __len__(self) -> int:
        return len(self.scene_index_2_qa)


class ClevrerAnnotationHelper:
    """
    Helper for Annotation index data
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/
    """

    def __init__(self, annotation_index_path: str):
        self.annotation_index_path = annotation_index_path
        if not os.path.exists(annotation_index_path):
            raise FileNotFoundError(f"QA file not found: {annotation_index_path}")

        with open(annotation_index_path, "r") as f:
            annotation_json = json.load(f)

        self.scene_index_2_annotation = {}
        for sample in annotation_json["samples"]:
            scene_index = sample["scene_index"]
            self.scene_index_2_annotation[scene_index] = sample

    def get_scene_indices(self) -> list[int]:
        return list(self.scene_index_2_annotation.keys())

    def get_annotations(self, scene_index: int) -> dict:
        return self.scene_index_2_annotation[scene_index]

    def __len__(self) -> int:
        return len(self.scene_index_2_annotation)


class ClevrerMediaHelper:
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
        resize: tuple[int, int] | None,
    ):
        # Load video index
        if not os.path.exists(video_index_path):
            raise FileNotFoundError(f"Index file not found: {video_index_path}")

        with open(video_index_path, "r") as f:
            video_data = json.load(f)

        self.samples = video_data["samples"]
        # chunk_size is the number of frames in each .npy chunk
        self.chunk_size = video_data["chunk_size"]
        self.video_root_dir = os.path.dirname(video_index_path)

        if not self.samples:
            raise ValueError(f"No samples found in {video_index_path}")

        self.resize = resize

    def load_and_resize_sample(self, sample_idx: int):
        """
        Loads a video chunk from disk and optionally resizes it.
        Returns:
            torch.Tensor: Tensor of shape (T, C, H, W)
        """
        rel_path = self.samples[sample_idx]
        abs_path = os.path.join(self.video_root_dir, rel_path)
        arr = np.load(abs_path)  # (T, H, W, C)
        video = torch.from_numpy(arr).float() / 255.0  # normalize to [0,1]
        video = video.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

        if self.resize:
            video = F.interpolate(
                video, size=self.resize, mode="bilinear", align_corners=False
            )

        return video

    def get_sample_path(self, sample_idx: int) -> str:
        rel_path = self.samples[sample_idx]
        abs_path = os.path.join(self.video_root_dir, rel_path)
        return abs_path

    def get_scene_index(self, sample_idx: int) -> int:
        # rel_path example: video_03903/video_03903_chunk0002.npy
        rel_path = self.samples[sample_idx]
        folder_name = rel_path.split("/")[0]
        return int(folder_name.split("_")[1])

    def num_images(self) -> int:
        return len(self.samples) * self.chunk_size

    def num_videos(self) -> int:
        return len(self.samples)

    def get_scene_indices(self) -> list[int]:
        # a sample can refer to the same scene index
        scene_indices = set()
        for sample_idx in range(self.num_videos()):
            scene_index = self.get_scene_index(sample_idx)
            scene_indices.add(scene_index)

        return list(scene_indices)


class ClevrerMedia(Dataset):
    """
    Base class that initializes media, annotation, and QA helpers.
    """

    def __init__(
        self,
        video_index_path: str,
        annotation_index_path: Optional[str],
        qa_path: Optional[str],
        resize: tuple[int, int] | None,
    ):
        super().__init__()
        self.media_helper = ClevrerMediaHelper(video_index_path, resize)
        self.annotation_helper = (
            ClevrerAnnotationHelper(annotation_index_path)
            if annotation_index_path
            else None
        )
        self.qa_helper = ClevrerQAHelper(qa_path) if qa_path else None

        self._check_scene_index_consistency()

    def _check_scene_index_consistency(self):
        """Ensure all helpers have matching scene indices (using sets)."""
        media_set = set(self.media_helper.get_scene_indices())

        if self.annotation_helper:
            ann_set = set(self.annotation_helper.get_scene_indices())
            if media_set != ann_set:
                diff_media = sorted(media_set - ann_set)
                diff_ann = sorted(ann_set - media_set)
                raise ValueError(
                    "[Scene Index Mismatch] Media vs Annotation\n"
                    f"  → Missing in Annotation: {diff_media[:5]}{'...' if len(diff_media) > 5 else ''}\n"
                    f"  → Missing in Media: {diff_ann[:5]}{'...' if len(diff_ann) > 5 else ''}\n"
                    f"  (Media: {len(media_set)} | Annotation: {len(ann_set)})"
                )

        if self.qa_helper:
            qa_set = set(self.qa_helper.get_scene_indices())
            if media_set != qa_set:
                diff_media = sorted(media_set - qa_set)
                diff_qa = sorted(qa_set - media_set)
                raise ValueError(
                    "[Scene Index Mismatch] Media vs QA\n"
                    f"  → Missing in QA: {diff_media[:5]}{'...' if len(diff_media) > 5 else ''}\n"
                    f"  → Missing in Media: {diff_qa[:5]}{'...' if len(diff_qa) > 5 else ''}\n"
                    f"  (Media: {len(media_set)} | QA: {len(qa_set)})"
                )

        print(
            f"Scene index check passed — {len(media_set)} unique scenes consistent across all helpers."
        )


class ClevrerVideo(ClevrerMedia):
    """
    Loads full video chunks as tensors.

    Returns:
        - video: torch.Tensor of shape (T, C, H, W)
        - scene_index: scene sunique identifier
    """

    def __len__(self):
        return self.media_helper.num_videos()

    def __getitem__(self, idx: int):
        video = self.media_helper.load_and_resize_sample(idx)
        scene_index = self.media_helper.get_scene_index(idx)
        return {"media": video, "scene_index": scene_index}


class ClevrerImage(ClevrerMedia):
    """
    Loads individual frames (as images) from CLEVRER chunks.

    Returns:
        - image: torch.Tensor of shape (C, H, W)
        - scene_index: scene sunique identifier
    """

    def __len__(self):
        return self.media_helper.num_images()

    def __getitem__(self, idx: int):
        sample_idx = idx // self.media_helper.chunk_size
        frame_idx = idx % self.media_helper.chunk_size

        video = self.media_helper.load_and_resize_sample(sample_idx)

        if frame_idx >= video.shape[0]:
            sample_path = self.media_helper.get_sample_path(sample_idx)
            raise IndexError(f"Frame {frame_idx} out of bounds in video {sample_path}")

        image = video[frame_idx]  # (C, H, W)
        scene_index = self.media_helper.get_scene_index(sample_idx)
        return {"media": image, "scene_index": scene_index}
