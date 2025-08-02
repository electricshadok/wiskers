import os
from typing import Optional
from urllib.request import Request, urlopen

import cv2

QA_URLS = {
    "train": (
        "http://data.csail.mit.edu/clevrer/questions/train.json",
        "train.json",
    ),
    "valid": (
        "http://data.csail.mit.edu/clevrer/questions/validation.json",
        "valid.json",
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
        "video_valid.zip",
    ),
    "test": (
        "http://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
        "video_test.zip",
    ),
}


def get_all_videos(root_dir: str) -> list[str]:
    video_paths = []
    for group_dir in os.listdir(root_dir):
        group_path = os.path.join(root_dir, group_dir)
        if not os.path.isdir(group_path):
            continue
        for file in os.listdir(group_path):
            if file.endswith(".mp4"):
                video_path = os.path.join(group_path, file)  # ← more robust
                video_paths.append(video_path)
    return video_paths


def frame_count_from_video(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)  # type: ignore
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore
    cap.release()
    return num_frames


def get_file_size(url: str, timeout: float = 10.0) -> Optional[int]:
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=timeout) as response:
            size = response.getheader("Content-Length")
            return int(size) if size is not None else None
    except Exception as e:
        print(f"Warning: Failed to get file size from {url} - {e}")
        return None
