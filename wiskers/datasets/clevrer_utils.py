import json
import os
import zipfile
from typing import Optional
from urllib.request import Request, urlopen, urlretrieve

import cv2
from tqdm import tqdm


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


def download_videos(video_dir: str, split: str):
    os.makedirs(video_dir, exist_ok=True)
    url_path, local_path = VIDEO_URLS[split]
    if url_path and local_path:
        local_path = os.path.join(video_dir, local_path)
        if not os.path.exists(local_path):
            size_mb = get_file_size(url_path) / (1024 * 1024)
            print(
                f"Downloading CLEVRER Video ({split}) of size {size_mb:.2f} MB to {local_path}..."
            )
            urlretrieve(url_path, local_path)
        else:
            print(f"CLEVRER Video ({split}) already downloaded.")
    else:
        print(f"CLEVRER Video ({split}) not specified...")

    raw_video_dir = os.path.join(video_dir, split)
    if not os.path.exists(raw_video_dir):
        os.makedirs(raw_video_dir, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            print(f"CLEVRER Unzip Video ({split}).")
            zip_ref.extractall(raw_video_dir)
    else:
        print(f"CLEVRER Video ({split}) already unzipped.")


def download_qa(qa_dir: str, split: str):
    os.makedirs(qa_dir, exist_ok=True)
    url_path, local_path = QA_URLS[split]
    if url_path and local_path:
        local_path = os.path.join(qa_dir, local_path)
        if not os.path.exists(local_path):
            size_mb = get_file_size(url_path) / (1024 * 1024)
            print(
                f"Downloading CLEVRER Question-Answer ({split}) of size {size_mb:.2f} MB to {local_path}..."
            )
            urlretrieve(url_path, local_path)
        else:
            print(f"CLEVRER Question-Answer ({split}) already downloaded.")
    else:
        print(f"CLEVRER Question-Answer ({split}) not specified...")


def build_frame_count_json(video_dir: str, json_path: str):
    video_paths = get_all_video_paths(video_dir)
    print(f"Num videos: {len(video_paths)}")
    print(f"Example video: {video_paths[0]}")

    # Track progress of frame counting
    frame_map = {}
    for path in tqdm(video_paths, desc="Counting frames"):
        try:
            count = frame_count_from_video(path)
            rel_path = os.path.relpath(path, video_dir)
            frame_map[rel_path] = count
        except Exception as e:
            print(f"Error reading {path}: {e}")

    with open(json_path, "w") as f:
        json.dump(frame_map, f, indent=2)

    total_frames = sum(frame_map.values())
    print(
        f"Saved frame count mapping to: {json_path} ({len(frame_map)} videos, {total_frames} frames)"
    )


def get_all_video_paths(root_dir: str) -> list[str]:
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
