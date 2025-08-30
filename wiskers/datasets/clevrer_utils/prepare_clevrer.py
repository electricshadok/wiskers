import json
import os
import zipfile
from glob import glob
from typing import Optional
from urllib.request import Request, urlopen, urlretrieve

import cv2
import numpy as np
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


def download_videos(video_dir: str, split: str) -> str:
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

    return raw_video_dir


def download_qa(qa_dir: str, split: str) -> str:
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

    return local_path


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


def get_file_size(url: str, timeout: float = 10.0) -> Optional[int]:
    try:
        request = Request(url, method="HEAD")
        with urlopen(request, timeout=timeout) as response:
            size = response.getheader("Content-Length")
            return int(size) if size is not None else None
    except Exception as e:
        print(f"Warning: Failed to get file size from {url} - {e}")
        return None


def extract_video_chunks_to_numpy(
    video_path: str,
    output_dir: str,
    chunk_size: int,
    stride: int = None,
    resize: tuple[int, int] = None,
    prefix: str = None,
) -> int:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    stride = stride or chunk_size
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    prefix = prefix or video_name

    buffer = []
    frame_idx = 0
    chunk_idx = 0
    original_shape = None
    chunk_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_shape = frame.shape
        if resize:
            frame = cv2.resize(frame, resize)

        buffer.append(frame)

        # If we've collected enough for a chunk
        if len(buffer) == chunk_size:
            chunk = np.stack(buffer)
            save_path = os.path.join(output_dir, f"{prefix}_chunk{chunk_idx:04d}.npy")
            np.save(save_path, chunk)
            chunk_idx += 1
            chunk_shape = chunk.shape

            # Move forward by stride
            # If stride < chunk_size, we keep overlapping frames by slicing the buffer
            # Otherwise, if stride >= chunk_size, we clear the buffer entirely (no overlap)
            buffer = buffer[stride:] if stride < chunk_size else []

        frame_idx += 1

    cap.release()
    print(
        f"Saved {chunk_idx} chunks of shape {chunk_shape} / original shape {original_shape} from {video_path} to {output_dir}"  # noqa: E501
    )
    return chunk_idx


def prepare_and_extract_clevrer_videos(
    raw_video_dir: str,
    processed_video_dir: str,
    chunk_size: int = 4,
    stride: int = 4,
    resize: tuple[int, int] | None = None,
    limit: int | None = None,
    index_filename: str = "index.json",
) -> str:
    """
    Preprocess CLEVRER videos by extracting fixed-length chunks and saving them as numpy arrays.

    Args:
        raw_video_dir (str): Path to the directory with raw .mp4 video files.
        processed_video_dir (str): Where to save the extracted chunk numpy arrays.
        chunk_size (int): Number of frames per chunk.
        stride (int): Step between chunks (stride = chunk_size → no overlap).
        resize (tuple or None): Resize (H, W) for frames, or None to keep original size.
        limit (int or None): Limit the number of videos to process.

    TODO: Add multiprocessing support to speed up processing across multiple CPU cores.
    """
    os.makedirs(processed_video_dir, exist_ok=True)
    index_path = os.path.join(processed_video_dir, index_filename)
    if os.path.exists(index_path):
        return index_path

    # Get and sort video paths lexicographically (consistent due to video naming)
    video_paths = sorted(get_all_video_paths(raw_video_dir))

    print(f"{len(video_paths)} videos found in {raw_video_dir}")
    print("Example:", video_paths[0])

    if limit is not None:
        video_paths = video_paths[:limit]

    # Collect all chunk paths (relative to processed_video_dir) if we’re writing an index
    all_rel_paths = []

    for video_path in tqdm(video_paths, desc="Processing videos"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Start processing {video_id} ...")

        output_dir = os.path.join(processed_video_dir, video_id)

        if os.path.exists(output_dir):
            print(f"{video_id} is already processed.")
        else:
            os.makedirs(output_dir, exist_ok=True)

            extract_video_chunks_to_numpy(
                video_path=video_path,
                output_dir=output_dir,
                chunk_size=chunk_size,
                stride=stride,
                resize=resize,
            )

        # After extraction, list chunks we just created and add them (relative) to the index
        chunk_files = sorted(glob(os.path.join(output_dir, "*.npy")))
        rel_files = [os.path.relpath(p, processed_video_dir) for p in chunk_files]
        all_rel_paths.extend(rel_files)

    payload = {
        "root": ".",  # paths are relative to this JSON
        "chunk_size": chunk_size,
        "stride": stride,
        "resize": list(resize) if resize else None,
        "num_samples": len(all_rel_paths),
        "samples": all_rel_paths,
    }
    with open(index_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved index with {len(all_rel_paths)} samples → {index_path}")
    return index_path
