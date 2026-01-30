import json
import multiprocessing
import os
import shutil
import zipfile
from glob import glob
from typing import Optional
from urllib.request import Request, urlopen, urlretrieve

import cv2
import numpy as np
from tqdm import tqdm


def bundle_clevrer_for_upload(processed_root: str, archive_name: str) -> str:
    """
    Package all processed CLEVRER assets under ``processed_root`` into a zip archive.
    """
    print(f"Starting CLEVRER bundle: root={processed_root}, archive={archive_name}")

    # Derive base name without .zip for shutil.make_archive
    base_name, ext = os.path.splitext(archive_name)
    if not base_name:
        base_name = "clevrer_processed"
    archive_base = os.path.join(processed_root, base_name)
    archive_path = f"{archive_base}.zip"

    # Remove existing archive to avoid self-inclusion or stale bundles
    if os.path.exists(archive_path):
        os.remove(archive_path)

    os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
    shutil.make_archive(
        base_name=archive_base,
        format="zip",
        root_dir=processed_root,
    )

    print(f"Created upload bundle at {archive_path}")
    return archive_path


def upload_file_to_gdrive(
    file_path: str, folder_id: str, credentials_path: str
) -> Optional[str]:
    """
    Upload a local file to Google Drive using a service account JSON key.

    Args:
        file_path: Path to the file to upload.
        folder_id: Destination Drive folder ID.
        credentials_path: Path to the service account credentials JSON.
    Returns:
        The uploaded file id, if available.
    """
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Google Drive upload requested but required packages are missing. "
            "Install with `pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib`."
        ) from exc

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for upload: {file_path}")

    print(
        f"Starting Google Drive upload: file={file_path}, folder_id={folder_id}, credentials={credentials_path}"
    )

    scopes = ["https://www.googleapis.com/auth/drive.file"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    file_metadata = {"name": os.path.basename(file_path), "parents": [folder_id]}
    media = MediaFileUpload(file_path, resumable=True)

    request = service.files().create(body=file_metadata, media_body=media, fields="id")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    file_id = response.get("id")
    print(f"Uploaded {file_path} to Google Drive with file id {file_id}")
    return file_id


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

ANNOTATION_URLS = {
    "train": (
        "http://data.csail.mit.edu/clevrer/annotations/train/annotation_train.zip",
        "annotation_train.zip",
    ),
    "valid": (
        "http://data.csail.mit.edu/clevrer/annotations/validation/annotation_validation.zip",
        "annotation_valid.zip",
    ),
    "test": (None, None),
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
        # Remove the zip to avoid storing both compressed and extracted copies
        if os.path.exists(local_path):
            os.remove(local_path)
    else:
        print(f"CLEVRER Video ({split}) already unzipped.")
        # If the archive still exists from a previous run, clean it up
        if local_path and os.path.exists(local_path):
            os.remove(local_path)

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


def download_annotations(annotation_dir: str, split: str) -> str:
    os.makedirs(annotation_dir, exist_ok=True)
    url_path, local_path = ANNOTATION_URLS[split]
    index_file = os.path.join(annotation_dir, f"{split}_index.json")
    output_dir = os.path.join(annotation_dir, split)

    # Fast-path: index already present → nothing to do
    if os.path.exists(index_file):
        print(f"CLEVRER Annotation ({split}) already prepared; skipping download.")
        return index_file

    if url_path and local_path:
        local_path = os.path.join(annotation_dir, local_path)
        if not os.path.exists(local_path):
            size_mb = get_file_size(url_path) / (1024 * 1024)
            print(
                f"Downloading CLEVRER Annotation ({split}) of size {size_mb:.2f} MB to {local_path}..."
            )
            urlretrieve(url_path, local_path)
        else:
            print(f"CLEVRER Annotation ({split}) already downloaded.")
    else:
        print(f"CLEVRER Annotation ({split}) not specified...")

    # unzip and create index file
    if local_path:
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(local_path, "r") as zip_ref:
            print(f"CLEVRER Unzip Annotation ({split}).")
            zip_ref.extractall(output_dir)
        build_annotation_index(output_dir, index_file)
        # Remove archive after successful extraction to avoid duplicates
        if os.path.exists(local_path):
            os.remove(local_path)

        return index_file

    return None


def build_annotation_index(annotation_root: str, output_file: str):
    annotation_data = {"root": ".", "num_scenes": 0, "samples": {}}
    samples = []

    # Create samples (scene_index, relative_path)
    for root, dirs, files in os.walk(annotation_root):
        for file in sorted(files):
            if file.startswith("annotation_") and file.endswith(".json"):
                # Extract number using simple string split
                parts = file.replace(".json", "").split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    scene_idx = int(parts[1])
                    relative_path = os.path.relpath(
                        os.path.join(root, file), os.path.dirname(output_file)
                    )
                    sample = {"scene_index": scene_idx, "relative_path": relative_path}
                    samples.append(sample)

    # Extract annotation informations
    print("CLEVRER Loading Annotations ...")
    num_samples = len(samples)
    for sample_idx in range(num_samples):
        sample = samples[sample_idx]
        sample_path = os.path.join(
            os.path.dirname(output_file), sample["relative_path"]
        )
        with open(sample_path, "r") as f:
            data = json.load(f)
            sample["object_counts"] = len(data["object_property"])
        if sample_idx % 500 == 0:
            print(
                f"...processing {sample_idx}/{num_samples} samples",
            )
        samples[sample_idx] = sample
    print("CLEVRER Loading Annotations Completed")

    annotation_data["samples"] = samples
    annotation_data["num_scenes"] = len(annotation_data["samples"])

    with open(output_file, "w") as f:
        json.dump(annotation_data, f, indent=2)

    print(f"Annotation index written to {output_file}")


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
            # OpenCV expects (width, height), but we follow PyTorch's (height, width) convention.
            # Flip the dimensions to match OpenCV's format.
            frame = cv2.resize(frame, (resize[1], resize[0]))

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


def _process_single_video(args: tuple[str, str, str, int, int, tuple[int, int] | None]):
    """
    Helper to process a single video in a multiprocessing pool.
    Returns the video id, relative chunk paths, and whether the work was skipped.
    """
    (
        video_path,
        processed_split_dir,
        processed_video_dir,
        chunk_size,
        stride,
        resize,
    ) = args

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(processed_split_dir, video_id)

    status = "skipped"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        extract_video_chunks_to_numpy(
            video_path=video_path,
            output_dir=output_dir,
            chunk_size=chunk_size,
            stride=stride,
            resize=resize,
        )
        status = "processed"

    chunk_files = sorted(glob(os.path.join(output_dir, "*.npy")))
    rel_files = [os.path.relpath(p, processed_video_dir) for p in chunk_files]
    return video_id, rel_files, status


def prepare_and_extract_clevrer_videos(
    raw_video_dir: str,
    processed_video_dir: str,
    split: str,
    chunk_size: int = 4,
    stride: int = 4,
    resize: tuple[int, int] | None = None,
    limit: int | None = None,
    num_workers: int | None = None,
) -> str:
    """
    Preprocess CLEVRER videos by extracting fixed-length chunks and saving them as numpy arrays.

    Args:
        raw_video_dir (str): Path to the directory with raw .mp4 video files.
        processed_video_dir (str): Where to save the extracted chunk numpy arrays.
        split (str): Dataset split name (e.g., "train", "valid", "test").
        chunk_size (int): Number of frames per chunk.
        stride (int): Step between chunks (stride = chunk_size → no overlap).
        resize (tuple or None): Resize (H, W) for frames, or None to keep original size.
        limit (int or None): Limit the number of videos to process.
        num_workers (int or None): Number of worker processes (defaults to CPU count).
    """
    index_filename = f"{split}_index.json"
    index_path = os.path.join(processed_video_dir, index_filename)
    if os.path.exists(index_path):
        return index_path

    processed_split_dir = os.path.join(processed_video_dir, split)
    os.makedirs(processed_split_dir, exist_ok=True)

    # Get and sort video paths lexicographically (consistent due to video naming)
    video_paths = sorted(get_all_video_paths(raw_video_dir))

    print(f"{len(video_paths)} videos found in {raw_video_dir}")
    if video_paths:
        print("Example:", video_paths[0])
    else:
        raise ValueError(f"No .mp4 videos found in {raw_video_dir}")

    if limit is not None:
        video_paths = video_paths[:limit]

    # Collect all chunk paths (relative to processed_video_dir) if we’re writing an index
    all_rel_paths = []
    task_args = [
        (
            video_path,
            processed_split_dir,
            processed_video_dir,
            chunk_size,
            stride,
            resize,
        )
        for video_path in video_paths
    ]

    resolved_workers = num_workers or (os.cpu_count() or 1)
    resolved_workers = max(1, min(resolved_workers, len(video_paths)))

    if resolved_workers > 1:
        with multiprocessing.Pool(processes=resolved_workers) as pool:
            results_iter = pool.imap(_process_single_video, task_args)
            for video_id, rel_files, status in tqdm(
                results_iter,
                total=len(video_paths),
                desc="Processing videos",
            ):
                all_rel_paths.extend(rel_files)
                if status == "skipped":
                    tqdm.write(f"{video_id} is already processed.")
                else:
                    tqdm.write(f"Finished processing {video_id}.")
    else:
        for args in tqdm(task_args, desc="Processing videos"):
            video_id, rel_files, status = _process_single_video(args)
            all_rel_paths.extend(rel_files)
            if status == "skipped":
                tqdm.write(f"{video_id} is already processed.")
            else:
                tqdm.write(f"Finished processing {video_id}.")

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
