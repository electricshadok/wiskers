import json
import os
import zipfile
from urllib.request import urlretrieve

from torch.utils.data import Dataset
from tqdm import tqdm

from wiskers.datasets.clevrer_utils import (
    frame_count_from_video,
    get_all_videos,
    get_file_size,
)


class Clevrer(Dataset):
    """
    CLEVRER: CoLlision Events for Video REpresentation and Reasoning
    http://clevrer.csail.mit.edu/
    """

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

    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir

    def download_all(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # Download JSON Question_Answer
        qa_dir = os.path.join(self.data_dir, "question_answer")
        os.makedirs(qa_dir, exist_ok=True)
        for setname, url_n_local in Clevrer.QA_URLS.items():
            url_path, local_path = url_n_local
            if url_path and local_path:
                local_path = os.path.join(qa_dir, local_path)
                if not os.path.exists(local_path):
                    size_mb = get_file_size(url_path) / (1024 * 1024)
                    print(
                        f"Downloading CLEVRER Question-Answer ({setname}) of size {size_mb:.2f} MB to {local_path}..."
                    )
                    urlretrieve(url_path, local_path)
                else:
                    print(f"CLEVRER Question-Answer ({setname}) already downloaded.")
            else:
                print(f"CLEVRER Question-Answer ({setname}) not specified...")

        # Downlod Zip video and Unzip them
        video_dir = os.path.join(self.data_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        for setname, url_n_local in Clevrer.VIDEO_URLS.items():
            url_path, local_path = url_n_local
            if url_path and local_path:
                local_path = os.path.join(video_dir, local_path)
                if not os.path.exists(local_path):
                    size_mb = get_file_size(url_path) / (1024 * 1024)
                    print(
                        f"Downloading CLEVRER Video ({setname}) of size {size_mb:.2f} MB to {local_path}..."
                    )
                    urlretrieve(url_path, local_path)
                else:
                    print(f"CLEVRER Video ({setname}) already downloaded.")
            else:
                print(f"CLEVRER Video ({setname}) not specified...")

            raw_video_dir = os.path.join(video_dir, setname)
            if not os.path.exists(raw_video_dir):
                os.makedirs(raw_video_dir, exist_ok=True)
                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    print(f"CLEVRER Unzip Video ({setname}).")
                    zip_ref.extractall(raw_video_dir)
            else:
                print(f"CLEVRER Video ({setname}) already unzipped.")

        splits = ["train", "valid", "test"]
        video_root = os.path.join(self.data_dir, "videos")

        for split in splits:
            video_dir = os.path.join(video_root, split)
            json_path = os.path.join(video_root, f"{split}.json")

            if os.path.exists(json_path):
                print(f"Skipping {split}, already processed: {json_path}")
                continue

            video_paths = get_all_videos(video_dir)
            print(f"Num videos: {len(video_paths)}")
            print(f"Example video: {video_paths[0]}")

            # Track progress of frame counting
            frame_map = {}
            for path in tqdm(video_paths, desc="Counting frames"):
                try:
                    count = frame_count_from_video(path)
                    frame_map[path] = count
                except Exception as e:
                    print(f"Error reading {path}: {e}")

            with open(json_path, "w") as f:
                json.dump(frame_map, f, indent=2)

            total_frames = sum(frame_map.values())
            print(
                f"Saved frame count mapping to: {json_path} ({len(frame_map)} videos, {total_frames} frames)"
            )
