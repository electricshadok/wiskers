import os
from dataclasses import dataclass, field
from typing import List, Optional

import lightning as L
from torch.utils.data import DataLoader

import wiskers.datasets.clevrer.datasets as datasets
from wiskers.datasets.clevrer.prepare import (
    bundle_clevrer_for_upload,
    download_annotations,
    download_qa,
    download_videos,
    prepare_and_extract_clevrer_videos,
    upload_file_to_gdrive,
)


@dataclass
class GDriveUploadConfig:
    enabled: bool = False
    folder_id: Optional[str] = None
    credentials_path: Optional[str] = None
    archive_name: str = "clevrer_processed.zip"

    def resolve_credentials_path(self) -> Optional[str]:
        return self.credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")


@dataclass
class PreprocessingConfig:
    chunk_size: int = 16
    stride: int = 16
    resize: List[int] = field(default_factory=lambda: [160, 240])
    limit: Optional[int] = None  # Optional limit on number of videos processed
    gdrive_upload: Optional[GDriveUploadConfig] = None

    def __post_init__(self):
        if isinstance(self.gdrive_upload, dict):
            self.gdrive_upload = GDriveUploadConfig(**self.gdrive_upload)


@dataclass
class TransformConfig:
    image_size: List[int] = field(default_factory=lambda: [80, 120])


class ClevrerMedia(L.LightningDataModule):
    """
    Base LightningDataModule for CLEVRER dataset variants (video/image).

    Handles shared logic like downloading, extraction, and QA/video index resolution.

    Args:
        data_dir (str): Path to the root data directory.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of subprocesses for data loading.
        chunk_size (int): Number of frames per video chunk.
        preprocessing (PreprocessingConfig): Pre-processing raw video parameters
        transform (TransformConfig): Post-processing transform parameters
        splits (List[str], optional): List of dataset splits to prepare and load.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        preprocessing: PreprocessingConfig,
        transform: TransformConfig,
        splits: Optional[List[str]] = None,
    ):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "clevrer")
        self.preprocessed_root = os.path.join(self.data_dir, "preprocessed")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing = preprocessing
        self.transform = transform
        self.splits = splits or ["train", "valid", "test"]
        self.qa_paths = {}
        self.annotation_index_paths = {}
        self.video_index_paths = {}

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.preprocessed_root, exist_ok=True)
        for split in self.splits:
            qa_root = os.path.join(self.preprocessed_root, "question_answer")
            annotation_root = os.path.join(self.preprocessed_root, "annotations")
            video_raw_root = os.path.join(self.data_dir, "video_raw")
            processed_video_dir = os.path.join(self.preprocessed_root, "video")

            # Download QA JSON
            qa_path = download_qa(qa_root, split)
            self.qa_paths[split] = qa_path
            # qa_helper = datasets.ClevrerQAHelper(qa_path)
            print(f"CLEVRER QA ({split}) {qa_path}")

            # Download & extract annotations
            # Note: CLEVRER only have annotations for valid and train sets
            annotation_index_path = download_annotations(annotation_root, split)
            self.annotation_index_paths[split] = annotation_index_path
            print(f"CLEVRER Annotation Index ({split}) {annotation_index_path}")

            # Download & extract videos
            if os.path.isdir(processed_video_dir) and os.listdir(processed_video_dir):
                print(
                    f"CLEVRER Videos already processed in {processed_video_dir}; skipping download."
                )
                raw_video_dir = ""  # Not needed since videos are already processed"
            else:
                raw_video_dir = download_videos(video_raw_root, split)

            video_index_path = prepare_and_extract_clevrer_videos(
                raw_video_dir=raw_video_dir,
                processed_video_dir=processed_video_dir,
                split=split,
                chunk_size=self.preprocessing.chunk_size,
                stride=self.preprocessing.stride,
                resize=self.preprocessing.resize,
                limit=self.preprocessing.limit,
            )
            self.video_index_paths[split] = video_index_path
            print(f"CLEVRER Video Index ({split}) {video_index_path}")

        self._maybe_upload_to_gdrive()

    def _make_dataloader(self, split: str, dataset_cls):
        dataset = dataset_cls(
            video_index_path=self.video_index_paths[split],
            annotation_index_path=self.annotation_index_paths[split],
            qa_path=self.qa_paths[split],
            resize=self.transform.image_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def _maybe_upload_to_gdrive(self):
        upload_cfg = getattr(self.preprocessing, "gdrive_upload", None)
        if not upload_cfg or not upload_cfg.enabled:
            return

        if not upload_cfg.folder_id:
            raise ValueError(
                "upload_to_gdrive is enabled but no destination folder_id was provided."
            )

        credentials_path = upload_cfg.resolve_credentials_path()
        if not credentials_path:
            raise ValueError(
                "upload_to_gdrive is enabled but no credentials_path was provided and "
                "GOOGLE_APPLICATION_CREDENTIALS is not set."
            )

        archive_path = bundle_clevrer_for_upload(
            processed_root=self.preprocessed_root,
            archive_name=upload_cfg.archive_name,
        )
        upload_file_to_gdrive(
            file_path=archive_path,
            folder_id=upload_cfg.folder_id,
            credentials_path=credentials_path,
        )


class ClevrerVideo(ClevrerMedia):
    """
    LightningDataModule for loading CLEVRER video chunks as 4D tensors (T, C, H, W).
    """

    def train_dataloader(self):
        return self._make_dataloader("train", datasets.ClevrerVideo)

    def val_dataloader(self):
        return self._make_dataloader("valid", datasets.ClevrerVideo)

    def test_dataloader(self):
        return self._make_dataloader("test", datasets.ClevrerVideo)


class ClevrerImage(ClevrerMedia):
    """
    LightningDataModule for loading individual frames from CLEVRER video chunks.
    Each frame is returned as a 3D tensor (C, H, W).
    """

    def train_dataloader(self):
        return self._make_dataloader("train", datasets.ClevrerImage)

    def val_dataloader(self):
        return self._make_dataloader("valid", datasets.ClevrerImage)

    def test_dataloader(self):
        return self._make_dataloader("test", datasets.ClevrerImage)
