import lightning as L
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms


class CIFAR10(L.LightningDataModule):
    """
    LightningDataModule for CIFAR-10 dataset.

    Args:
        data_dir (str): Path to the directory containing the CIFAR-10 dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        image_size (int): Size of the images after resizing.
        random_horizontal_flip (bool): Whether to apply random horizontal flip during data augmentation.
    """

    CATEGORIES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        random_horizontal_flip: bool,
    ):
        super().__init__()
        self.data_dir = data_dir

        # Define the image transformations
        transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(image_size, antialias=True),
        ]
        if random_horizontal_flip:
            transformations.append(transforms.RandomHorizontalFlip(p=0.5))

        self.transform = transforms.Compose(transformations)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self):
        """
        Downloads the CIFAR-10 dataset if it does not already exist in the data directory.
        """
        dset.CIFAR10(self.data_dir, train=True, download=True)
        dset.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            generator = torch.Generator().manual_seed(42)
            train_full = dset.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.train_dataset, self.val_dataset = random_split(
                train_full, [0.8, 0.2], generator
            )

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.test_dataset = dset.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

        elif stage == "predict":
            self.predict_dataset = dset.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class CIFAR10Subset(CIFAR10):
    """
    LightningDataModule for creating a subset of the CIFAR-10 dataset with a specific category.

    Args:
        data_dir (str): Path to the directory containing the CIFAR-10 dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        image_size (int): Size of the images after resizing.
        random_horizontal_flip (bool): Whether to apply random horizontal flip during data augmentation.
        category_name (str): Name of the category (e.g., 'airplane', 'automobile') to create a subset for.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        image_size: int,
        random_horizontal_flip: bool,
        category_name: str,
    ):
        super().__init__(
            data_dir, batch_size, num_workers, image_size, random_horizontal_flip
        )
        if category_name not in self.CATEGORIES:
            raise ValueError(f"Expected {self.CATEGORIES} but got {category_name}")

        self.category_index = self.CATEGORIES.index(category_name)

    def _get_subset(self, dataset):
        """
        Get a subset of examples from the CIFAR-10 dataset that belong to the specified category.

        Args:
            dataset: The CIFAR-10 dataset.

        Returns:
            subset (Subset): Subset of examples from the specified category.
        """
        indices = [
            i
            for i, (_, label) in enumerate(self.train_dataset)
            if label == self.category_index
        ]
        return Subset(dataset, indices)

    def setup(self, stage: str):
        super().setup(stage)

        if stage == "fit":
            self.train_dataset = self._get_subset(self.train_dataset)
            self.val_dataset = self._get_subset(self.val_dataset)
        elif stage == "test":
            self.test_dataset = self._get_subset(self.test_dataset)
        elif stage == "predict":
            self.predict_dataset = self._get_subset(self.predict_dataset)
