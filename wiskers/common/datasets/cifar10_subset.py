from torch.utils.data import Subset

from wiskers.common.datasets.cifar10 import CIFAR10


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
        super().__init__(data_dir, batch_size, num_workers, image_size, random_horizontal_flip)
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
        indices = [i for i, (_, label) in enumerate(self.train_dataset) if label == self.category_index]
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
