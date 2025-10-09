from abc import ABC, abstractmethod

import lightning as L
import torch


class BaseLightningModule(L.LightningModule, ABC):
    """
    Base Lightning module that defines shared logging and sample-generation interface.
    Other model modules (e.g., VAE, WorldModel) should inherit from this class.
    """

    def _log_tensor_stats(self, stage: str, tensor_name: str, data: torch.tensor):
        """
        Logs min, max, and mean of a tensor at a given stage.

        Args:
            stage (str): Stage of step ('train', 'val', 'test').
            tensor_name (str): Name of the tensor.
            data (torch.tensor): Data to analyze.

        The method logs these statistics per epoch using `self.log`. It is intended as a private utility
        within its class.
        """
        stats = {
            "min": data.min(),
            "max": data.max(),
            "mean": data.mean(),
        }

        for stat_name, state_value in stats.items():
            self.log(
                f"{stage}_stats/{tensor_name}_{stat_name}",
                state_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                reduce_fx=stat_name,
            )

    @abstractmethod
    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generates samples.

        Args:
            num_images (int): Number of images to generate.

        Returns:
            torch.Tensor: Tensor of generated images with pixel values in [0, 1].
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement `generate_samples()`"
        )
