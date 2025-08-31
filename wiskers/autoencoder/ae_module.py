from typing import List, Tuple, Union

import lightning as L
import torch
import torch.nn.functional as F

from wiskers.autoencoder.models.ae_2d import Autoencoder2D
from wiskers.common.arg_utils import format_image_size, get_activation


class AEModule(L.LightningModule):
    """
    LightningModule for training and inference of an autocencoder model.

    Args:
        # Model configuration
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        z_dim (int): Bottleneck dimension for vae.
        image_size (int or tuple): Input image size (H, W).
        activation (str): Activation function.
        # Optimizer configuration
        learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(
        self,
        # Model Configuration
        in_channels: int = 3,
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        z_dim: int = 64,
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: str = "relu",
        # Optimizer Configuration
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.model = Autoencoder2D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            z_dim=z_dim,
            image_size=image_size,
            activation=get_activation(activation),
        )
        self.learning_rate = learning_rate
        self.z_dim = z_dim

        # Set 'example_input_array' for ONNX export initialization
        image_size = format_image_size(image_size)
        self.example_input_array = torch.randn(
            1, in_channels, image_size[0], image_size[1]
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

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

    def _unpack_batch(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ):
        if isinstance(batch, (tuple, list)):
            return batch[0]  # images, labels (labels unused)
        return batch  # some datasets return only images

    def _shared_step(self, batch, batch_idx: int, stage: str):
        """
        Processes a batch from a given stage (train, val, test).

        Args:
            batch: Tuple of images and labels.
            batch_idx (int): Index of the current batch.
            stage (str): Current stage ('train', 'val', or 'test').
        """
        valid_stages = ["train", "val", "test"]
        if stage not in valid_stages:
            raise ValueError(f"stage should {valid_stages}")

        images = self._unpack_batch(batch)

        prediction = self.model(images)

        reconstruction_loss = F.mse_loss(images, prediction)
        loss = reconstruction_loss

        # Log losses
        losses = {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
        }

        for name, value in losses.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # Log statistics on tensors
        self._log_tensor_stats(stage, "image", images)
        self._log_tensor_stats(stage, "prediction", prediction)

        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples: int) -> torch.Tensor:
        """
        Generates samples

        Args:
            num_images (int): Number of images to generate.

        Returns:
            torch.Tensor: Tensor of generated images with pixel values in [0, 1].
        """
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        samples = self.model.decoder(z)
        samples = samples.clip(0.0, 1.0)
        return samples

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
