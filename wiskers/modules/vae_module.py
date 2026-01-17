from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from wiskers.common.losses import kl_divergence_standard_normal
from wiskers.common.runtime.arg_utils import format_image_size, instantiate
from wiskers.models.autoencoder.vae_2d import VAE2D
from wiskers.modules.base_module import BaseLightningModule


class VAEModule(BaseLightningModule):
    """
    LightningModule for training and inference of a vae model.

    Args:
        # Model configuration
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        image_size (int or tuple): Input image size (H, W).
        activation (str): Activation function.
        # Optimizer configuration
        learning_rate (float): Learning rate for the optimizer.
    """

    def __init__(
        self,
        # Model Configuration
        in_channels: int = 3,
        stem_channels: Optional[int] = None,
        out_channels: int = 3,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128],
        attentions: List[bool] = [True, True, True],
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: str = "torch.nn.ReLU",
        # Optimizer Configuration
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.model = VAE2D(
            in_channels=in_channels,
            stem_channels=stem_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            image_size=image_size,
            activation=instantiate(activation),
        )
        self.learning_rate = learning_rate

        # Set 'example_input_array' for ONNX export initialization
        image_size = format_image_size(image_size)
        self.example_input_array = torch.randn(
            1, in_channels, image_size[0], image_size[1]
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

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

        images, labels = batch

        prediction, mu, logvar = self.model(images)

        kl_loss = kl_divergence_standard_normal(mu, logvar)

        # reconstruction loss
        reconstruction_loss = F.mse_loss(images, prediction)

        loss = kl_loss + reconstruction_loss

        # Log losses
        losses = {
            "loss": loss,
            "kl_loss": kl_loss,
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
        Generates samples.

        Args:
            num_images (int): Number of images to generate.

        Returns:
            torch.Tensor: Tensor of generated images with pixel values in [0, 1].
        """
        mid_c, mid_h, mid_w = self.model.get_latent_shape()
        z = torch.randn(num_samples, mid_c, mid_h, mid_w, device=self.device)
        samples = self.model.decoder(z)
        samples = samples.clip(0.0, 1.0)
        return samples

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
