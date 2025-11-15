from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from wiskers.autoencoder.models.vqvae_2d import VQ_VAE2D
from wiskers.common.arg_utils import format_image_size, torch_instantiate
from wiskers.common.base_module import BaseLightningModule


class WorldModelModule(BaseLightningModule):
    """
    A LightningModule that combines spatial and temporal modeling for video or physics prediction.
    Encodes input frames into a latent space (via VAE/VQ-VAE) and predicts their temporal evolution.

    Args:
        # Model configuration
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heads (int): Number of self-attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        image_size (int or tuple): Input image size (H, W).
        activation (str): Activation function.
        # Codebook configuration
        num_codes (int): Number of discrete embeddings in the codebook (K).
        beta (float): Weight for the commitment loss term, typically between 0.1 and 0.5.
        use_ema (bool): Whether to use EMA updates for the codebook.
        decay (float): EMA decay factor (only used if use_ema=True).
        eps (float): Small constant for numerical stability.
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
        image_size: Union[int, Tuple[int, int]] = 32,
        activation: str = "nn.ReLU",
        # Codebook Configuration
        num_codes: int = 512,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        # Optimizer Configuration
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.model = VQ_VAE2D(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            image_size=image_size,
            activation=torch_instantiate(activation),
            num_codes=num_codes,
            beta=beta,
            use_ema=use_ema,
            decay=decay,
            eps=eps,
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
        recon_x, vq_loss, indices = self.model(x)
        return recon_x

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
            raise ValueError(f"stage should be one of {valid_stages}")

        images = self._unpack_images(batch)

        # Compute losses
        recon_x, vq_loss, indices = self.model(images)
        reconstruction_loss = F.mse_loss(images, recon_x)
        loss = vq_loss + reconstruction_loss

        # Log losses
        losses = {
            "loss": loss,
            "vq_loss": vq_loss,
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
        self._log_tensor_stats(stage, "prediction", recon_x)

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
