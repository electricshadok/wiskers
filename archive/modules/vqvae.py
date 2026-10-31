from typing import Tuple, Union

import torch
import torchvision.utils as vutils
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from wiskers.common.metrics import codebook_usage_metrics
from wiskers.common.runtime.arg_utils import instantiate
from wiskers.modules.gen.base import BaseLightningModule


class VQVAE(BaseLightningModule):
    """
    A LightningModule that combines spatial and temporal modeling for video or physics prediction.
    Encodes input frames into a latent space (via VAE/VQ-VAE) and predicts their temporal evolution.

    Args:
        # Model configuration
        image_size (int or tuple): Input image size (H, W).
        model (dict or nn.Module): Hydra config or instance of the full autoencoder model (e.g. VQ_VAE2D).
        losses (dict): Loss configuration with optional keys:
            - reconstruction (str): Dotted path to reconstruction loss callable.
            - vq_weight (float): Scale for the vector-quantization loss.
            - reconstruction_weight (float): Scale for reconstruction loss.
            - ssim_weight (float): Weight for (1 - SSIM) loss component.
        # Optimizer configuration
        optimizer (dict, optional): Hydra config for an optimizer. Defaults to Adam if not provided.
        lr_scheduler (dict, optional): Hydra config for a torch LR scheduler.
    """

    def __init__(
        self,
        # Model Configuration
        image_size: Tuple[int, int],
        model: Union[dict, torch.nn.Module],
        losses: dict,
        # Optimizer Configuration
        optimizer: dict,
        lr_scheduler: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.image_size = image_size

        self.model = instantiate(model, _convert_="all")

        self.losses = instantiate(losses, _convert_="all")
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        # Set 'example_input_array' for ONNX export initialization
        in_channels = self.model._encoder.get_in_channels()
        self.example_input_array = torch.randn(
            1, in_channels, image_size[0], image_size[1]
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = instantiate(
            self.optimizer_cfg, params=self.model.parameters(), _convert_="all"
        )

        if self.lr_scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(
            self.lr_scheduler_cfg, optimizer=optimizer, _convert_="all"
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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

        # Extract image data
        images = self._unpack_images(batch)

        # Inference
        recon_x, vq_loss, indices = self.model(images)

        # Losses
        losses = self.losses(images, recon_x, vq_loss)
        loss = losses["loss"]

        self._log_tensor(losses, stage, prog_bar=True)

        # Metrics (codebook usage)
        with torch.no_grad():
            if hasattr(self.model._quantizer, "num_codes"):
                metrics = codebook_usage_metrics(
                    indices=indices,
                    num_codes=self.model._quantizer.num_codes,  # type: ignore[attr-defined]
                )
                self._log_tensor(metrics, stage, prog_bar=False)

        # Log statistics on tensors
        self._log_tensor_stats(stage, "image", images)
        self._log_tensor_stats(stage, "prediction", recon_x)

        # Collect images for visualization
        if stage == "train":
            self._collect_images(images, recon_x)

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

    def _collect_images(
        self, images: torch.Tensor, recons: torch.Tensor, max_buffer: int = 15
    ):
        if not hasattr(self, "image_buffer"):
            return

        for x, y in zip(images, recons):
            if len(self.image_buffer) >= max_buffer:
                break
            self.image_buffer.append(
                (
                    x.detach().cpu(),
                    y.detach().cpu(),
                )
            )

    def on_train_epoch_start(self):
        if self.global_rank == 0:
            self.image_buffer = []

    def on_train_epoch_end(self):
        if self.global_rank != 0:
            return

        # No images collected
        if not hasattr(self, "image_buffer") or len(self.image_buffer) == 0:
            return

        # Unpack triples
        inputs = [x for (x, _) in self.image_buffer]
        preds = [y for (_, y) in self.image_buffer]

        inputs_tensor = torch.stack(inputs)  # (N, C, H, W)
        preds_tensor = torch.stack(preds)  # (N, C, H, W)
        diffs_tensor = torch.abs(inputs_tensor - preds_tensor)

        # Rows
        n = inputs_tensor.size(0)

        row_inputs = vutils.make_grid(inputs_tensor, nrow=n, padding=2, normalize=True)
        row_preds = vutils.make_grid(preds_tensor, nrow=n, padding=2, normalize=True)
        row_diffs = vutils.make_grid(diffs_tensor, nrow=n, padding=2, normalize=True)

        # Vertical concatenation → one tall image
        full_vis = torch.cat([row_inputs, row_preds, row_diffs], dim=1)

        # Log
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                "train/input_pred_diff",
                full_vis,
                global_step=self.current_epoch,
            )

        # Clear buffer
        self.image_buffer = []
