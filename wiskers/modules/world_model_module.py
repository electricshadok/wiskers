from typing import Optional, Tuple, Union

import torch
import torchvision.utils as vutils
from lightning.pytorch.loggers import TensorBoardLogger

from wiskers.common.blocks.quantizer import VectorQuantizer
from wiskers.common.losses import ssim_with_loss
from wiskers.common.metrics import codebook_usage_metrics
from wiskers.common.runtime.arg_utils import format_image_size, instantiate
from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder
from wiskers.models.autoencoder.vqvae_2d import VQ_VAE2D
from wiskers.modules.base_module import BaseLightningModule


class LossesConfig:
    """Container for loss configuration so it can be Hydra-instantiated."""

    def __init__(
        self,
        reconstruction: Union[str, dict] = "wiskers.common.losses.MixedL1L2Loss",
        vq_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 0.0,
    ) -> None:
        self.reconstruction = instantiate(reconstruction)
        self.vq_weight = float(vq_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.ssim_weight = float(ssim_weight)


class WorldModelModule(BaseLightningModule):
    """
    A LightningModule that combines spatial and temporal modeling for video or physics prediction.
    Encodes input frames into a latent space (via VAE/VQ-VAE) and predicts their temporal evolution.

    Args:
        # Model configuration
        image_size (int or tuple): Input image size (H, W).
        encoder (dict or nn.Module): Hydra config or instance of CNNEncoder (required).
        decoder (dict or nn.Module): Hydra config or instance of CNNDecoder (required).
        quantizer (dict or VectorQuantizer): Hydra config or instance of a VectorQuantizer.
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
        encoder: Union[dict, CNNEncoder],
        decoder: Union[dict, CNNDecoder],
        image_size: Union[int, Tuple[int, int]] = 32,
        quantizer: Union[dict, VectorQuantizer] = None,
        losses: Union[dict, "LossesConfig"] = None,
        # Optimizer Configuration
        optimizer: Optional[dict] = None,
        lr_scheduler: Optional[dict] = None,
    ) -> None:
        super().__init__()
        # Avoid storing the quantizer module in hparams; keep config via checkpoint if needed.
        self.save_hyperparameters(ignore=["quantizer"])
        self.image_size = image_size
        if isinstance(encoder, dict):
            encoder = instantiate(encoder, _convert_="all")

        if isinstance(decoder, dict):
            decoder = instantiate(decoder, _convert_="all")

        if encoder is None or decoder is None:
            raise ValueError("Encoder and decoder must be provided or constructible.")

        latent_shape = encoder.get_latent_shape(image_size)

        # Build quantizer from config or validate provided instance
        if isinstance(quantizer, dict):
            quantizer_cfg = dict(quantizer)
            quantizer_cfg.setdefault("code_dim", latent_shape[0])
            quantizer = instantiate(quantizer_cfg, _convert_="all")
        if not isinstance(quantizer, VectorQuantizer):
            raise TypeError("quantizer must be a VectorQuantizer instance.")
        if quantizer.code_dim != latent_shape[0]:
            raise ValueError(
                f"quantizer.code_dim must match encoder latent channels: {latent_shape[0]}"
            )

        self.model = VQ_VAE2D(
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            latent_shape=latent_shape,
        )
        if isinstance(losses, dict):
            losses_cfg = instantiate(losses, _convert_="all")
        elif isinstance(losses, LossesConfig):
            losses_cfg = losses
        else:
            raise TypeError("losses must be a dict or LossesConfig.")

        self.losses = losses_cfg
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler

        # Set 'example_input_array' for ONNX export initialization
        in_channels = encoder.get_in_channels()
        image_size = format_image_size(image_size)
        self.example_input_array = torch.randn(
            1, in_channels, image_size[0], image_size[1]
        )

    def configure_optimizers(self):
        if self.optimizer_cfg is None:
            optimizer = torch.optim.Adam(self.model.parameters())
        else:
            optimizer = instantiate(
                self.optimizer_cfg,
                params=self.model.parameters(),
                _convert_="all",
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
        rec_loss = self.losses.reconstruction(images, recon_x)
        if self.losses.ssim_weight > 0.0:
            ssim_val, ssim_loss = ssim_with_loss(recon_x, images, data_range=1.0)
        else:
            ssim_val = torch.tensor(0.0, device=images.device)
            ssim_loss = torch.tensor(0.0, device=images.device)

        loss = (
            self.losses.vq_weight * vq_loss
            + self.losses.reconstruction_weight * rec_loss
            + self.losses.ssim_weight * ssim_loss
        )
        losses = {
            "loss": loss,
            "vq_loss": vq_loss,
            "vq_loss_weighted": self.losses.vq_weight * vq_loss,
            "reconstruction_loss": rec_loss,
            "ssim": ssim_val,
            "ssim_loss": ssim_loss,
        }

        self._log_tensor(losses, stage, prog_bar=True)

        # Metrics (codebook usage)
        with torch.no_grad():
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
