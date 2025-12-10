from typing import List

import torch
import torch.nn.functional as F

from wiskers.common.runtime.arg_utils import instantiate
from wiskers.models.diffusion.schedulers.registry import Schedulers
from wiskers.models.diffusion.unet_2d import UNet2D
from wiskers.modules.base_module import BaseLightningModule


class DiffuserModule(BaseLightningModule):
    """
    LightningModule for training and inference of a diffusion model.

    Args:
        # Model Configuration
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_dim (int): Time dimension.
        num_heads (int): Number of attention heads.
        widths (List[int]): Filter width per level.
        attentions (List[bool]) : Enable attention per level.
        image_size (int): Size of the image.
        activation (str): Activation function.
        # Scheduler Configuration
        scheduler_type (str): Type of scheduler to use for beta schedule.
            scheduler_type can be ("linear", "quadratic", "cosine", or "sigmoid").
        num_steps (float): Number of diffusion steps.
        beta_start (float): Start value of beta for scheduling.
        beta_end (float): End value of beta for scheduling.
        # Optimizer Configuration
        learning_rate (float): Learning rate for the optimizer.
        prediction_type (str): Prediction type ("noise", "sample", "v-prediction")
    """

    def __init__(
        self,
        # Model Configuration
        in_channels: int = 3,
        out_channels: int = 3,
        time_dim: int = 256,
        num_heads: int = 8,
        widths: List[int] = [32, 64, 128, 256],
        attentions: List[bool] = [True, True, True],
        image_size: int = 32,
        activation: str = "torch.nn.ReLU",
        # Scheduler Configuration
        scheduler_type: str = "ddpm",
        num_steps: float = 1000,
        beta_start: float = 1e-5,
        beta_end: float = 1e-2,
        beta_schedule: str = "linear",
        # Optimizer Configuration
        learning_rate: float = 1e-4,
        prediction_type: str = "noise",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.time_dim = time_dim
        self.image_size = image_size
        self.model = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=time_dim,
            num_heads=num_heads,
            widths=widths,
            attentions=attentions,
            activation=instantiate(activation),
        )
        self.learning_rate = learning_rate
        if prediction_type not in ["noise", "sample", "v-prediction"]:
            raise ValueError(f"{prediction_type} is not a valid prediction type")
        self.prediction_type = prediction_type

        # Register scheduler and associated buffer
        self.scheduler = Schedulers.get(scheduler_type)(
            num_steps, beta_start, beta_end, beta_schedule
        )

        # Set 'example_input_array' for ONNX export initialization
        self.example_input_array = (
            torch.randn(1, in_channels, image_size, image_size),
            torch.zeros((1)).long(),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def forward(self, x, t):
        return self.model(x, t)

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
        batch_size = images.shape[0]

        # Uniformly sample timesteps
        num_steps = self.scheduler.num_steps()
        t = torch.randint(
            0, num_steps, (batch_size,), dtype=torch.long, device=images.device
        )

        # Diffuse the batched images with noise
        noise = torch.randn_like(images)
        noisy_images = self.scheduler.q_sample(images, t, noise)

        # Predict the noise from the batch images
        prediction = self.model(noisy_images, t)

        if self.prediction_type == "noise":
            loss = F.mse_loss(noise, prediction)
        elif self.prediction_type == "sample":
            loss = F.mse_loss(images, prediction)
        else:
            raise NotImplementedError("prediction not implemented")
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Log statistics on tensors
        self._log_tensor_stats(stage, "noisy_image", noisy_images)
        self._log_tensor_stats(stage, "image", images)
        self._log_tensor_stats(stage, "noise", noise)
        self._log_tensor_stats(stage, "prediction", prediction)

        return loss

    @torch.no_grad()
    def generate_samples(
        self, num_samples: int, num_inference_steps: int
    ) -> torch.Tensor:
        """
        Generates samples using diffusion model.

        Args:
            num_images (int): Number of images to generate.
            num_inference_steps (int): Number of steps for the diffusion process.

        Returns:
            torch.Tensor: Tensor of generated images with pixel values in [0, 1].
        """
        samples = torch.randn(
            num_samples,
            self.model.in_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )

        for step_id in reversed(range(0, num_inference_steps)):
            t = torch.full(
                (num_samples,), step_id, device=self.device, dtype=torch.long
            )
            samples = self.scheduler.p_sample(self.model, samples, t, step_id)

        samples = samples.clip(0.0, 1.0)
        return samples

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
