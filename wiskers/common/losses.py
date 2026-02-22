from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from wiskers.common.runtime.arg_utils import instantiate


class L1Loss(nn.Module):
    """L1 loss module."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)


class L2Loss(nn.Module):
    """L2 (MSE) loss module."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class MixedL1L2Loss(nn.Module):
    """Mix L1 and L2 losses for sharper yet stable reconstructions."""

    def __init__(self, alpha: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(input, target, reduction=self.reduction)
        l2 = F.mse_loss(input, target, reduction=self.reduction)
        return self.alpha * l1 + (1.0 - self.alpha) * l2


def kl_divergence_standard_normal(
    mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I).

    Args:
        mu (torch.Tensor): Mean tensor with shape (batch_size, latent_dim, ...).
        logvar (torch.Tensor): Log-variance tensor with the same shape as `mu`.

    Returns:
        torch.Tensor: Scalar tensor with the batch mean KL divergence.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()


def ssim_with_loss(
    input: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Structural Similarity Index (SSIM) and its complementary loss.

    Returns:
        ssim_value: SSIM in [0, 1], higher is better.
        ssim_loss: (1 - SSIM), suitable for minimization.
    """
    ssim_value = structural_similarity_index_measure(
        input, target, data_range=data_range
    )
    return ssim_value, 1 - ssim_value


class Losses:
    """Container for loss configuration so it can be Hydra-instantiated."""

    def __init__(
        self,
        reconstruction: Union[str, dict] = "wiskers.common.losses.MixedL1L2Loss",
        vq_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 0.0,
    ) -> None:
        self.reconstruction = instantiate(reconstruction)
        self.vq_weight = vq_weight
        self.reconstruction_weight = reconstruction_weight
        self.ssim_weight = ssim_weight

    def __call__(self, images: torch.Tensor, recon_x: torch.Tensor, vq_loss: torch.Tensor) -> dict:
        rec_loss = self.reconstruction(images, recon_x)
        if self.ssim_weight > 0.0:
            ssim_val, ssim_loss = ssim_with_loss(recon_x, images, data_range=1.0)
        else:
            ssim_val = torch.tensor(0.0, device=images.device)
            ssim_loss = torch.tensor(0.0, device=images.device)

        loss = (
            self.vq_weight * vq_loss
            + self.reconstruction_weight * rec_loss
            + self.ssim_weight * ssim_loss
        )
        losses = {
            "loss": loss,
            "vq_loss": vq_loss,
            "vq_loss_weighted": self.vq_weight * vq_loss,
            "reconstruction_loss": rec_loss,
            "ssim": ssim_val,
            "ssim_loss": ssim_loss,
        }
        return losses
