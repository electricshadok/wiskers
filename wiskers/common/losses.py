import torch
import torch.nn as nn
import torch.nn.functional as F


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
