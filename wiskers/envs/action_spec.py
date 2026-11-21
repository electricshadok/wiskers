"""
PyTorch-native action space specifications independent of any environment backend.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ActionSpec(Protocol):
    """Protocol defining the PyTorch action contract for environments, policies, and planners."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def action_dim(self) -> int: ...

    def contains(self, action: torch.Tensor) -> bool:
        """Return True if ``action`` satisfies shape, dtype, and value constraints."""
        ...

    def sample(self) -> torch.Tensor:
        """Sample a valid random action tensor conforming to this specification."""
        ...


@dataclass(frozen=True)
class DiscreteActionSpec:
    """Single categorical discrete action in ``{0, ..., n - 1}`` with shape ``(1,)``."""

    n: int
    dtype: torch.dtype = torch.int64

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}.")
        if self.dtype.is_floating_point or self.dtype == torch.bool:
            raise ValueError(f"DiscreteActionSpec requires an integer dtype, got {self.dtype}.")

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def action_dim(self) -> int:
        return 1

    def contains(self, action: torch.Tensor) -> bool:
        if action.shape != self.shape or action.dtype != self.dtype:
            return False
        val = int(action.item())
        return 0 <= val < self.n

    def sample(self) -> torch.Tensor:
        return torch.randint(0, self.n, self.shape, dtype=self.dtype)


@dataclass(frozen=True, eq=False)
class BoundedActionSpec:
    """Finite bounded 1D floating-point action tensor specification."""

    low: torch.Tensor
    high: torch.Tensor

    def __post_init__(self) -> None:
        if self.low.ndim != 1 or self.low.shape[0] <= 0:
            raise ValueError(f"Bounds must be non-empty 1D tensors, got shape {tuple(self.low.shape)}.")
        if self.low.shape != self.high.shape:
            raise ValueError(f"Bounds shapes {tuple(self.low.shape)}, {tuple(self.high.shape)} must match.")
        if self.low.dtype != self.high.dtype:
            raise ValueError(f"Bounds dtypes {self.low.dtype}, {self.high.dtype} must match.")
        if self.low.device != self.high.device:
            raise ValueError(f"Bounds devices {self.low.device}, {self.high.device} must match.")
        if not self.low.dtype.is_floating_point:
            raise ValueError(f"BoundedActionSpec requires a floating-point dtype, got {self.low.dtype}.")
        if not (torch.isfinite(self.low).all() and torch.isfinite(self.high).all()):
            raise ValueError("Bounds must be finite (no NaN or inf).")
        if torch.any(self.low > self.high):
            raise ValueError("Each element of low must be <= high.")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.low.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.low.dtype

    @property
    def action_dim(self) -> int:
        return self.low.shape[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundedActionSpec):
            return NotImplemented
        if self.low.device != other.low.device:
            return False
        return torch.equal(self.low, other.low) and torch.equal(self.high, other.high)

    def contains(self, action: torch.Tensor) -> bool:
        if action.shape != self.shape or action.dtype != self.dtype:
            return False
        low = self.low.to(action.device)
        high = self.high.to(action.device)
        return bool(torch.all(action >= low) and torch.all(action <= high))

    def sample(self) -> torch.Tensor:
        return self.low + torch.rand(self.shape, dtype=self.dtype, device=self.low.device) * (self.high - self.low)
