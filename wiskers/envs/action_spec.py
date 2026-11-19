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
    """Bounded (or unbounded) numeric action tensor specification for any float or integer dtype."""

    low: torch.Tensor
    high: torch.Tensor

    def __post_init__(self) -> None:
        if self.low.ndim == 0 or any(d <= 0 for d in self.low.shape):
            raise ValueError(f"Bounds must have positive non-empty shape, got {tuple(self.low.shape)}.")
        if self.low.shape != self.high.shape:
            raise ValueError(f"Bounds shapes {tuple(self.low.shape)}, {tuple(self.high.shape)} must match.")
        if self.low.dtype != self.high.dtype:
            raise ValueError(f"Bounds dtypes {self.low.dtype}, {self.high.dtype} must match.")
        if self.low.dtype == torch.bool:
            raise ValueError("BoundedActionSpec requires a numeric dtype (float or integer), got torch.bool.")
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
        return torch.equal(self.low, other.low) and torch.equal(self.high, other.high)

    def contains(self, action: torch.Tensor) -> bool:
        if action.shape != self.shape or action.dtype != self.dtype:
            return False
        low = self.low.to(action.device)
        high = self.high.to(action.device)
        return bool(torch.all(action >= low) and torch.all(action <= high))

    def sample(self) -> torch.Tensor:
        if not self.dtype.is_floating_point:
            low_i = self.low.to(dtype=torch.float64)
            high_i = self.high.to(dtype=torch.float64)
            sampled = (low_i + torch.rand(self.shape, dtype=torch.float64) * (high_i - low_i + 1.0)).floor()
            return sampled.clamp(min=low_i, max=high_i).to(dtype=self.dtype)

        low = self.low.to(dtype=torch.float32)
        high = self.high.to(dtype=torch.float32)
        bounded = torch.isfinite(low) & torch.isfinite(high)
        if torch.all(bounded):
            return (low + torch.rand(self.shape, dtype=torch.float32) * (high - low)).to(dtype=self.dtype)
        out = torch.randn(self.shape, dtype=torch.float32)
        uniform = low + torch.rand(self.shape, dtype=torch.float32) * (high - low)
        return torch.where(bounded, uniform, out).to(dtype=self.dtype)
