"""
Base contract and fallback adapter for the WebUI Dataset Inspector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch


StatusLevel = Literal["success", "warning", "error", "info"]


@dataclass(frozen=True)
class StepInfo:
    """UI-ready descriptor for a single field in a transition step."""

    value: str
    bar: float | None = None
    status: StatusLevel | None = None
    note: str | None = None


def done_step_info(done: bool) -> StepInfo:
    """Format episode termination state consistently across environment adapters."""
    return StepInfo(
        value=str(done),
        status="error" if done else "success",
        note="🚨 Episode terminated" if done else "🟢 Episode active",
    )


class EnvUIAdapter(ABC):
    """
    Abstract adapter translating environment-specific actions and rewards into UI descriptors.

    Subclasses declare ``env_name``, ``config_stem``, and ``display_name`` so they can
    be registered centrally without touching UI or page-orchestration code.
    """

    env_name: str = ""
    config_stem: str = ""
    display_name: str = ""

    def action_label(self, index: int) -> str | None:
        """Optional human-readable label for a single action dimension or discrete action index."""
        return None

    @abstractmethod
    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        """Map a transition step into an ordered dict of UI-ready field descriptors."""
        ...


class DefaultUIAdapter(EnvUIAdapter):
    """Fallback adapter for environments without a registered adapter."""

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        return {
            "Action": StepInfo(value=str(action.tolist())),
            "Reward": StepInfo(value=f"{reward:.3f}"),
            "Done": done_step_info(done),
        }
