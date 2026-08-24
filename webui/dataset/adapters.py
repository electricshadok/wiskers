from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


# ── Contract ──────────────────────────────────────────────────────────────────

class StepInfo(dict):
    """
    UI-ready descriptor for a single field in a transition step.

    Keys:
        value  : str          — formatted value shown in st.metric
        bar    : float | None — progress bar value in [0, 1], or None
        status : str | None   — one of "success", "warning", "error", "info", or None
        note   : str | None   — text shown below the metric (colored by status if set)
    """


class DataModuleUIAdapter(ABC):
    """
    Abstract adapter between a LightningDataModule and the Streamlit dataset UI.

    Each environment subclasses this to provide environment-specific formatting
    without leaking UI logic into training code.
    """

    @abstractmethod
    def describe_step(
        self,
        action: torch.Tensor,
        reward: float,
        done: bool,
    ) -> Dict[str, StepInfo]:
        """
        Map a transition step into an ordered dict of UI-ready field descriptors.
        The UI renders them as columns in the order they appear.
        """
        ...


# ── Concrete Adapters ─────────────────────────────────────────────────────────

class CarRacingUIAdapter(DataModuleUIAdapter):
    """UI adapter for CarRacing-v3 (continuous action space)."""

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> Dict[str, StepInfo]:
        a = action.cpu().numpy()

        if reward > 0:
            reward_status, reward_note = "success", "✅ new tile visited"
        elif reward == -0.1:
            reward_status, reward_note = "warning", "⏱️ time penalty (grass or revisited tile — indistinguishable)"
        else:
            reward_status, reward_note = "error", "⚠️ large penalty"

        return {
            "Steering": StepInfo(value=f"{a[0]:.3f}", bar=float((a[0] + 1.0) / 2.0), status=None, note=None),
            "Gas":      StepInfo(value=f"{a[1]:.3f}", bar=float(a[1]),                status=None, note=None),
            "Brake":    StepInfo(value=f"{a[2]:.3f}", bar=float(a[2]),                status=None, note=None),
            "Reward":   StepInfo(value=f"{reward:.3f}", bar=None, status=reward_status, note=reward_note),
            "Done":     StepInfo(
                value=str(done), bar=None,
                status="error" if done else "success",
                note="🚨 Episode terminated" if done else "🟢 Episode active",
            ),
        }


class BreakoutUIAdapter(DataModuleUIAdapter):
    """UI adapter for ALE/Breakout-v5 (discrete action space)."""

    ACTION_LABELS = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> Dict[str, StepInfo]:
        idx = int(action.item())
        label = self.ACTION_LABELS.get(idx, str(idx))

        return {
            "Action": StepInfo(value=label, bar=None, status=None, note=f"index: {idx}"),
            "Reward": StepInfo(
                value=f"{reward:.3f}", bar=None,
                status="success" if reward > 0 else "info",
                note=f"✅ +{reward:.3f} brick hit!" if reward > 0 else "◼ no brick this step",
            ),
            "Done": StepInfo(
                value=str(done), bar=None,
                status="error" if done else "success",
                note="🚨 Episode ended" if done else "🟢 Episode active",
            ),
        }


class DefaultUIAdapter(DataModuleUIAdapter):
    """Fallback adapter for DataModules without a registered adapter."""

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> Dict[str, StepInfo]:
        return {
            "Action": StepInfo(value=str(action.tolist()), bar=None, status=None, note=None),
            "Reward": StepInfo(value=f"{reward:.3f}",      bar=None, status=None, note=None),
            "Done":   StepInfo(value=str(done),             bar=None, status=None, note=None),
        }


# ── Registry & Factory ────────────────────────────────────────────────────────

def get_adapter(data_module: Any) -> DataModuleUIAdapter:
    """
    Return the UI adapter for the given DataModule instance.

    Registration is done by class name to avoid circular imports between
    webui and wiskers training code.
    """
    _REGISTRY: Dict[str, type] = {
        "CarRacingDataModule": CarRacingUIAdapter,
        "BreakoutDataModule":  BreakoutUIAdapter,
    }
    cls_name = type(data_module).__name__
    adapter_cls = _REGISTRY.get(cls_name, DefaultUIAdapter)
    return adapter_cls()
