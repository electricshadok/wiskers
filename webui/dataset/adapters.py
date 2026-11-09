from abc import ABC, abstractmethod
from typing import Dict

import torch

from wiskers.datasets.trajectories.datamodule import TrajectoryDataModule


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
        a = action.detach().cpu().flatten()
        steering, gas, brake = float(a[0].item()), float(a[1].item()), float(a[2].item())

        if reward > 0:
            reward_status, reward_note = "success", "✅ new tile visited"
        elif reward == -0.1:
            reward_status, reward_note = "warning", "⏱️ time penalty (grass or revisited tile — indistinguishable)"
        else:
            reward_status, reward_note = "error", "⚠️ large penalty"

        return {
            "Steering": StepInfo(value=f"{steering:.3f}", bar=(steering + 1.0) / 2.0, status=None, note=None),
            "Gas":      StepInfo(value=f"{gas:.3f}",      bar=gas,                     status=None, note=None),
            "Brake":    StepInfo(value=f"{brake:.3f}",    bar=brake,                   status=None, note=None),
            "Reward":   StepInfo(value=f"{reward:.3f}",   bar=None, status=reward_status, note=reward_note),
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


class FrankaKitchenUIAdapter(DataModuleUIAdapter):
    """UI adapter for FrankaKitchen-v1 (continuous 9-DoF action space)."""

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> Dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        arm_norm = float(torch.linalg.norm(a[:7]).item())
        gripper_val = float(a[7:].mean().item())

        return {
            "Arm Velocity (L2)": StepInfo(
                value=f"{arm_norm:.3f}",
                bar=min(1.0, arm_norm / 2.65),
                status=None,
                note="Joints 1-7 velocity norm",
            ),
            "Gripper Action": StepInfo(
                value=f"{gripper_val:+.3f}",
                bar=(gripper_val + 1.0) / 2.0,
                status=None,
                note="Close (-1) / Open (+1)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                bar=None,
                status="success" if reward > 0 else "info",
                note="✅ Subtask completed!" if reward > 0 else "Task in progress",
            ),
            "Done": StepInfo(
                value=str(done),
                bar=None,
                status="error" if done else "success",
                note="🚨 Episode terminated" if done else "🟢 Episode active",
            ),
        }


class AdroitHandUIAdapter(DataModuleUIAdapter):
    """UI adapter for AdroitHand-v1 (continuous 24..30-DoF dexterous hand action space)."""

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> Dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        wrist_norm = float(torch.linalg.norm(a[:4]).item())
        finger_norm = float(torch.linalg.norm(a[4:]).item())
        grip_mean = float(a[4:].mean().item())

        return {
            "Wrist / Arm (L2)": StepInfo(
                value=f"{wrist_norm:.3f}",
                bar=min(1.0, wrist_norm / 2.0),
                status=None,
                note="Base & wrist joints norm",
            ),
            "Finger Joints (L2)": StepInfo(
                value=f"{finger_norm:.3f}",
                bar=min(1.0, finger_norm / 5.0),
                status=None,
                note="Dexterous finger actuation norm",
            ),
            "Mean Grip": StepInfo(
                value=f"{grip_mean:+.3f}",
                bar=(grip_mean + 1.0) / 2.0,
                status=None,
                note="Extend (-1) / Flex (+1)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                bar=None,
                status="success" if reward > 0 else "info",
                note="✅ Positive task reward" if reward > 0 else " shaping / distance penalty",
            ),
            "Done": StepInfo(
                value=str(done),
                bar=None,
                status="error" if done else "success",
                note="🚨 Episode terminated" if done else "🟢 Episode active",
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

def get_adapter(data_module: TrajectoryDataModule) -> DataModuleUIAdapter:
    """
    Return the UI adapter for the given TrajectoryDataModule instance.
    """
    env = data_module.collector.env
    spec = env.spec or env.unwrapped.spec
    env_str = spec.id.lower() if (spec is not None and spec.id) else type(env.unwrapped).__name__.lower()

    if "carracing" in env_str or "car_racing" in env_str:
        return CarRacingUIAdapter()
    if "breakout" in env_str:
        return BreakoutUIAdapter()
    if "franka" in env_str or "kitchen" in env_str:
        return FrankaKitchenUIAdapter()
    if "adroit" in env_str or "hand" in env_str:
        return AdroitHandUIAdapter()

    return DefaultUIAdapter()
