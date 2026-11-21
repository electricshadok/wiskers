"""
WebUI adapter for AdroitHand dexterous manipulation environments.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class AdroitHandUIAdapter(EnvUIAdapter):
    """UI adapter for AdroitHand-v1 (continuous 24..30-DoF dexterous hand action space)."""

    env_name = "AdroitHandEnv"
    config_stem = "adroit_hand"
    display_name = "Adroit Hand"

    def action_label(self, index: int) -> str | None:
        if 0 <= index < 4:
            return f"Base/Wrist {index + 1}"
        if index >= 4:
            return f"Finger {index - 3}"
        return None

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        wrist_norm = float(torch.linalg.norm(a[:4]).item())
        finger_norm = float(torch.linalg.norm(a[4:]).item())
        grip_mean = float(a[4:].mean().item())

        return {
            "Wrist / Arm (L2)": StepInfo(
                value=f"{wrist_norm:.3f}",
                bar=min(1.0, wrist_norm / 2.0),
                note="Base & wrist joints norm",
            ),
            "Finger Joints (L2)": StepInfo(
                value=f"{finger_norm:.3f}",
                bar=min(1.0, finger_norm / 5.0),
                note="Dexterous finger actuation norm",
            ),
            "Mean Grip": StepInfo(
                value=f"{grip_mean:+.3f}",
                bar=(grip_mean + 1.0) / 2.0,
                note="Extend (-1) / Flex (+1)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                status="success" if reward > 0 else "info",
                note="✅ Positive task reward" if reward > 0 else "shaping / distance penalty",
            ),
            "Done": done_step_info(done),
        }
