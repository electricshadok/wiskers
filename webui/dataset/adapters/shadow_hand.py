"""
WebUI adapter for Shadow Dexterous Hand environments.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class ShadowHandUIAdapter(EnvUIAdapter):
    """UI adapter for Shadow Dexterous Hand (continuous 20-DoF action space)."""

    env_name = "ShadowHandEnv"
    config_stem = "shadow_hand"
    display_name = "Shadow Hand"

    def action_label(self, index: int) -> str | None:
        if 0 <= index < 2:
            return f"Wrist {index + 1}"
        if 2 <= index < 20:
            return f"Finger {index - 1}"
        return None

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        wrist_norm = float(torch.linalg.norm(a[:2]).item())
        finger_norm = float(torch.linalg.norm(a[2:]).item())
        grip_mean = float(a[2:].mean().item())

        return {
            "Wrist (L2)": StepInfo(
                value=f"{wrist_norm:.3f}",
                bar=min(1.0, wrist_norm / 1.42),
                note="Wrist joints (2 DoF) norm",
            ),
            "Finger Joints (L2)": StepInfo(
                value=f"{finger_norm:.3f}",
                bar=min(1.0, finger_norm / 4.25),
                note="Dexterous finger actuation norm",
            ),
            "Mean Grip": StepInfo(
                value=f"{grip_mean:+.3f}",
                bar=(grip_mean + 1.0) / 2.0,
                note="Extend (-1) / Flex (+1)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                status="success" if reward == 0 else "info",
                note="✅ Goal reached" if reward == 0 else "In progress",
            ),
            "Done": done_step_info(done),
        }
