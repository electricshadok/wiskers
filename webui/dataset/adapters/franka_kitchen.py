"""
WebUI adapter for FrankaKitchen-v1.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class FrankaKitchenUIAdapter(EnvUIAdapter):
    """UI adapter for FrankaKitchen-v1 (continuous 9-DoF action space)."""

    env_name = "FrankaKitchenEnv"
    config_stem = "franka_kitchen"
    display_name = "Franka Kitchen"

    def action_label(self, index: int) -> str | None:
        if 0 <= index < 7:
            return f"Arm Joint {index + 1}"
        if 7 <= index < 9:
            return f"Gripper {index - 6}"
        return None

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        arm_norm = float(torch.linalg.norm(a[:7]).item())
        gripper_val = float(a[7:].mean().item())

        return {
            "Arm Velocity (L2)": StepInfo(
                value=f"{arm_norm:.3f}",
                bar=min(1.0, arm_norm / 2.65),
                note="Joints 1-7 velocity norm",
            ),
            "Gripper Action": StepInfo(
                value=f"{gripper_val:+.3f}",
                bar=(gripper_val + 1.0) / 2.0,
                note="Close (-1) / Open (+1)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                status="success" if reward > 0 else "info",
                note="✅ Subtask completed!" if reward > 0 else "Task in progress",
            ),
            "Done": done_step_info(done),
        }
