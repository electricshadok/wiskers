"""
WebUI adapter for ALE/Breakout.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class BreakoutUIAdapter(EnvUIAdapter):
    """UI adapter for ALE/Breakout-v5 (discrete action space)."""

    env_name = "BreakoutEnv"
    config_stem = "breakout"
    display_name = "ALE/Breakout-v5"

    ACTION_LABELS = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}

    def action_label(self, index: int) -> str | None:
        return self.ACTION_LABELS.get(index)

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        idx = int(action.item())
        label = self.ACTION_LABELS.get(idx, str(idx))

        return {
            "Action": StepInfo(value=label, note=f"index: {idx}"),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                status="success" if reward > 0 else "info",
                note=f"✅ +{reward:.3f} brick hit!" if reward > 0 else "◼ no brick this step",
            ),
            "Done": done_step_info(done),
        }
