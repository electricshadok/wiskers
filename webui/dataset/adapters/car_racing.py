"""
WebUI adapter for CarRacing-v3.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class CarRacingUIAdapter(EnvUIAdapter):
    """UI adapter for CarRacing-v3 (continuous steering, gas, brake)."""

    env_name = "CarRacingEnv"
    config_stem = "car_racing"
    display_name = "CarRacing-v3"

    ACTION_LABELS = ("Steering", "Gas", "Brake")

    def action_label(self, index: int) -> str | None:
        return self.ACTION_LABELS[index] if 0 <= index < len(self.ACTION_LABELS) else None

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        a = action.detach().cpu().flatten()
        steering, gas, brake = float(a[0].item()), float(a[1].item()), float(a[2].item())

        if reward > 0:
            reward_status, reward_note = "success", "✅ new tile visited"
        elif reward == -0.1:
            reward_status, reward_note = "warning", "⏱️ time penalty (grass or revisited tile — indistinguishable)"
        else:
            reward_status, reward_note = "error", "⚠️ large penalty"

        return {
            "Steering": StepInfo(value=f"{steering:.3f}", bar=(steering + 1.0) / 2.0),
            "Gas": StepInfo(value=f"{gas:.3f}", bar=gas),
            "Brake": StepInfo(value=f"{brake:.3f}", bar=brake),
            "Reward": StepInfo(value=f"{reward:.3f}", status=reward_status, note=reward_note),
            "Done": done_step_info(done),
        }
