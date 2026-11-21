"""
WebUI adapter for InvertedPendulum-v5.
"""

import torch

from webui.dataset.adapters.base import EnvUIAdapter, StepInfo, done_step_info


class InvertedPendulumUIAdapter(EnvUIAdapter):
    """UI adapter for InvertedPendulum-v5 (continuous 1D cart force action space)."""

    env_name = "InvertedPendulumEnv"
    config_stem = "inverted_pendulum"
    display_name = "InvertedPendulum-v5"

    def action_label(self, index: int) -> str | None:
        return "Cart Force (N)" if index == 0 else None

    def describe_step(self, action: torch.Tensor, reward: float, done: bool) -> dict[str, StepInfo]:
        force = float(action.detach().cpu().flatten()[0].item())
        return {
            "Cart Force": StepInfo(
                value=f"{force:+.3f} N",
                bar=max(0.0, min(1.0, (force + 3.0) / 6.0)),
                note="Left (-3 N) / Right (+3 N)",
            ),
            "Reward": StepInfo(
                value=f"{reward:.3f}",
                status="success" if reward > 0 else "error",
                note="✅ Pole upright" if reward > 0 else "❌ Pole fell",
            ),
            "Done": done_step_info(done),
        }
