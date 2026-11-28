import argparse
from typing import Any

import lightning as L
from hydra.utils import instantiate

from wiskers.controller import Controller
from wiskers.data.buffer import RolloutBuffer
from wiskers.data.collector import Collector
from wiskers.data.transforms import EpisodeTransform
from wiskers.envs.gym_env import GymEnv
from wiskers.pipeline.backends import InferenceBackend
from wiskers.pipeline.config import load_config


class PlayCLI:
    """
    Command-line interface for playing a trained policy or model checkpoint in an environment.

    Args:
        config_name: Configuration name (e.g. ``'training/car_racing_world_models'``) or YAML path.
        config_path: Root configuration directory (default: ``'./configs'``).
        overrides: Optional CLI configuration overrides.
    """

    def __init__(
        self,
        config_name: str,
        config_path: str = "./configs",
        overrides: list[str] | None = None,
    ) -> None:
        self.config = load_config(config_name, config_path=config_path, overrides=overrides)
        L.seed_everything(seed=self.config.seed, workers=True)
        self.backend: InferenceBackend = instantiate(self.config.backend, _convert_="all")

    def load_model(self, checkpoint_path: str | None = None) -> Any:
        """Load the configured model artifact via ``self.backend``."""
        model_path = checkpoint_path if checkpoint_path is not None else self.config.checkpoint_path
        return self.backend.load(str(model_path))

    @staticmethod
    def _resolve_controller(model: Any) -> Controller:
        """Return ``model`` if it implements ``Controller``, or its ``.controller`` attribute."""
        if isinstance(model, Controller):
            return model
        controller = getattr(model, "controller", None)
        if isinstance(controller, Controller):
            return controller
        raise TypeError("Loaded model must implement Controller or expose a .controller attribute.")

    def run(
        self,
        checkpoint_path: str | None = None,
        num_episodes: int | None = None,
        max_steps: int | None = None,
    ) -> RolloutBuffer:
        """Roll out the loaded controller in the configured environment and return collected episodes."""
        model = self.load_model(checkpoint_path)
        controller = self._resolve_controller(model)

        env: GymEnv = instantiate(self.config.data_module.collector.env, _convert_="all")
        transform: EpisodeTransform = instantiate(self.config.transform, _convert_="all")
        episodes = num_episodes if num_episodes is not None else int(self.config.get("eval_episodes", 5))
        steps = max_steps if max_steps is not None else int(self.config.data_module.collector.max_steps)

        collector = Collector(
            env=env,
            controller=controller,
            transform=transform,
            num_episodes=episodes,
            min_steps=0,
            max_steps=steps,
            seed=self.config.seed,
            verbose=True,
        )
        buffer = RolloutBuffer()
        try:
            collector.collect(buffer)
        finally:
            collector.close()

        avg_len = buffer.num_steps / buffer.num_episodes if buffer.num_episodes else 0.0
        avg_return = sum(ep.reward.sum().item() for ep in buffer) / buffer.num_episodes if buffer.num_episodes else 0.0
        print(
            f"[Play] episodes={buffer.num_episodes} | total_steps={buffer.num_steps} | "
            f"avg_len={avg_len:.1f} | avg_return={avg_return:.2f}"
        )
        return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained model checkpoint in the environment.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="./configs",
        help="Root directory for configurations (default: ./configs)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name relative to config-path (e.g. training/car_racing_world_models)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model artifact path overriding config.checkpoint_path",
    )
    args, extra_args = parser.parse_known_args()
    cmd = PlayCLI(args.config, config_path=args.config_path, overrides=extra_args)
    cmd.run(checkpoint_path=args.checkpoint)
