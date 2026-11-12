import argparse
import os

import lightning as L
import torchvision
from hydra.utils import instantiate

from wiskers.cli.utils import load_config


class PredictCLI:
    """
    Command-line interface for running image prediction using a trained PyTorch Lightning model checkpoint.

    Args:
        config_name (str): Configuration name (e.g. 'training/carracing_world_models') or path.
        config_path (str): Root configuration directory (default: 'configs').
    """

    def __init__(self, config_name: str, config_path: str = "./configs") -> None:
        self.config = load_config(config_name, config_path=config_path)

        # Initialize random number generators
        L.seed_everything(seed=self.config.seed, workers=True)

        # Setup inference engine
        self.backend_engine = instantiate(self.config.backend_engine)

    def run(self):
        """
        Runs the image generation process with the user settings.
        """
        datamodule = instantiate(self.config.data_module, _convert_="all")
        datamodule.prepare_data()
        # NOTE: In future, this section will be extended to perform video prediction
        # from image sequences in the test set.
        # test_loader = datamodule.test_dataloader()
        # batch = next(iter(test_loader))

        os.makedirs(self.config.output_dir, exist_ok=True)

        best_model_path = instantiate(self.config.best_model_path)

        samples = self.backend_engine(best_model_path, self.config.num_images)
        output_path = os.path.join(self.config.output_dir, "output.png")
        torchvision.utils.save_image(samples, output_path)
        print(f"Export output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run prediction with Hydra configuration."
    )
    parser.add_argument(
        "--config-path",
        "-cp",
        type=str,
        default="./configs",
        help="Root directory for configurations (default: ./configs)",
    )
    parser.add_argument(
        "--config-name",
        "--config",
        "-cn",
        type=str,
        required=True,
        help="Configuration name relative to config-path (e.g. training/carracing_world_models)",
    )
    args = parser.parse_args()
    cmd = PredictCLI(args.config_name, config_path=args.config_path)
    cmd.run()
