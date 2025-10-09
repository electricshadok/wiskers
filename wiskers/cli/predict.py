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
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the PredictCLI.

        Args:
            config_path (str): Path to the configuration file.
        """
        # Load the configuration
        self.config = load_config(config_path)

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
        description="Run the training script with a given configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()
    cmd = PredictCLI(args.config)
    cmd.run()
