import argparse
import glob
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

    def get_output_dir(self) -> str:
        run_dir = os.path.join(self.config.experiment_dir, self.config.run_name)
        output_dir = os.path.join(run_dir, "outputs")
        return output_dir

    def get_model_path(self) -> str:
        # Instance the correct object for inference
        run_name = (
            self.config.run_name if self.config.run_name else self.find_latest_run()
        )
        run_dir = os.path.join(self.config.experiment_dir, run_name)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"{checkpoint_dir} doesn't exist")

        checkpoint_filepaths = glob.glob(
            os.path.join(checkpoint_dir, "*." + self.backend_engine.ext)
        )
        if len(checkpoint_filepaths) == 0:
            raise FileNotFoundError(
                f"No '.{self.backend_engine.ext}' files found in {checkpoint_dir}"
            )

        return checkpoint_filepaths[-1]  # For now get the latest checkpoint

    def find_latest_run(self):
        """
        Returns the subdirectory in `base_dir` with the highest numeric suffix in the format <prefix>_<number>.
        Example: among ['run_1', 'run_2', 'run_3'], returns 'run_3'.
        """
        max_suffix = -1
        latest_run = None

        for name in os.listdir(self.config.experiment_dir):
            full_path = os.path.join(self.config.experiment_dir, name)
            if not os.path.isdir(full_path):
                continue

            if "_" in name:
                prefix, suffix = name.rsplit("_", 1)
                if suffix.isdigit():
                    num = int(suffix)
                    if num > max_suffix:
                        max_suffix = num
                        latest_run = name

        return latest_run

    def run(self):
        """
        Runs the image generation process with the user settings.
        """
        datamodule = instantiate(self.config.data_module, _convert_="all")
        datamodule.prepare_data()
        # test_loader = datamodule.test_dataloader()
        # batch = next(iter(test_loader))

        output_dir = (
            self.config.output_dir if self.config.output_dir else self.get_output_dir()
        )
        os.makedirs(output_dir, exist_ok=True)

        model_path = (
            self.config.best_model_path
            if self.config.best_model_path
            else self.get_model_path()
        )

        samples = self.backend_engine(model_path, self.config.num_images)
        output_path = os.path.join(output_dir, "output.png")
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
