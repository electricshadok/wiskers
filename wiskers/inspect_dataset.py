import argparse

import lightning as L

from wiskers.common.commands.utils import load_config
from wiskers.utils import get_data_module


class InspectDatasetCLI:
    """
    Command-line interface for loading dataset and print its details.

    Args:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the InspectDatasetCLI.
        """
        self.config = load_config(config_path)

        L.seed_everything(seed=self.config.seed, workers=True)

        self.datamodule = get_data_module(
            self.config.data_module_type,
            **self.config.data_module,
        )

    def run(self):
        # TODO
        print(self.config.data_module_type)

        self.datamodule.prepare_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training script with a given configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    cmd = InspectDatasetCLI(args.config)
    cmd.run()
