import argparse

import lightning as L
from hydra.utils import instantiate

from wiskers.common.commands.utils import load_config


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

        self.datamodule = instantiate(self.config.data_module, _convert_="all")

    def run(self):
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.datamodule.setup("test")
        self.datamodule.setup("predict")

        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()

        # Print dataset lengths and batch info
        for name, loader in [
            ("Train", train_loader),
            ("Val", val_loader),
            ("Test", test_loader),
        ]:
            ds = loader.dataset
            print(f"{name} dataset length: {len(ds)}")
            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                for i, elem in enumerate(batch):
                    shape_info = (
                        f", shape={tuple(elem.shape)}" if hasattr(elem, "shape") else ""
                    )
                    print(f"  [{i}] {type(elem).__name__}{shape_info}")
            else:
                shape_info = (
                    f", shape={tuple(batch.shape)}" if hasattr(batch, "shape") else ""
                )
                print(f"  {type(batch).__name__}{shape_info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the training script with a given configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()
    cmd = InspectDatasetCLI(args.config)
    cmd.run()
