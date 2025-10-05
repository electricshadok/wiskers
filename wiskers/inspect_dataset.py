import argparse
from typing import Any

import lightning as L
from hydra.utils import instantiate

from wiskers.common.commands.utils import load_config


def describe_element(elem: Any, indent: int = 0, max_items: int = 3) -> None:
    """Recursively describe element types and shapes."""
    prefix = " " * indent
    if isinstance(elem, dict):
        print(f"{prefix}dict with {len(elem)} keys:")
        for k, v in elem.items():
            print(f"{prefix}  key='{k}':")
            describe_element(v, indent + 4, max_items=max_items)
    elif isinstance(elem, (list, tuple)):
        print(f"{prefix}{type(elem).__name__} of length {len(elem)}")
        for i, v in enumerate(elem[:max_items]):
            print(f"{prefix}  [{i}]:")
            describe_element(v, indent + 4, max_items=max_items)
        if len(elem) > max_items:
            print(f"{prefix}  ... ({len(elem) - max_items} more)")
    else:
        shape_info = f", shape={tuple(elem.shape)}" if hasattr(elem, "shape") else ""
        dtype_info = (
            f", dtype={getattr(elem, 'dtype', None)}" if hasattr(elem, "dtype") else ""
        )
        print(f"{prefix}{type(elem).__name__}{shape_info}{dtype_info}")


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
            print(f"{name} batch description:")
            describe_element(batch, indent=2)


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
