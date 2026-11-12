import argparse
from typing import Any

import lightning as L
from hydra.utils import instantiate

from wiskers.cli.utils import load_config


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
    Command-line interface for loading dataset and printing its details.

    Args:
        config_name (str): Configuration name (e.g. 'dataset/carracing') or path.
        config_path (str): Root configuration directory (default: 'configs').
    """

    def __init__(self, config_name: str, config_path: str = "./configs") -> None:
        self.config = load_config(config_name, config_path=config_path)

        L.seed_everything(seed=self.config.seed, workers=True)

        self.datamodule = instantiate(self.config.data_module, _convert_="all")

    def run(self):
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.datamodule.setup("test")
        self.datamodule.setup("predict")

        loaders = []
        for name, method_name in [
            ("Train", "train_dataloader"),
            ("Val", "val_dataloader"),
            ("Test", "test_dataloader"),
        ]:
            try:
                method = getattr(self.datamodule, method_name)
                loaders.append((name, method()))
            except Exception:
                continue

        # Print dataset lengths and batch info
        for name, loader in loaders:
            ds = getattr(loader, "dataset", None)
            if ds is not None and hasattr(ds, "__len__"):
                print(f"{name} dataset length: {len(ds)}")
            batch = next(iter(loader))
            print(f"{name} batch description:")
            describe_element(batch, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and describe trajectory dataset batches."
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
        help="Configuration name relative to config-path (e.g. dataset/carracing)",
    )
    args = parser.parse_args()
    cmd = InspectDatasetCLI(args.config_name, config_path=args.config_path)
    cmd.run()
