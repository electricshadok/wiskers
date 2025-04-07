import argparse
import os
import pathlib
import shutil

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from safetensors.torch import save_model

from wiskers.common.commands.utils import load_config
from wiskers.common.datasets.cifar10 import CIFAR10
from wiskers.common.datasets.cifar10_subset import CIFAR10Subset
from wiskers.vae.vae_module import VAEModule


class TrainCLI:
    """
    Command-line interface for training a PyTorch Lightning model.

    Args:
        config_path (str): Path to the configuration file.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the TrainCLI.

        Args:
            config_path (str): Path to the configuration file.
        """
        # Load the configuration
        self.config = load_config(config_path)

        # Initialize random number generators
        L.seed_everything(seed=self.config.seed, workers=True)

        # Prepare callbacks, loggers, model and data loaders
        self.logger = TensorBoardLogger(**self.config.tensor_board_logger)
        run_name = os.path.basename(self.logger.log_dir)
        checkpoint_dir = os.path.join(self.config.best_models_dir, run_name)

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

        self.checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.2f}",
        )

        self.callbacks = [self.checkpoint_callback]
        if self.config.earlystopping:
            self.callbacks.append(EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5))

        self.model = VAEModule(
            **self.config.module.model,
            **self.config.module.optimizer,
        )
        if self.config.data_module_type == "cifar10":
            self.datamodule = CIFAR10(**self.config.data_module)
        elif self.config.data_module_type.startswith("cifar10:"):
            category_name = self.config.data_module_type.split(":")[1]
            self.datamodule = CIFAR10Subset(**self.config.data_module, category_name=category_name)
        else:
            raise ValueError(f"Unsupported data_module_type: {self.config.data_module_type}")

    def run(self):
        """
        Runs the training and testing of the model using the user settings.
        """
        # Train the model
        trainer = L.Trainer(logger=self.logger, callbacks=self.callbacks, **self.config.trainer)
        trainer.fit(model=self.model, datamodule=self.datamodule)

        if hasattr(self.config.trainer, "fast_dev_run"):
            # Early exit in development mode
            return

        # Test the model
        trainer.test(datamodule=self.datamodule)

        # Export best ONNX
        if self.config.export_onnx:
            best_model_path = self.checkpoint_callback.best_model_path
            onnx_path = pathlib.Path(best_model_path).with_suffix(".onnx")
            self.model.to_onnx(onnx_path, export_params=True)
            print(f"Saved {onnx_path}")

        # Export best safetensors
        if self.config.export_safetensors:
            best_model_path = self.checkpoint_callback.best_model_path
            st_path = pathlib.Path(best_model_path).with_suffix(".safetensors")
            save_model(self.model, st_path)
            print(f"Saved {st_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training script with a given configuration file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    cmd = TrainCLI(args.config)
    cmd.run()
