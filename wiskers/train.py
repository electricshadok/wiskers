import argparse
import os
import pathlib
import shutil

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from safetensors.torch import save_model

from wiskers.common.commands.utils import load_config


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
        checkpoint_dir = os.path.join(self.logger.log_dir, "checkpoints")

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
            self.callbacks.append(
                EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5)
            )

        self.model = instantiate(self.config.module)

        self.datamodule = instantiate(self.config.data_module)

    def run(self, fast_dev_run=False, quick_run=False):
        """
        Runs the training and testing of the model using the user settings.
        """
        trainer_args = dict(self.config.trainer)

        if quick_run:
            # override to run a few quick batches/epochs
            trainer_args.update(
                {
                    "max_epochs": 1,
                    "limit_train_batches": 10,
                    "limit_val_batches": 5,
                    "limit_test_batches": 5,
                }
            )

        # Train the model
        trainer = L.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            profiler=PyTorchProfiler(),
            fast_dev_run=fast_dev_run,
            **trainer_args,
        )
        trainer.fit(model=self.model, datamodule=self.datamodule)

        if fast_dev_run:
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
    parser = argparse.ArgumentParser(
        description="Run the training script with a given configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single batch to test configuration",
    )
    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Run a short real training + test + export",
    )
    args = parser.parse_args()
    cmd = TrainCLI(args.config)
    cmd.run(fast_dev_run=args.fast_dev_run, quick_run=args.quick_run)
