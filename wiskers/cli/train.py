import argparse
import os
import pathlib
import shutil
from typing import List, Optional

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import OmegaConf
from safetensors.torch import save_model

from wiskers.cli.utils import load_config


class TrainCLI:
    """
    Lightweight interface for training a PyTorch Lightning model.

    Executes model and datamodule fit without experiment pipeline overhead
    (no loggers, callbacks, profiler, testing, or artifact export).

    Args:
        config_name (str): Configuration name (e.g. 'training/carracing_world_models') or path.
        config_path (str): Root configuration directory (default: 'configs').
        overrides (Optional[List[str]]): CLI configuration overrides.
    """

    def __init__(
        self,
        config_name: str,
        config_path: str = "./configs",
        overrides: Optional[List[str]] = None,
    ) -> None:
        self.config = load_config(
            config_name, config_path=config_path, overrides=overrides
        )
        L.seed_everything(seed=self.config.seed, workers=True)

        self.model = instantiate(self.config.module, _convert_="all")
        self.datamodule = instantiate(self.config.data_module, _convert_="all")

    def run(self, fast_dev_run: bool = True) -> L.Trainer:
        """
        Runs model training fit without pipeline overhead.

        Args:
            fast_dev_run (bool): If True, runs a single batch sanity check.

        Returns:
            L.Trainer: The PyTorch Lightning trainer instance.
        """
        trainer_args = dict(self.config.trainer)
        trainer = L.Trainer(
            logger=False,
            callbacks=[],
            profiler=None,
            fast_dev_run=fast_dev_run,
            **trainer_args,
        )
        trainer.fit(model=self.model, datamodule=self.datamodule)
        return trainer


class TrainPipelineCLI(TrainCLI):
    """
    Full-featured experiment training pipeline.

    Extends TrainCLI with experiment logging, checkpoints, early stopping,
    profiling, testing, and model export (ONNX, safetensors).

    Args:
        config_name (str): Configuration name (e.g. 'training/carracing_world_models') or path.
        config_path (str): Root configuration directory (default: 'configs').
        overrides (Optional[List[str]]): CLI configuration overrides.
    """

    def __init__(
        self,
        config_name: str,
        config_path: str = "./configs",
        overrides: Optional[List[str]] = None,
    ) -> None:
        super().__init__(config_name, config_path=config_path, overrides=overrides)
        self.logger = None
        self.checkpoint_callback = None
        self.callbacks = []

    def _setup_logging_and_callbacks(self) -> None:
        """Sets up the logger and callbacks for training."""
        if self.logger is not None:
            return

        self.logger = TensorBoardLogger(**self.config.tensor_board_logger)
        os.makedirs(self.logger.log_dir, exist_ok=True)

        # Save fully resolved configuration for experiment reproducibility
        config_save_path = os.path.join(self.logger.log_dir, "config.yaml")
        OmegaConf.save(self.config, config_save_path)

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

    def _export_models(self) -> None:
        """Exports best model checkpoints to ONNX and safetensors if configured."""
        if self.config.export_onnx:
            assert self.checkpoint_callback is not None
            best_model_path = self.checkpoint_callback.best_model_path
            onnx_path = pathlib.Path(best_model_path).with_suffix(".onnx")
            self.model.to_onnx(onnx_path, export_params=True)
            print(f"Saved {onnx_path}")

        if self.config.export_safetensors:
            assert self.checkpoint_callback is not None
            best_model_path = self.checkpoint_callback.best_model_path
            st_path = pathlib.Path(best_model_path).with_suffix(".safetensors")
            save_model(self.model, st_path)
            print(f"Saved {st_path}")

    def run(self, quick_run: bool = False) -> L.Trainer:
        """
        Runs the full training pipeline: fit, test, and artifact export.

        Args:
            quick_run (bool): If True, runs a few quick batches/epochs with testing and export.

        Returns:
            L.Trainer: The PyTorch Lightning trainer instance.
        """
        self._setup_logging_and_callbacks()
        trainer_args = dict(self.config.trainer)

        if quick_run:
            trainer_args.update(
                {
                    "max_epochs": 1,
                    "limit_train_batches": 10,
                    "limit_val_batches": 5,
                    "limit_test_batches": 5,
                }
            )

        trainer = L.Trainer(
            logger=self.logger,
            callbacks=self.callbacks,
            profiler=PyTorchProfiler(),
            fast_dev_run=False,
            **trainer_args,
        )
        trainer.fit(model=self.model, datamodule=self.datamodule)
        trainer.test(datamodule=self.datamodule)
        self._export_models()
        return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training with Hydra configuration."
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
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single batch to test configuration without pipeline",
    )
    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Run a short real training with pipeline + test + export",
    )
    args, extra_args = parser.parse_known_args()

    if args.fast_dev_run:
        cmd = TrainCLI(
            args.config_name, config_path=args.config_path, overrides=extra_args
        )
        cmd.run(fast_dev_run=True)
    else:
        cmd = TrainPipelineCLI(
            args.config_name, config_path=args.config_path, overrides=extra_args
        )
        cmd.run(quick_run=args.quick_run)
