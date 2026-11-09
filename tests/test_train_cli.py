import os
import tempfile
from unittest.mock import patch

from wiskers.cli.train import TrainCLI


def test_train_cli_fast_dev_run_does_not_create_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = TrainCLI("configs/training/carracing_world_models.yaml")
        cli.config.tensor_board_logger.save_dir = tmpdir

        # Nothing should be created on initialization
        assert cli.logger is None
        assert cli.checkpoint_callback is None
        assert os.listdir(tmpdir) == []

        # Running in fast_dev_run mode should not create checkpoints or log dir
        with patch("lightning.Trainer.fit"):
            cli.run(fast_dev_run=True)
            assert os.listdir(tmpdir) == []


def test_train_cli_regular_run_creates_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = TrainCLI("configs/training/carracing_world_models.yaml")
        cli.config.tensor_board_logger.save_dir = tmpdir

        # Nothing should be created before run()
        assert os.listdir(tmpdir) == []

        # Running quick_run / regular setup creates checkpoint directory
        with patch("lightning.Trainer.fit"), patch("lightning.Trainer.test"):
            cli.run(quick_run=True)
            assert cli.logger is not None
            assert cli.checkpoint_callback is not None
            checkpoint_dir = os.path.join(cli.logger.log_dir, "checkpoints")
            assert os.path.exists(checkpoint_dir)

