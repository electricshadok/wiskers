import os
import tempfile
from unittest.mock import patch

from wiskers.cli.train import TrainCLI
from wiskers.cli.utils import load_config


def test_load_config_defaults_inheritance():
    cfg = load_config("configs/training/carracing_world_models.yaml")
    assert "data_module" in cfg
    assert cfg.data_module.batch_size == 16
    assert cfg.module._target_ == "wiskers.world_models.WorldModels"
    assert cfg.module.mdn_rnn.action_size == 3
    assert "defaults" not in cfg


def test_load_config_franka_kitchen_world_models_override():
    cfg = load_config("configs/training/franka_kitchen_world_models.yaml")
    assert "data_module" in cfg
    assert cfg.module._target_ == "wiskers.world_models.WorldModels"
    assert cfg.module.mdn_rnn.action_size == 9
    assert cfg.module.controller.action_size == 9
    assert "defaults" not in cfg


def test_load_config_override_dataset_param():
    test_yaml = """
defaults:
  - configs/datasets/carracing.yaml

data_module:
  batch_size: 64
  replay_buffer:
    capacity_episodes: 200
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
        f.write(test_yaml)
        f.flush()
        cfg = load_config(f.name)
        assert cfg.data_module.batch_size == 64
        assert cfg.data_module.replay_buffer.capacity_episodes == 200
        # Untouched default fields are preserved
        assert cfg.data_module.sampler.num_steps == 50


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

