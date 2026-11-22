import os
import pathlib
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from wiskers.pipeline.config import load_config
from wiskers.pipeline.train import TrainCLI, TrainPipelineCLI


def test_load_config_defaults_inheritance():
    cfg = load_config("training/car_racing_world_models")
    assert "data_module" in cfg
    assert cfg.data_module.batch_size == 16
    assert cfg.module._target_ == "wiskers.systems.world_models.WorldModels"
    assert cfg.module.mdn_rnn.action_size == 3
    assert "defaults" not in cfg


def test_load_config_franka_kitchen_world_models_override():
    cfg = load_config("training/franka_kitchen_world_models")
    assert "data_module" in cfg
    assert cfg.module._target_ == "wiskers.systems.world_models.WorldModels"
    assert cfg.module.mdn_rnn.action_size == 9
    assert cfg.module.controller.action_size == 9
    assert "defaults" not in cfg


def test_load_config_override_dataset_param():
    test_yaml = """
defaults:
  - dataset: car_racing

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


def test_load_config_cli_overrides():
    cfg = load_config(
        "training/car_racing_world_models",
        overrides=["dataset=breakout", "trainer.max_epochs=42", "data_module.sampler.num_steps=32"],
    )
    assert cfg.data_module.collector.env._target_ == "wiskers.envs.breakout.env.BreakoutEnv"
    assert cfg.trainer.max_epochs == 42
    assert cfg.data_module.sampler.num_steps == 32
    assert cfg.data_module.collector.min_steps == 32


def test_train_cli_fast_dev_run_does_not_create_pipeline_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = TrainCLI("training/car_racing_world_models")
        cli.config.tensor_board_logger.save_dir = tmpdir

        with patch("lightning.Trainer.fit"):
            cli.run(fast_dev_run=True)
            # TrainCLI has no loggers or checkpoints
            assert os.listdir(tmpdir) == []


def test_train_pipeline_cli_creates_folders_and_runs_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = TrainPipelineCLI("training/car_racing_world_models")
        cli.config.tensor_board_logger.save_dir = tmpdir

        # Nothing should be created before run()
        assert cli.logger is None
        assert cli.checkpoint_callback is None
        assert os.listdir(tmpdir) == []

        # Running with quick_run=True sets up pipeline and directories
        with patch("lightning.Trainer.fit"), patch("lightning.Trainer.test"):
            cli.run(quick_run=True)
            assert cli.logger is not None
            assert cli.checkpoint_callback is not None
            checkpoint_dir = os.path.join(cli.logger.log_dir, "checkpoints")
            assert os.path.exists(checkpoint_dir)

            # Fully resolved config is saved for reproducibility
            saved_config_path = os.path.join(cli.logger.log_dir, "config.yaml")
            assert os.path.exists(saved_config_path)
            loaded = load_config(saved_config_path)
            assert "data_module" in loaded
            assert "module" in loaded


def test_load_config_with_custom_config_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create custom dataset config: <tmpdir>/dataset/custom.yaml
        dataset_dir = os.path.join(tmpdir, "dataset")
        os.makedirs(dataset_dir)
        with open(os.path.join(dataset_dir, "custom.yaml"), "w") as f:
            f.write("seed: 123\ncustom_key: custom_value\n")

        # Create custom training config: <tmpdir>/training/exp.yaml
        train_dir = os.path.join(tmpdir, "training")
        os.makedirs(train_dir)
        with open(os.path.join(train_dir, "exp.yaml"), "w") as f:
            f.write("defaults:\n  - dataset: custom\n  - _self_\nseed: 456\n")

        cfg = load_config("training/exp", config_path=tmpdir)
        assert cfg.custom_key == "custom_value"
        assert cfg.seed == 456


def test_train_pipeline_export_models_reloads_best_checkpoint(tmp_path):
    cli = TrainPipelineCLI("training/car_racing_world_models")
    try:
        best_ckpt_path = tmp_path / "best.ckpt"
        best_state = {k: v.clone() for k, v in cli.model.state_dict().items()}
        torch.save({"state_dict": best_state}, best_ckpt_path)

        # Simulate worse last-epoch weights remaining in cli.model after fit()
        with torch.no_grad():
            for param in cli.model.parameters():
                param.fill_(99.0)

        cli.config.export_onnx = False
        cli.config.export_safetensors = True
        cli.checkpoint_callback = SimpleNamespace(best_model_path=str(best_ckpt_path))
        cli._export_models()

        st_path = pathlib.Path(best_ckpt_path).with_suffix(".safetensors")
        assert st_path.exists()
        exported = load_file(str(st_path))
        for key, tensor in exported.items():
            assert torch.equal(tensor, best_state[key]), f"Mismatch for {key}"
    finally:
        cli._close_collector()
