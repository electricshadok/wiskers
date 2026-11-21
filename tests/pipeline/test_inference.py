from types import SimpleNamespace

import lightning as L
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from onnx import TensorProto, helper
from safetensors.torch import save_model
from torch.utils.data import DataLoader, TensorDataset

from wiskers.data.buffer import RolloutBuffer
from wiskers.envs.inverted_pendulum.policies import InvertedPendulumNeuralPolicy
from wiskers.pipeline.backends import CheckpointInference, ONNXInference, SafeTensorInference
from wiskers.pipeline.predict import PredictCLI


class _DummyLightningModule(L.LightningModule):
    def __init__(self, in_features: int = 4, out_features: int = 2) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self.forward(batch[0]).pow(2).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def test_checkpoint_inference_loads_eval_lightning_module(tmp_path):
    model = _DummyLightningModule()
    loader = DataLoader(TensorDataset(torch.randn(2, 4)), batch_size=2)
    trainer = L.Trainer(logger=False, enable_checkpointing=False, max_steps=1, enable_model_summary=False)
    trainer.fit(model, train_dataloaders=loader)

    ckpt_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    backend = CheckpointInference(model_class=_DummyLightningModule)
    loaded = backend.load(str(ckpt_path))

    assert backend.ext == "ckpt"
    assert isinstance(loaded, _DummyLightningModule)
    assert not loaded.training

    model.eval()
    x = torch.randn(3, 4)
    with torch.inference_mode():
        assert torch.equal(loaded(x), model(x))


def test_safetensor_inference_loads_eval_module(tmp_path):
    torch.manual_seed(0)
    source = nn.Linear(4, 2)
    st_path = tmp_path / "weights.safetensors"
    save_model(source, str(st_path))

    target = nn.Linear(4, 2)
    backend = SafeTensorInference(model=target)
    loaded = backend.load(str(st_path))

    assert backend.ext == "safetensors"
    assert loaded is target
    assert not loaded.training

    source.eval()
    x = torch.randn(3, 4)
    with torch.inference_mode():
        assert torch.equal(loaded(x), source(x))


def test_onnx_inference_loads_session(tmp_path):
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    relu = helper.make_node("Relu", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([relu], "relu", [x_info], [y_info])
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    onnx_path = tmp_path / "relu.onnx"
    onnx.save(onnx_model, str(onnx_path))

    backend = ONNXInference(providers=["CPUExecutionProvider"])
    session = backend.load(str(onnx_path))

    assert backend.ext == "onnx"
    x = np.array([[-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    out = session.run(None, {"x": x})[0]
    assert np.allclose(out, np.maximum(x, 0.0))


def test_predict_cli_rolls_out_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(
        MujocoRenderer,
        "render",
        lambda self, *args, **kwargs: np.zeros((self.height, self.width, 3), dtype=np.uint8),
    )

    policy = InvertedPendulumNeuralPolicy(image_size=(3, 64, 64))
    st_path = tmp_path / "pendulum_policy.safetensors"
    save_model(policy, str(st_path))

    cfg_path = tmp_path / "predict.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seed: 0",
                f"checkpoint_path: {st_path}",
                "eval_episodes: 2",
                "transform:",
                "  _target_: wiskers.data.transforms.EpisodeToTensor",
                "  image_size: [3, 64, 64]",
                "  obs_range: [0.0, 1.0]",
                "data_module:",
                "  collector:",
                "    env:",
                "      _target_: wiskers.envs.inverted_pendulum.env.InvertedPendulumEnv",
                "      image_size: [64, 64]",
                "    max_steps: 5",
                "backend:",
                "  _target_: wiskers.pipeline.backends.SafeTensorInference",
                "  model:",
                "    _target_: wiskers.envs.inverted_pendulum.policies.InvertedPendulumNeuralPolicy",
                "    image_size: ${transform.image_size}",
            ]
        )
    )

    cli = PredictCLI(str(cfg_path))
    buffer = cli.run()

    assert isinstance(buffer, RolloutBuffer)
    assert buffer.num_episodes == 2
    assert 2 <= buffer.num_steps <= 10
    assert buffer[0].action.shape[1:] == (1,)

    # Controller resolution supports both direct Controller and `.controller` attribute
    assert PredictCLI._resolve_controller(policy) is policy
    wrapper = SimpleNamespace(controller=policy)
    assert PredictCLI._resolve_controller(wrapper) is policy

    with pytest.raises(TypeError, match="must implement Controller or expose a .controller attribute"):
        PredictCLI._resolve_controller(nn.Linear(2, 2))
