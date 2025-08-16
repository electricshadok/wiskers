import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from wiskers.autoencoder.models.autoencoder_2d import Autoencoder2D


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 3, 32, 32),
        (8, 1, 1, 32, 32),
    ],
)
def test_autoencoder2D(batch_size, in_channels, out_channels, height, width):
    z_dim = 64
    net = Autoencoder2D(in_channels, out_channels, num_heads=2, z_dim=z_dim)
    x = torch.randn(batch_size, in_channels, height, width)
    out_x = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        height,
        width,
    )
    assert out_x.dtype == x.dtype


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 3, 32, 32),
    ],
)
def test_autoencoder2D_to_onnx(batch_size, in_channels, out_channels, height, width, tmp_path):
    net = Autoencoder2D(in_channels, out_channels, num_heads=2)
    x = torch.randn(batch_size, in_channels, height, width)

    # Export the model
    onnx_file_path = tmp_path / "autoencoder_2D.onnx"
    torch.onnx.export(
        net,
        x,
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Check the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    # Run the model on ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file_path)
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Verify the output shape
    assert isinstance(ort_outs[0], np.ndarray)
    assert ort_outs[0].shape == (batch_size, out_channels, height, width)
