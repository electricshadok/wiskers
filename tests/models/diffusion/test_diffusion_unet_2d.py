import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from wiskers.models.diffusion.unet_2d import UNet2D


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 3, 32, 32, 512),
        (8, 1, 1, 32, 32, 512),
    ],
)
def test_unet2D(batch_size, in_channels, out_channels, height, width, time_dim):
    net = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        num_heads=2,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    t = torch.randint(0, 10, (batch_size,), dtype=torch.long)
    out_x = net(x, t)

    assert isinstance(out_x, torch.Tensor)
    assert out_x.shape == (
        batch_size,
        out_channels,
        height,
        width,
    )
    assert out_x.dtype == torch.float32


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, time_dim",
    [
        (4, 3, 3, 32, 32, 512),
    ],
)
def test_unet2D_to_onnx(batch_size, in_channels, out_channels, height, width, time_dim, tmp_path):
    net = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        num_heads=2,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    t = torch.randint(0, 10, (batch_size,), dtype=torch.long)

    # Export the model
    onnx_file_path = tmp_path / "unet2D.onnx"
    torch.onnx.export(
        net,
        (x, t),
        onnx_file_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input", "time_step"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Check the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)

    # Run the model on ONNX Runtime
    ort_session = ort.InferenceSession(onnx_file_path)
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy(), ort_session.get_inputs()[1].name: t.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Verify the output shape
    assert isinstance(ort_outs[0], np.ndarray)
    assert ort_outs[0].shape == (batch_size, out_channels, height, width)
