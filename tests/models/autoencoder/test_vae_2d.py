import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder
from wiskers.models.autoencoder.vae_2d import VAE2D


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 3, 32, 32),
        (8, 1, 1, 32, 32),
    ],
)
def test_vae2D(batch_size, in_channels, out_channels, height, width):
    encoder = CNNEncoder(
        in_channels=in_channels,
        num_heads=2,
        block_channels=[32, 64, 128],
        block_attentions=[True, True, True],
    )
    decoder = CNNDecoder(
        out_channels=out_channels,
        num_heads=2,
        block_channels=[128, 64, 32],
        block_attentions=[True, True, True],
    )
    net = VAE2D(
        image_size=(height, width),
        encoder=encoder,
        decoder=decoder,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    out_x, mu, logvar = net(x)

    assert out_x.shape == (
        batch_size,
        out_channels,
        height,
        width,
    )
    assert out_x.dtype == x.dtype

    mid_c, mid_h, mid_w = net.get_latent_shape()

    assert mu.shape == (batch_size, mid_c, mid_h, mid_w)
    assert logvar.shape == (batch_size, mid_c, mid_h, mid_w)


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width",
    [
        (4, 3, 3, 32, 32),
    ],
)
def test_vae2D_to_onnx(batch_size, in_channels, out_channels, height, width, tmp_path):
    encoder = CNNEncoder(
        in_channels=in_channels,
        num_heads=2,
        block_channels=[32, 64, 128],
        block_attentions=[True, True, True],
    )
    decoder = CNNDecoder(
        out_channels=out_channels,
        num_heads=2,
        block_channels=[128, 64, 32],
        block_attentions=[True, True, True],
    )
    net = VAE2D(
        image_size=(height, width),
        encoder=encoder,
        decoder=decoder,
    )
    x = torch.randn(batch_size, in_channels, height, width)

    # Export the model
    onnx_file_path = tmp_path / "vae_2D.onnx"
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
