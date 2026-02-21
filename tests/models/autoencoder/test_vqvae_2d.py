import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from wiskers.common.blocks.quantizer import VectorQuantizer
from wiskers.models.autoencoder.encoder_decoder import CNNDecoder, CNNEncoder
from wiskers.models.autoencoder.vqvae_2d import VQ_VAE2D


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, use_ema",
    [
        (4, 3, 3, 32, 32, False),  # gradient-based VQ
        (8, 1, 1, 32, 32, True),  # EMA-based VQ
    ],
)
def test_vqvae2D(batch_size, in_channels, out_channels, height, width, use_ema):
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
    latent_shape = encoder.get_latent_shape((height, width))
    latent_channels = latent_shape[0]
    quantizer = VectorQuantizer(
        num_codes=128,
        code_dim=latent_channels,
        beta=0.25,
        use_ema=use_ema,
    )
    net = VQ_VAE2D(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        latent_shape=latent_shape,
    )
    x = torch.randn(batch_size, in_channels, height, width)
    recon_x, vq_loss, indices = net(x)

    assert recon_x.shape == x.shape
    assert recon_x.dtype == x.dtype
    assert vq_loss.ndim == 0, "vq_loss should be scalar"
    assert indices.ndim == 1, "indices should be 1D (flattened)"


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, use_ema",
    [
        (4, 3, 3, 32, 32, False),  # gradient-based VQ
        (4, 3, 3, 32, 32, True),  # EMA-based VQ
    ],
)
def test_vqvae2D_to_onnx(
    batch_size, in_channels, out_channels, height, width, tmp_path, use_ema
):
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
    latent_shape = encoder.get_latent_shape((height, width))
    latent_channels = latent_shape[0]
    quantizer = VectorQuantizer(
        num_codes=128,
        code_dim=latent_channels,
        beta=0.25,
        use_ema=use_ema,
    )
    net = VQ_VAE2D(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        latent_shape=latent_shape,
    )
    x = torch.randn(batch_size, in_channels, height, width)

    # Export the model
    onnx_file_path = tmp_path / "vqvae_2D.onnx"
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
    assert ort_outs[0].shape == x.shape
