import pytest
import torch

from wiskers.common.blocks.vq import VectorQuantizer


@pytest.mark.parametrize(
    "batch_size, code_dim, height, width, num_codes",
    [
        (4, 64, 16, 16, 512),
        (8, 128, 8, 8, 1024),
    ],
)
def test_vector_quantizer(batch_size, code_dim, height, width, num_codes):
    net = VectorQuantizer(num_codes=num_codes, code_dim=code_dim)
    x = torch.randn(batch_size, code_dim, height, width)

    z_q_st, vq_loss, indices = net(x)

    assert z_q_st.shape == x.shape, "Output shape must match input shape"
    assert vq_loss.dim() == 0, "Loss should be a scalar tensor"
    assert torch.isfinite(vq_loss), "Loss should not contain NaN or Inf values"
    assert indices.shape[0] == batch_size * height * width, "Indices shape mismatch"
