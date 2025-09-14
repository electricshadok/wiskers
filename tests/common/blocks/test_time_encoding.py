import pytest
import torch

from wiskers.common.blocks.time_encoding import SinusoidalPositionEmbedding


@pytest.mark.parametrize("batch_size, time_dim", [(16, 128), (64, 512)])
def test_time_encoding(batch_size, time_dim):
    step_id = 12
    t = (torch.ones(batch_size) * step_id).long()  # (B)

    pe = SinusoidalPositionEmbedding(time_dim)
    te = pe(t)  # (B, time_dim)

    assert te.shape == (batch_size, time_dim)
    assert te.dtype == torch.float32
