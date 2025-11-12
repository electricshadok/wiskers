import pytest
import torch

from wiskers.common.blocks.quantizer import Codebook, EMA_Codebook, VectorQuantizer


@pytest.mark.parametrize(
    "num_codes, code_dim, num_vectors", [(512, 64, 128), (1024, 128, 64)]
)
def test_codebook_shape(num_codes, code_dim, num_vectors):
    codebook = Codebook(num_codes=num_codes, code_dim=code_dim)
    z = torch.randn(num_vectors, code_dim)

    distances = codebook.compute_distances(z)
    indices = torch.randint(0, num_codes, (num_vectors,))
    embeddings = codebook.lookup(indices)

    assert distances.shape == (num_vectors, num_codes), "Distance matrix shape mismatch"
    assert embeddings.shape == (
        num_vectors,
        code_dim,
    ), "Embedding lookup shape mismatch"


@pytest.mark.parametrize(
    "num_codes, code_dim, num_vectors", [(512, 64, 128), (1024, 128, 64)]
)
def test_ema_codebook_shape(num_codes, code_dim, num_vectors):
    codebook = EMA_Codebook(num_codes=num_codes, code_dim=code_dim)
    z = torch.randn(num_vectors, code_dim)

    distances = codebook.compute_distances(z)
    indices = torch.randint(0, num_codes, (num_vectors,))
    embeddings = codebook.lookup(indices)
    codebook.update(z, indices)

    assert distances.shape == (num_vectors, num_codes), "Distance matrix shape mismatch"
    assert embeddings.shape == (
        num_vectors,
        code_dim,
    ), "Embedding lookup shape mismatch"


@pytest.mark.parametrize(
    "batch_size, code_dim, height, width, num_codes, use_ema",
    [
        (4, 64, 16, 16, 128, False),
        (8, 128, 8, 8, 256, True),
    ],
)
def test_vector_quantizer(batch_size, code_dim, height, width, num_codes, use_ema):
    net = VectorQuantizer(num_codes=num_codes, code_dim=code_dim, use_ema=use_ema)
    x = torch.randn(batch_size, code_dim, height, width)

    z_q_st, vq_loss, indices = net(x)

    assert z_q_st.shape == x.shape, "Output shape must match input shape"
    assert vq_loss.dim() == 0, "Loss should be a scalar tensor"
    assert torch.isfinite(vq_loss), "Loss should not contain NaN or Inf values"
    assert indices.shape[0] == batch_size * height * width, "Indices shape mismatch"
