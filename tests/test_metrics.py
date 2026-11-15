import torch

from wiskers.metrics import codebook_usage_metrics


def test_codebook_usage_metrics_uniform() -> None:
    # 4 codes, each used 10 times
    indices = torch.tensor([0, 1, 2, 3] * 10)
    out = codebook_usage_metrics(indices, num_codes=4)

    assert "perplexity" in out
    assert "dead_codes" in out
    assert torch.isclose(out["perplexity"], torch.tensor(4.0))
    assert out["dead_codes"].item() == 0


def test_codebook_usage_metrics_dead_codes() -> None:
    # 4 codes, only code 0 used
    indices = torch.tensor([0, 0, 0, 0])
    out = codebook_usage_metrics(indices, num_codes=4)

    assert torch.isclose(out["perplexity"], torch.tensor(1.0))
    assert out["dead_codes"].item() == 3


def test_codebook_usage_metrics_empty() -> None:
    indices = torch.tensor([], dtype=torch.int64)
    out = codebook_usage_metrics(indices, num_codes=4)

    assert out["perplexity"].item() == 0.0
    assert out["dead_codes"].item() == 4
