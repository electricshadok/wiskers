import torch


def codebook_usage_metrics(
    indices: torch.Tensor, num_codes: int
) -> dict[str, torch.Tensor]:
    """
    Compute codebook usage statistics for a VQ-VAE.

    Args:
        indices: Tensor of quantizer output indices.
        num_codes: Size of the codebook.

    Returns:
        Dict containing perplexity and number of dead codes.
    """
    counts = torch.bincount(indices, minlength=num_codes).float()
    total = counts.sum()

    if total <= 0:
        zero = torch.tensor(0.0, device=counts.device)
        return {
            "perplexity": zero,
            "dead_codes": torch.tensor(
                counts.numel(), device=counts.device, dtype=torch.int64
            ),
        }

    probs = counts / total
    nonzero_probs = probs[probs > 0]
    perplexity = torch.exp(-(nonzero_probs * nonzero_probs.log()).sum())
    dead_codes = (counts == 0).sum()

    return {"perplexity": perplexity, "dead_codes": dead_codes}
