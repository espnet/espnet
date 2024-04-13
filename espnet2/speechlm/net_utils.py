import torch


def length_mask(lengths: torch.Tensor) -> torch.Tensor:
    assert lengths.dim() == 1
    mask = torch.le(
        torch.arange(lengths.max(), device=lengths.device).unsqueeze(0),
        lengths.unsqueeze(1),
    ).long()
    return mask


def causal_mask(qlen: int, device: torch.device) -> torch.Tensor:
    return torch.ones((qlen, qlen), device=device).tril_(0).unsqueeze(0)
