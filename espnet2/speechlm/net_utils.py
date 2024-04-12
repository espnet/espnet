import torch

def length_mask(lengths: torch.Tensor) -> torch.Tensor:
    assert lengths.dim() == 1
    mask = torch.le(
        torch.arange(lengths.max(), detvice=lengths.device).unsqueeze(1),
        lengths.unsqueeze(0)
    ).bool()
    return mask

def causal_mask(lengths: torch.Tensor) -> torch.Tensor:
    assert lengths.dim() == 1
    max_len = lengths.max()
    axis = torch.arange(max_len, device=lengths.device)
    mask = torch.le(axis.unsqueeze(0), axis.unsqueeze(1)).bool()
    return mask
