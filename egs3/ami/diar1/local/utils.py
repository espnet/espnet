import torch

def len2mask(length, max_len=None, dtype=None):

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
