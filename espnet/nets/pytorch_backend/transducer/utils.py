"""Utility functions for transducer models."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


def prepare_loss_inputs(ys_pad, hlens, blank_id=0, ignore_id=-1):
    """Prepare tensors for transducer loss computation.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        hlens (torch.Tensor): batch of hidden sequence lengthts (B)
                              or batch of masks (B, 1, Tmax)
        blank_id (int): index of blank label
        ignore_id (int): index of initial padding

    Returns:
        ys_in_pad (torch.Tensor): batch of padded target sequences + blank (B, Lmax + 1)
        target (torch.Tensor): batch of padded target sequences (B, Lmax)
        pred_len (torch.Tensor): batch of hidden sequence lengths (B)
        target_len (torch.Tensor): batch of output sequence lengths (B)

    """
    device = ys_pad.device

    ys = [y[y != ignore_id] for y in ys_pad]

    blank = ys[0].new([blank_id])

    ys_in = [torch.cat([blank, y], dim=0) for y in ys]
    ys_in_pad = pad_list(ys_in, blank_id)

    target = pad_list(ys, blank_id).type(torch.int32)
    target_len = torch.IntTensor([y.size(0) for y in ys])

    if torch.is_tensor(hlens):
        if hlens.dim() > 1:
            hs = [h[h != 0] for h in hlens]
            hlens = list(map(int, [h.size(0) for h in hs]))
        else:
            hlens = list(map(int, hlens))

    pred_len = torch.IntTensor(hlens)

    pred_len = pred_len.to(device)
    target = target.to(device)
    target_len = target_len.to(device)

    return ys_in_pad, target, pred_len, target_len
