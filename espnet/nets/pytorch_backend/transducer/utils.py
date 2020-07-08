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


def is_prefix(x, pref):
    """Check prefix.

    Args:
        x (list): list of predicted id
        pref (list): list of predict id

    Returns:
       (boolean): whether pref is a prefix of x.

    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref)):
        if pref[i] != x[i]:
            return False

    return True


def substract(x, subset):
    """Remove elements of subset if predicted sequences exist in x.

    Args:
        x (list): beam search predicted sequences
        subset (list): beam search predicted sequences

    Returns:
       final (list of dict): new set

    """
    final = []

    for x_ in x:
        if any(x_["yseq"] == sub["yseq"] for sub in subset):
            continue
        final.append(x_)

    return final


def get_idx_lm_state(lm_states, idx, lm_type, lm_layers):
    """Get lm state for given id.

    Args:
        lm_states (list or dict): lm_states for beam
        idx (int): index to extract state from beam state
        lm_type (str): type of lm
        lm_layers (int): number of lm layers

    Returns:
       idx_state (dict): dict of lm state for given id

    """
    if lm_type == "wordlm":
        return lm_states[idx]

    idx_state = {}

    idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
    idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state


def get_beam_lm_states(lm_states_list, lm_type, lm_layers):
    """Create beam lm states.

    Args:
        lm_states (list or dict): list of lm states
        lm_type (str): type of lm
        lm_layers (int): number of lm layers

    Returns:
       beam_states (list): list of lm states for beam

    """
    if lm_type == "wordlm":
        return lm_states_list

    beam_states = {}

    beam_states["c"] = [
        torch.stack([state["c"][layer] for state in lm_states_list])
        for layer in range(lm_layers)
    ]
    beam_states["h"] = [
        torch.stack([state["h"][layer] for state in lm_states_list])
        for layer in range(lm_layers)
    ]

    return beam_states
