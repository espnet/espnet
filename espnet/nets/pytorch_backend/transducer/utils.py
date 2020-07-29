"""Utility functions for transducer models."""

import numpy as np
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
        x (list): token id sequence
        pref (list): token id sequence

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
    """Remove elements of subset if corresponding token id sequence exist in x.

    Args:
        x (list): set of hypothesis
        subset (list): subset of hypothesis

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
    """Get LM state from batch for given id.

    Args:
        lm_states (list or dict): batch of LM states
        idx (int): index to extract state from batch state
        lm_type (str): type of LM
        lm_layers (int): number of LM layers

    Returns:
       idx_state (dict): LM state for given id

    """
    if lm_type == "wordlm":
        return lm_states[idx]

    idx_state = {}

    idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
    idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state


def get_batch_lm_states(lm_states_list, lm_type, lm_layers):
    """Create batch of LM states.

    Args:
        lm_states (list or dict): list of individual LM states
        lm_type (str): type of LM
        lm_layers (int): number of LM layers

    Returns:
       beam_states (list): batch of LM states

    """
    if lm_type == "wordlm":
        return lm_states_list

    batch_states = {}

    batch_states["c"] = [
        torch.stack([state["c"][layer] for state in lm_states_list])
        for layer in range(lm_layers)
    ]
    batch_states["h"] = [
        torch.stack([state["h"][layer] for state in lm_states_list])
        for layer in range(lm_layers)
    ]

    return batch_states


def recombine_hyps(hyps):
    """Recombine hypothesis with equivalent output sequence.

    Args:
        hyps (list): list of hypothesis

    Returns:
       final (list): list of recombined hypothesis

    """
    final = []

    for i, hyp in enumerate(hyps):
        seq_final = [f["yseq"] for f in final if f["yseq"]]

        if hyp["yseq"] in seq_final:
            dict_pos = seq_final.index(hyp["yseq"])

            final[dict_pos]["score"] = np.logaddexp(
                final[dict_pos]["score"], hyp["score"]
            )
        else:
            final.append(hyp)

    return hyps


def pad_sequence(seqlist, pad_token):
    """Left pad list of token id sequences.

    Args:
        seqlist (list): list of token id sequences
        pad_token (int): padding token id

    Returns:
        final (list): list of padded token id sequences

    """
    maxlen = max(len(x) for x in seqlist)

    final = [([pad_token] * (maxlen - len(x))) + x for x in seqlist]

    return final


def pad_state(state, pred_length, pad_token):
    """Left pad batch of states and trim if necessary.

    Args:
        state (list): list of of L decoder states (B, ?, dec_dim)
        pred_length (int): maximum length authorized (trimming)
        pad_token (int): padding token id

    Returns:
        final (list): list of L padded decoder states (B, pred_length, dec_dim)

    """
    batch = len(state)
    maxlen = max([s.size(0) for s in state])
    ddim = state[0].size(1)

    final_dims = (batch, maxlen, ddim)
    final = state[0].data.new(*final_dims).fill_(pad_token)

    for i, s in enumerate(state):
        final[i, (maxlen - s.size(0)) : maxlen, :] = s

    trim_val = final[0].size(0) - (pred_length - 1)

    return final[:, trim_val:, :]
