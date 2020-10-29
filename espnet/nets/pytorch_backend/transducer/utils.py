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
        x (list): set of hypotheses
        subset (list): subset of hypotheses

    Returns:
       final (list): new set

    """
    final = []

    for x_ in x:
        if any(x_.yseq == sub.yseq for sub in subset):
            continue
        final.append(x_)

    return final


def select_lm_state(lm_states, idx, lm_type, lm_layers):
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
        idx_state = lm_states[idx]
    else:
        idx_state = {}

        idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
        idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state


def create_lm_batch_state(lm_states_list, lm_type, lm_layers):
    """Create batch of LM states.

    Args:
        lm_states (list or dict): list of individual LM states
        lm_type (str): type of LM
        lm_layers (int): number of LM layers

    Returns:
       batch_states (list): batch of LM states

    """
    if lm_type == "wordlm":
        batch_states = lm_states_list
    else:
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


def init_lm_state(lm_model):
    """Initialize LM state.

    Args:
        lm_model (torch.nn.Module): LM module

    Returns:
        lm_state (dict): initial LM state

    """
    lm_layers = len(lm_model.rnn)
    lm_units_typ = lm_model.typ
    lm_units = lm_model.n_units

    p = next(lm_model.parameters())

    h = [
        torch.zeros(lm_units).to(device=p.device, dtype=p.dtype)
        for _ in range(lm_layers)
    ]

    lm_state = {"h": h}

    if lm_units_typ == "lstm":
        lm_state["c"] = [
            torch.zeros(lm_units).to(device=p.device, dtype=p.dtype)
            for _ in range(lm_layers)
        ]

    return lm_state


def recombine_hyps(hyps):
    """Recombine hypotheses with equivalent output sequence.

    Args:
        hyps (list): list of hypotheses

    Returns:
       final (list): list of recombined hypotheses

    """
    final = []

    for hyp in hyps:
        seq_final = [f.yseq for f in final if f.yseq]

        if hyp.yseq in seq_final:
            seq_pos = seq_final.index(hyp.yseq)

            final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
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


def check_state(state, max_len, pad_token):
    """Left pad or trim state according to max_len.

    Args:
        state (list): list of of L decoder states (in_len, dec_dim)
        max_len (int): maximum length authorized
        pad_token (int): padding token id

    Returns:
        final (list): list of L padded decoder states (1, max_len, dec_dim)

    """
    if state is None or max_len < 1 or state[0].size(1) == max_len:
        return state

    curr_len = state[0].size(1)

    if curr_len > max_len:
        trim_val = int(state[0].size(1) - max_len)

        for i, s in enumerate(state):
            state[i] = s[:, trim_val:, :]
    else:
        layers = len(state)
        ddim = state[0].size(2)

        final_dims = (1, max_len, ddim)
        final = [state[0].data.new(*final_dims).fill_(pad_token) for _ in range(layers)]

        for i, s in enumerate(state):
            final[i][:, (max_len - s.size(1)) : max_len, :] = s

        return final

    return state


def pad_batch_state(state, pred_length, pad_token):
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
