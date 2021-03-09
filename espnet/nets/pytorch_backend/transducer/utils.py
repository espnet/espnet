"""Utility functions for transducer models."""

import os

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

    ys_in_pad = pad_list([torch.cat([blank, y], dim=0) for y in ys], blank_id)
    ys_out_pad = pad_list([torch.cat([y, blank], dim=0) for y in ys], ignore_id)

    target = pad_list(ys, blank_id).type(torch.int32).to(device)
    target_len = torch.IntTensor([y.size(0) for y in ys]).to(device)

    if torch.is_tensor(hlens):
        if hlens.dim() > 1:
            hs = [h[h != 0] for h in hlens]
            hlens = list(map(int, [h.size(0) for h in hs]))
        else:
            hlens = list(map(int, hlens))

    pred_len = torch.IntTensor(hlens).to(device)

    return ys_in_pad, ys_out_pad, target, pred_len, target_len


def valid_aux_task_layer_list(aux_layer_ids, enc_num_layers):
    """Check whether input list of auxiliary layer ids is valid.

       Return the valid list sorted with duplicated removed.

    Args:
        aux_layer_ids (list): Auxiliary layers ids
        enc_num_layers (int): Number of encoder layers

    Returns:
        valid (list): Validated list of layers for auxiliary task

    """
    if (
        not isinstance(aux_layer_ids, list)
        or not aux_layer_ids
        or not all(isinstance(layer, int) for layer in aux_layer_ids)
    ):
        raise ValueError("--aux-task-layer-list argument takes a list of layer ids.")

    sorted_list = sorted(aux_layer_ids, key=int, reverse=False)
    valid = list(filter(lambda x: 0 <= x < enc_num_layers, sorted_list))

    if sorted_list != valid:
        raise ValueError(
            "Provided list of layer ids for auxiliary task is incorrect. "
            "IDs should be between [0, %d]" % (enc_num_layers - 1)
        )

    return valid


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


def select_lm_state(lm_states, idx, lm_layers, is_wordlm):
    """Get LM state from batch for given id.

    Args:
        lm_states (list or dict): batch of LM states
        idx (int): index to extract state from batch state
        lm_layers (int): number of LM layers
        is_wordlm (bool): whether provided LM is a word-LM

    Returns:
       idx_state (dict): LM state for given id

    """
    if is_wordlm:
        idx_state = lm_states[idx]
    else:
        idx_state = {}

        idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
        idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state


def create_lm_batch_state(lm_states_list, lm_layers, is_wordlm):
    """Create batch of LM states.

    Args:
        lm_states (list or dict): list of individual LM states
        lm_layers (int): number of LM layers
        is_wordlm (bool): whether provided LM is a word-LM

    Returns:
       batch_states (list): batch of LM states

    """
    if is_wordlm:
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
    """Check state and left pad or trim if necessary.

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


def check_batch_state(state, max_len, pad_token):
    """Check batch of states and left pad or trim if necessary.

    Args:
        state (list): list of of L decoder states (B, ?, dec_dim)
        max_len (int): maximum length authorized
        pad_token (int): padding token id

    Returns:
        final (list): list of L decoder states (B, pred_len, dec_dim)

    """
    final_dims = (len(state), max_len, state[0].size(1))
    final = state[0].data.new(*final_dims).fill_(pad_token)

    for i, s in enumerate(state):
        curr_len = s.size(0)

        if curr_len < max_len:
            final[i, (max_len - curr_len) : max_len, :] = s
        else:
            final[i, :, :] = s[(curr_len - max_len) :, :]

    return final


def custom_torch_load(model_path, model, training=True):
    """Load transducer model modules and parameters with training-only ones removed.

    Args:
        model_path (str): Model path
        model (torch.nn.Module): The model with pretrained modules

    """
    if "snapshot" in os.path.basename(model_path):
        model_state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )["model"]
    else:
        model_state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )

    if not training:
        model_state_dict = {
            k: v for k, v in model_state_dict.items() if not k.startswith("aux")
        }

    model.load_state_dict(model_state_dict)

    del model_state_dict
