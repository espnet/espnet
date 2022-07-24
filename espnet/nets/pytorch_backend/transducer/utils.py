"""Utility functions for Transducer models."""

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.transducer_decoder_interface import ExtendedHypothesis
from espnet.nets.transducer_decoder_interface import Hypothesis


def get_decoder_input(
    labels: torch.Tensor, blank_id: int, ignore_id: int
) -> torch.Tensor:
    """Prepare decoder input.

    Args:
        labels: Label ID sequences. (B, L)

    Returns:
        decoder_input: Label ID sequences with blank prefix. (B, U)

    """
    device = labels.device

    labels_unpad = [label[label != ignore_id] for label in labels]
    blank = labels[0].new([blank_id])

    decoder_input = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    return decoder_input


def valid_aux_encoder_output_layers(
    aux_layer_id: List[int],
    enc_num_layers: int,
    use_symm_kl_div_loss: bool,
    subsample: List[int],
) -> List[int]:
    """Check whether provided auxiliary encoder layer IDs are valid.

    Return the valid list sorted with duplicates removed.

    Args:
        aux_layer_id: Auxiliary encoder layer IDs.
        enc_num_layers: Number of encoder layers.
        use_symm_kl_div_loss: Whether symmetric KL divergence loss is used.
        subsample: Subsampling rate per layer.

    Returns:
        valid: Valid list of auxiliary encoder layers.

    """
    if (
        not isinstance(aux_layer_id, list)
        or not aux_layer_id
        or not all(isinstance(layer, int) for layer in aux_layer_id)
    ):
        raise ValueError(
            "aux-transducer-loss-enc-output-layers option takes a list of layer IDs."
            " Correct argument format is: '[0, 1]'"
        )

    sorted_list = sorted(aux_layer_id, key=int, reverse=False)
    valid = list(filter(lambda x: 0 <= x < enc_num_layers, sorted_list))

    if sorted_list != valid:
        raise ValueError(
            "Provided argument for aux-transducer-loss-enc-output-layers is incorrect."
            " IDs should be between [0, %d]" % enc_num_layers
        )

    if use_symm_kl_div_loss:
        sorted_list += [enc_num_layers]

        for n in range(1, len(sorted_list)):
            sub_range = subsample[(sorted_list[n - 1] + 1) : sorted_list[n] + 1]
            valid_shape = [False if n > 1 else True for n in sub_range]

            if False in valid_shape:
                raise ValueError(
                    "Encoder layers %d and %d have different shape due to subsampling."
                    " Symmetric KL divergence loss doesn't cover such case for now."
                    % (sorted_list[n - 1], sorted_list[n])
                )

    return valid


def is_prefix(x: List[int], pref: List[int]) -> bool:
    """Check if pref is a prefix of x.

    Args:
        x: Label ID sequence.
        pref: Prefix label ID sequence.

    Returns:
        : Whether pref is a prefix of x.

    """
    if len(pref) >= len(x):
        return False

    for i in range(len(pref) - 1, -1, -1):
        if pref[i] != x[i]:
            return False

    return True


def subtract(
    x: List[ExtendedHypothesis], subset: List[ExtendedHypothesis]
) -> List[ExtendedHypothesis]:
    """Remove elements of subset if corresponding label ID sequence already exist in x.

    Args:
        x: Set of hypotheses.
        subset: Subset of x.

    Returns:
       final: New set of hypotheses.

    """
    final = []

    for x_ in x:
        if any(x_.yseq == sub.yseq for sub in subset):
            continue
        final.append(x_)

    return final


def select_k_expansions(
    hyps: List[ExtendedHypothesis],
    logps: torch.Tensor,
    beam_size: int,
    gamma: float,
    beta: float,
) -> List[ExtendedHypothesis]:
    """Return K hypotheses candidates for expansion from a list of hypothesis.

    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        beam_logp: Log-probabilities for hypotheses expansions.
        beam_size: Beam size.
        gamma: Allowed logp difference for prune-by-value method.
        beta: Number of additional candidates to store.

    Return:
        k_expansions: Best K expansion hypotheses candidates.

    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [(int(k), hyp.score + float(logp)) for k, logp in enumerate(logps[i])]
        k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

        k_expansions.append(
            sorted(
                filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i),
                key=lambda x: x[1],
                reverse=True,
            )[: beam_size + beta]
        )

    return k_expansions


def select_lm_state(
    lm_states: Union[List[Any], Dict[str, Any]],
    idx: int,
    lm_layers: int,
    is_wordlm: bool,
) -> Union[List[Any], Dict[str, Any]]:
    """Get ID state from LM hidden states.

    Args:
        lm_states: LM hidden states.
        idx: LM state ID to extract.
        lm_layers: Number of LM layers.
        is_wordlm: Whether provided LM is a word-level LM.

    Returns:
       idx_state: LM hidden state for given ID.

    """
    if is_wordlm:
        idx_state = lm_states[idx]
    else:
        idx_state = {}

        idx_state["c"] = [lm_states["c"][layer][idx] for layer in range(lm_layers)]
        idx_state["h"] = [lm_states["h"][layer][idx] for layer in range(lm_layers)]

    return idx_state


def create_lm_batch_states(
    lm_states: Union[List[Any], Dict[str, Any]], lm_layers, is_wordlm: bool
) -> Union[List[Any], Dict[str, Any]]:
    """Create LM hidden states.

    Args:
        lm_states: LM hidden states.
        lm_layers: Number of LM layers.
        is_wordlm: Whether provided LM is a word-level LM.

    Returns:
        new_states: LM hidden states.

    """
    if is_wordlm:
        return lm_states

    new_states = {}

    new_states["c"] = [
        torch.stack([state["c"][layer] for state in lm_states])
        for layer in range(lm_layers)
    ]
    new_states["h"] = [
        torch.stack([state["h"][layer] for state in lm_states])
        for layer in range(lm_layers)
    ]

    return new_states


def init_lm_state(lm_model: torch.nn.Module):
    """Initialize LM hidden states.

    Args:
        lm_model: LM module.

    Returns:
        lm_state: Initial LM hidden states.

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


def recombine_hyps(hyps: List[Hypothesis]) -> List[Hypothesis]:
    """Recombine hypotheses with same label ID sequence.

    Args:
        hyps: Hypotheses.

    Returns:
       final: Recombined hypotheses.

    """
    final = []

    for hyp in hyps:
        seq_final = [f.yseq for f in final if f.yseq]

        if hyp.yseq in seq_final:
            seq_pos = seq_final.index(hyp.yseq)

            final[seq_pos].score = np.logaddexp(final[seq_pos].score, hyp.score)
        else:
            final.append(hyp)

    return final


def pad_sequence(labels: List[int], pad_id: int) -> List[int]:
    """Left pad label ID sequences.

    Args:
        labels: Label ID sequence.
        pad_id: Padding symbol ID.

    Returns:
        final: Padded label ID sequences.

    """
    maxlen = max(len(x) for x in labels)

    final = [([pad_id] * (maxlen - len(x))) + x for x in labels]

    return final


def check_state(
    state: List[Optional[torch.Tensor]], max_len: int, pad_id: int
) -> List[Optional[torch.Tensor]]:
    """Check decoder hidden states and left pad or trim if necessary.

    Args:
        state: Decoder hidden states. [N x (?, D_dec)]
        max_len: maximum sequence length.
        pad_id: Padding symbol ID.

    Returns:
        final: Decoder hidden states. [N x (1, max_len, D_dec)]

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
        final = [state[0].data.new(*final_dims).fill_(pad_id) for _ in range(layers)]

        for i, s in enumerate(state):
            final[i][:, (max_len - s.size(1)) : max_len, :] = s

        return final

    return state


def check_batch_states(states, max_len, pad_id):
    """Check decoder hidden states and left pad or trim if necessary.

    Args:
        state: Decoder hidden states. [N x (B, ?, D_dec)]
        max_len: maximum sequence length.
        pad_id: Padding symbol ID.

    Returns:
        final: Decoder hidden states. [N x (B, max_len, dec_dim)]

    """
    final_dims = (len(states), max_len, states[0].size(1))
    final = states[0].data.new(*final_dims).fill_(pad_id)

    for i, s in enumerate(states):
        curr_len = s.size(0)

        if curr_len < max_len:
            final[i, (max_len - curr_len) : max_len, :] = s
        else:
            final[i, :, :] = s[(curr_len - max_len) :, :]

    return final


def custom_torch_load(model_path: str, model: torch.nn.Module, training: bool = True):
    """Load Transducer model with training-only modules and parameters removed.

    Args:
        model_path: Model path.
        model: Transducer model.

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
        task_keys = ("mlp", "ctc_lin", "kl_div", "lm_lin", "error_calculator")

        model_state_dict = {
            k: v
            for k, v in model_state_dict.items()
            if not any(mod in k for mod in task_keys)
        }

    model.load_state_dict(model_state_dict)

    del model_state_dict
