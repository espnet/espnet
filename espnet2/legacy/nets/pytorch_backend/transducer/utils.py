"""Utility functions for Transducer models."""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from espnet2.legacy.nets.pytorch_backend.nets_utils import pad_list
from espnet2.legacy.nets.transducer_decoder_interface import (
    ExtendedHypothesis,
    Hypothesis,
)



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
    topk_idxs: torch.Tensor,
    topk_logps: torch.Tensor,
    gamma: float,
) -> List[ExtendedHypothesis]:
    """Return K hypotheses candidates for expansion from a list of hypothesis.

    K candidates are selected according to the extended hypotheses probabilities
    and a prune-by-value method. Where K is equal to beam_size + beta.

    Args:
        hyps: Hypotheses.
        topk_idxs: Indices of candidates hypothesis.
        topk_logps: Log-probabilities for hypotheses expansions.
        gamma: Allowed logp difference for prune-by-value method.

    Return:
        k_expansions: Best K expansion hypotheses candidates.

    """
    k_expansions = []

    for i, hyp in enumerate(hyps):
        hyp_i = [
            (int(k), hyp.score + float(v)) for k, v in zip(topk_idxs[i], topk_logps[i])
        ]
        k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

        k_expansions.append(
            sorted(
                filter(lambda x: (k_best_exp - gamma) <= x[1], hyp_i),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    return k_expansions



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

