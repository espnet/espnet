import logging

import k2
import torch

from k2 import Fsa


# Copied from: shorturl.at/opuxR
def compile_HLG(
    L: Fsa,
    G: Fsa,
    H: Fsa,
    labels_disambig_id_start: int,
    aux_labels_disambig_id_start: int,
) -> Fsa:
    """Creates a decoding graph using a lexicon fst ``L`` and language model fsa ``G``.

    Involves arc sorting, intersection, determinization,
    removal of disambiguation symbols and adding epsilon self-loops.

    Args:
        L:
            An ``Fsa`` that represents the lexicon (L), i.e. has phones as ``symbols``
                and words as ``aux_symbols``.
        G:
            An ``Fsa`` that represents the language model (G), i.e. it's an acceptor
            with words as ``symbols``.
        H:  An ``Fsa`` that represents a specific topology used to convert the network
            outputs to a sequence of phones.
            Typically, it's a CTC topology fst, in which when 0 appears on the left
            side, it represents the blank symbol; when it appears on the right side,
            it indicates an epsilon.
        labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            phonetic alphabet.
        aux_labels_disambig_id_start:
            An integer ID corresponding to the first disambiguation symbol in the
            words vocabulary.
    :return:
    """
    L = k2.arc_sort(L)
    G = k2.arc_sort(G)
    # Attach a new attribute `lm_scores` so that we can recover
    # the `am_scores` later.
    # The scores on an arc consists of two parts:
    #  scores = am_scores + lm_scores
    # NOTE: we assume that both kinds of scores are in log-space.
    G.lm_scores = G.scores.clone()

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting L*G")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[
            LG.aux_labels.values() >= aux_labels_disambig_id_start
        ] = 0
    logging.info("Removing epsilons")
    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape = {LG.shape}")
    logging.info("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.info(f"LG shape = {LG.shape}")
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing ctc_topo LG")
    HLG = k2.compose(H, LG, inner_labels="phones")

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(
        f"LG is arc sorted: {(HLG.properties & k2.fsa_properties.ARC_SORTED) != 0}"
    )

    return HLG


# Copied from: https://shorturl.at/diuCN
def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith("#"))
