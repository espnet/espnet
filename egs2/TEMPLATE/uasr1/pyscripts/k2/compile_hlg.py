#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  Johns Hopkins University (author: Dongji Gao)
#
# This script is adapted from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compile_hlg.py
#
# See https://github.com/k2-fsa/icefall/blob/master/LICENSE
# for clarification regarding multiple authors


"""
This script takes as input lang_dir and generates HLG from

    - H, the ctc topology, built from tokens contained in lang_dir/lexicon.txt

      (Dongji: Add a H topology with no blank symbol for UASR decoding)

    - L, the lexicon, built from lang_dir/L_disambig.pt

        Caution: We use a lexicon that contains disambiguation symbols

    - G, the LM, built from data/lm/G_3_gram.fst.txt

The generated HLG is saved in $lang_dir/HLG.pt
"""
import argparse
import logging
from pathlib import Path

import k2
import torch
from icefall.lexicon import Lexicon


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_dir",
        type=str,
        help="""Lang directory.
        """,
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        help="""Graph directory.
        """,
    )
    parser.add_argument(
        "--ngram_num",
        type=int,
        default=4,
        help="""Max order in language model.
        """,
    )

    return parser.parse_args()


def make_h_no_blank(max_token_id, self_loop_penalty=-3):
    num_states = max_token_id + 1
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(1, num_states):
            if i == j:
                arcs += f"{i} {i} {i} 0 {self_loop_penalty}\n"
            else:
                arcs += f"{i} {j} {j} {j} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans


def compile_HLG(lang_dir: str, graph_dir: str, ngram_num: int) -> k2.Fsa:
    """
    Args:
      lang_dir:
        The language directory contains lexicon fst
      graph_dir:
        The graph and output directory contains grammar fst
      ngram_num:
        Max order of n-gram language model

    Return:
      An FSA representing HLG.
    """
    lexicon = Lexicon(lang_dir)
    max_token_id = max(lexicon.tokens)
    logging.info(f"Building ctc_topo. max_token_id: {max_token_id}")
    H = make_h_no_blank(max_token_id)
    L = k2.Fsa.from_dict(torch.load(f"{lang_dir}/L_disambig.pt"))

    logging.info(f"Loading G_{ngram_num}_gram.fst.txt")
    with open(f"{graph_dir}/G_{ngram_num}_gram.fst.txt") as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        torch.save(G.as_dict(), f"{graph_dir}/G_{ngram_num}_gram.pt")

    first_token_disambig_id = lexicon.token_table["#0"]
    first_word_disambig_id = lexicon.word_table["#0"]

    L = k2.arc_sort(L)
    G = k2.arc_sort(G)

    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f"LG shape: {LG.shape}")

    logging.info("Connecting LG")
    LG = k2.connect(LG)
    logging.info(f"LG shape after k2.connect: {LG.shape}")

    logging.info(type(LG.aux_labels))
    logging.info("Determinizing LG")

    LG = k2.determinize(LG)
    logging.info(type(LG.aux_labels))

    logging.info("Connecting LG after k2.determinize")
    LG = k2.connect(LG)

    logging.info("Removing disambiguation symbols on LG")

    LG.labels[LG.labels >= first_token_disambig_id] = 0
    # See https://github.com/k2-fsa/k2/issues/874
    # for why we need to set LG.properties to None
    LG.__dict__["_properties"] = None

    assert isinstance(LG.aux_labels, k2.RaggedTensor)
    LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

    LG = k2.remove_epsilon(LG)
    logging.info(f"LG shape after k2.remove_epsilon: {LG.shape}")

    LG = k2.connect(LG)
    LG.aux_labels = LG.aux_labels.remove_values_eq(0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing H and LG")
    # CAUTION: The name of the inner_labels is fixed
    # to `tokens`. If you want to change it, please
    # also change other places in icefall that are using
    # it.
    HLG = k2.compose(H, LG, inner_labels="tokens")

    logging.info("Connecting HLG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting HLG")
    HLG = k2.arc_sort(HLG)
    logging.info(f"HLG.shape: {HLG.shape}")

    return HLG


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    graph_dir = Path(args.graph_dir)

    logging.info(f"Processing {lang_dir} and {graph_dir}")

    HLG = compile_HLG(lang_dir, graph_dir, args.ngram_num)
    logging.info(f"Saving HLG.pt to {graph_dir}")
    torch.save(HLG.as_dict(), f"{graph_dir}/HLG.pt")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
