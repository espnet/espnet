#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
#                2021  Gaopeng Xu
# Apache 2.0

import k2
import logging
import os
import torch
from k2 import Fsa, SymbolTable
from pathlib import Path
from typing import Union
from typing import Dict, Optional, Tuple, List
import re
def find_first_disambig_symbol(symbols: k2.SymbolTable) -> int:
    return min(v for k, v in symbols._sym2id.items() if k.startswith('#'))

def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    '''Build CTC topology.
    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.
    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.
    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    '''
    assert 0 in tokens, 'We assume 0 is ID of the blank symbol'

    num_states = len(tokens)
    final_state = num_states
    arcs = ''
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f'{i} {i} {tokens[i]} 0 0.0\n'
            else:
                arcs += f'{i} {j} {tokens[j]} {tokens[j]} 0.0\n'
        arcs += f'{i} {final_state} -1 -1 0.0\n'
    arcs += f'{final_state}'
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans



def compile_HLG(
        L: Fsa,
        G: Fsa,
        H: Fsa,
        labels_disambig_id_start: int,
        aux_labels_disambig_id_start: int
) -> Fsa:
    """
    Creates a decoding graph using a lexicon fst ``L`` and language model fsa ``G``.
    Involves arc sorting, intersection, determinization, removal of disambiguation symbols
    and adding epsilon self-loops.

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
    logging.info("Intersecting L and G")
    LG = k2.compose(L, G)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting L*G")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Determinizing L*G")
    LG = k2.determinize(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting det(L*G)")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Removing disambiguation symbols on L*G")
    LG.labels[LG.labels >= labels_disambig_id_start] = 0
    if isinstance(LG.aux_labels, torch.Tensor):
        LG.aux_labels[LG.aux_labels >= aux_labels_disambig_id_start] = 0
    else:
        LG.aux_labels.values()[LG.aux_labels.values() >= aux_labels_disambig_id_start] = 0
    logging.info("Removing epsilons")
    LG = k2.remove_epsilon(LG)
    logging.info(f'LG shape = {LG.shape}')
    logging.info("Connecting rm-eps(det(L*G))")
    LG = k2.connect(LG)
    logging.info(f'LG shape = {LG.shape}')
    LG.aux_labels = k2.ragged.remove_values_eq(LG.aux_labels, 0)

    logging.info("Arc sorting LG")
    LG = k2.arc_sort(LG)

    logging.info("Composing ctc_topo LG")
    HLG = k2.compose(H, LG, inner_labels='phones')

    logging.info("Connecting LG")
    HLG = k2.connect(HLG)

    logging.info("Arc sorting LG")
    HLG = k2.arc_sort(HLG)
    logging.info(
        f'LG is arc sorted: {(HLG.properties & k2.fsa_properties.ARC_SORTED) != 0}'
    )
    return HLG


def get_phone_symbols(symbol_table: k2.SymbolTable,
                      pattern: str = r'^#\d+$') -> List[int]:
    '''Return a list of phone IDs containing no disambiguation symbols.

    Caution:
      0 is not a phone ID so it is excluded from the return value.

    Args:
      symbol_table:
        A symbol table in k2.
      pattern:
        Symbols containing this pattern are disambiguation symbols.
    Returns:
      Return a list of symbol IDs excluding those from disambiguation symbols.
    '''
    regex = re.compile(pattern)
    symbols = symbol_table.symbols
    ans = []
    for s in symbols:
        if not regex.match(s):
            ans.append(symbol_table[s])
    if 0 in ans:
        ans.remove(0)
    ans.sort()
    return ans


lang_dir = Path('data/fst/lang')
symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'tokens.txt')
phone_ids = get_phone_symbols(phone_symbol_table)
phone_ids_with_blank = [0] + phone_ids
ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
if not os.path.exists(lang_dir / 'HLG.pt'):
    print("Loading L.fst.txt")
    with open(lang_dir / 'L_disambig.fst.txt') as f:
        L = k2.Fsa.from_openfst(f.read(), acceptor=False)
    with open(lang_dir / 'G.fst.txt') as f:
        G = k2.Fsa.from_openfst(f.read(), acceptor=False)
    first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
    first_word_disambig_id = find_first_disambig_symbol(symbol_table)
    HLG = compile_HLG(L=L,
                     G=G,
                     H=ctc_topo,
                     labels_disambig_id_start=first_phone_disambig_id,
                     aux_labels_disambig_id_start=first_word_disambig_id)
    torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')

