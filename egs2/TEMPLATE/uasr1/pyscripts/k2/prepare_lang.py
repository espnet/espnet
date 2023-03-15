#!/usrilbin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  Johns Hopkins University (author: Dongji Gao)
#
# This script is adapted from \
#    k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py
#
# See https://github.com/k2-fsa/icefall/blob/master/LICENSE
# for clarification regarding multiple authors


"""
This script takes as input a lexicon file "data/lang_phone/lexicon.txt"
consisting of words and tokens (i.e., phones) and does the following:
1. Add disambiguation symbols to the lexicon and generate lexicon_disambig.txt

2. Generate tokens.txt, the token table mapping a token to a unique integer.

   (Dongji: Add option to use existing tokens.txt (--token_list YOUR_TOKEN_LIST))

3. Generate words.txt, the word table mapping a word to a unique integer.

4. Generate L.pt, in k2 format. It can be loaded by
        d = torch.load("L.pt")
        lexicon = k2.Fsa.from_dict(d)

5. Generate L_disambig.pt, in k2 format.
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import k2
import torch
from icefall.lexicon import read_lexicon, write_lexicon

from espnet2.utils.types import str2bool

Lexicon = List[Tuple[str, List[str]]]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang_dir",
        type=str,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        Generated files by this script are saved into this directory.
        """,
    )

    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="""True for debugging, which will generate
        a visualization of the lexicon FST.
        Caution: If your lexicon contains hundreds of thousands
        of lines, please set it to False!
        """,
    )
    parser.add_argument(
        "--token_list",
        type=str,
        default="",
    )
    parser.add_argument(
        "--sil_token",
        type=str,
        default="sil",
    )
    parser.add_argument(
        "--sil_prob",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    """Write a symbol to ID mapping to a file.
    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.
    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")


def get_tokens(lexicon: Lexicon) -> List[str]:
    """Get tokens from a lexicon.
    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
    Returns:
      Return a list of unique tokens.
    """
    ans = set()
    for _, tokens in lexicon:
        ans.update(tokens)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def get_words(lexicon: Lexicon) -> List[str]:
    """Get words from a lexicon.
    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
    Returns:
      Return a list of unique words.
    """
    ans = set()
    for word, _ in lexicon:
        ans.add(word)
    sorted_ans = sorted(list(ans))
    return sorted_ans


def add_disambig_symbols(lexicon: Lexicon) -> Tuple[Lexicon, int]:
    """It adds pseudo-token disambiguation symbols #1, #2 and so on
    at the ends of tokens to ensure that all pronunciations are different,
    and that none is a prefix of another.
    See also add_lex_disambig.pl from kaldi.
    Args:
      lexicon:
        It is returned by :func:`read_lexicon`.
    Returns:
      Return a tuple with two elements:
        - The output lexicon with disambiguation symbols
        - The ID of the max disambiguation symbol that appears
          in the lexicon
    """

    # (1) Work out the count of each token-sequence in the
    # lexicon.
    count = defaultdict(int)
    for _, tokens in lexicon:
        count[" ".join(tokens)] += 1

    # (2) For each left sub-sequence of each token-sequence, note down
    # that it exists (for identifying prefixes of longer strings).
    issubseq = defaultdict(int)
    for _, tokens in lexicon:
        tokens = tokens.copy()
        tokens.pop()
        while tokens:
            issubseq[" ".join(tokens)] = 1
            tokens.pop()

    # (3) For each entry in the lexicon:
    # if the token sequence is unique and is not a
    # prefix of another word, no disambig symbol.
    # Else output #1, or #2, #3, ... if the same token-seq
    # has already been assigned a disambig symbol.
    ans = []

    # We start with #1 since #0 has its own purpose
    first_allowed_disambig = 1
    max_disambig = first_allowed_disambig - 1
    last_used_disambig_symbol_of = defaultdict(int)

    for word, tokens in lexicon:
        tokenseq = " ".join(tokens)
        assert tokenseq != ""
        if issubseq[tokenseq] == 0 and count[tokenseq] == 1:
            ans.append((word, tokens))
            continue

        cur_disambig = last_used_disambig_symbol_of[tokenseq]
        if cur_disambig == 0:
            cur_disambig = first_allowed_disambig
        else:
            cur_disambig += 1

        if cur_disambig > max_disambig:
            max_disambig = cur_disambig
        last_used_disambig_symbol_of[tokenseq] = cur_disambig
        tokenseq += f" #{cur_disambig}"
        ans.append((word, tokenseq.split()))
    return ans, max_disambig


def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Generate ID maps, i.e., map a symbol to a unique ID.
    Args:
      symbols:
        A list of unique symbols.
    Returns:
      A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}


def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.
    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.
    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.
    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_token:
        It is the token ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.
    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst_nosil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
):
    start_state = 0
    next_state = 1
    arcs = []

    assert token2id["<eps>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = start_state

        word = word2id[word]
        tokens = [token2id[i] for i in tokens]

        for i in range(len(tokens) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, tokens[i], w, 0])

            cur_state = next_state
            next_state += 1

        i = len(tokens) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, start_state, tokens[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([start_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arcs: arcs[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def lexicon_to_fst(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    sil_token: str = "SIL",
    sil_prob: float = 0.5,
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format) with optional silence at
    the beginning and end of each word.
    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      sil_token:
        The silence token.
      sil_prob:
        The probability for adding a silence at the beginning and end
        of the word.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    assert sil_prob > 0.0 and sil_prob < 1.0
    # CAUTION: we use score, i.e, negative cost.
    sil_score = math.log(sil_prob)
    no_sil_score = math.log(1.0 - sil_prob)

    start_state = 0
    loop_state = 1  # words enter and leave from here
    sil_state = 2  # words terminate here when followed by silence; this state
    # has a silence transition to loop_state.
    next_state = 3  # the next un-allocated state, will be incremented as we go.
    arcs = []

    assert token2id["<eps>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    sil_token = token2id[sil_token]

    arcs.append([start_state, loop_state, eps, eps, no_sil_score])
    arcs.append([start_state, sil_state, eps, eps, sil_score])
    arcs.append([sil_state, loop_state, sil_token, eps, 0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        tokens = [token2id[i] for i in tokens]

        for i in range(len(tokens) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, tokens[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last token of this word
        # It has two out-going arcs, one to the loop state,
        # the other one to the sil_state.
        i = len(tokens) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, tokens[i], w, no_sil_score])
        arcs.append([cur_state, sil_state, tokens[i], w, sil_score])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lexicon_filename = lang_dir / "lexicon.txt"
    sil_token = args.sil_token
    sil_prob = args.sil_prob

    lexicon = read_lexicon(lexicon_filename)
    if args.token_list is None:
        tokens = get_tokens(lexicon)
    else:
        tokens = []
        with open(args.token_list, "r") as tl:
            for line in tl.readlines():
                token = line.split()[0]
                assert token not in tokens
                tokens.append(token)

    words = get_words(lexicon)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in tokens
        tokens.append(f"#{i}")

    if "<eps>" not in tokens:
        tokens = ["<eps>"] + tokens

    assert "<eps>" not in words
    assert "#0" not in words
    assert "<s>" not in words
    assert "</s>" not in words

    words = ["<eps>", "<unk>"] + words + ["#0", "<s>", "</s>"]

    token2id = generate_id_map(tokens)
    word2id = generate_id_map(words)

    write_mapping(lang_dir / "tokens.txt", token2id)
    write_mapping(lang_dir / "words.txt", word2id)
    write_lexicon(lang_dir / "lexicon_disambig.txt", lexicon_disambig)

    if sil_prob == 0:
        L = lexicon_to_fst_nosil(
            lexicon,
            token2id=token2id,
            word2id=word2id,
        )

        L_disambig = lexicon_to_fst_nosil(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            need_self_loops=True,
        )
    else:
        L = lexicon_to_fst(
            lexicon,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
        )

        L_disambig = lexicon_to_fst(
            lexicon_disambig,
            token2id=token2id,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
            need_self_loops=True,
        )

    torch.save(L.as_dict(), lang_dir / "L.pt")
    torch.save(L_disambig.as_dict(), lang_dir / "L_disambig.pt")

    if args.debug:
        labels_sym = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        aux_labels_sym = k2.SymbolTable.from_file(lang_dir / "words.txt")

        L.labels_sym = labels_sym
        L.aux_labels_sym = aux_labels_sym
        L.draw(f"{lang_dir / 'L.svg'}", title="L.pt")

        L_disambig.labels_sym = labels_sym
        L_disambig.aux_labels_sym = aux_labels_sym
        L_disambig.draw(f"{lang_dir / 'L_disambig.svg'}", title="L_disambig.pt")


if __name__ == "__main__":
    main()
