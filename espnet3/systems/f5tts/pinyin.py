# MIT License
#
# Copyright (c) 2024 Yushen CHEN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""F5-TTS zh+en pinyin tokenization.

Adapted from F5-TTS's ``convert_char_to_pinyin``
(https://github.com/SWivid/F5-TTS, ``src/f5_tts/model/utils.py``),
which is MIT-licensed; the notice above is that project's own. ``is_chinese`` is
carried over unchanged, while ``convert_char_to_pinyin`` was reworked for this
recipe, so this is an adaptation rather than a copy. Behaviour: Chinese
characters become pinyin syllables (``Style.TONE3`` + tone sandhi, segmented with
``rjieba``), while English / letters / symbols are kept and split into individual
characters. This is exactly F5's "Emilia_ZH_EN_pinyin" scheme. Note it is *not*
"pinyin for English"; English stays char-level.

Provides three ways to use it:
  * ``f5_pinyin_g2p`` + ``register_f5_pinyin_g2p``: expose it as an ESPnet g2p
    (``g2p_type="f5_pinyin"``) so it works inside ``CommonPreprocessor`` /
    ``build_tokenizer`` when you build your own vocab (with a ``<unk>``).
  * ``convert_char_to_pinyin`` / ``text_to_pinyin_ids``: the raw tokenizer and
    F5's exact id mapping (unknown token -> 0), used by ``F5PinyinPreprocessor``
    to match F5's fixed ``vocab.txt`` (which has no ``<unk>``).

``rjieba`` and ``pypinyin`` are imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def convert_char_to_pinyin(
    text_list: List[str], polyphone: bool = True
) -> List[List[str]]:
    """F5-TTS tokenizer: hanzi -> pinyin syllables, else char-level.

    Args:
        text_list: Raw transcripts. Each entry is tokenized independently.
        polyphone: Enable pypinyin polyphone handling and tone sandhi for pure
            east-asian segments.

    Returns:
        One token list per input string, in the same order.

    Example:
        .. code-block:: python

            >>> convert_char_to_pinyin(["hello"])
            [['h', 'e', 'l', 'l', 'o']]
            >>> convert_char_to_pinyin(["\u4e2d\u6587"])
            [[' ', 'zhong1', ' ', 'wen2']]

    Note:
        This is F5's "Emilia_ZH_EN_pinyin" scheme, not "pinyin for English":
        Chinese becomes ``Style.TONE3`` syllables each preceded by a space,
        while English stays character-level. ``rjieba`` runs on every input,
        English included, so both optional dependencies are needed even for
        English-only text. Output depends on the installed ``rjieba`` /
        ``pypinyin`` versions.
    """
    import rjieba
    from pypinyin import Style, lazy_pinyin

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"  # common chinese characters

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in rjieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(
                seg
            ):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(
                            lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        )
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


def f5_pinyin_g2p(text: str) -> List[str]:
    """ESPnet-style g2p wrapper: a single string -> list of F5 pinyin tokens.

    Args:
        text: One raw transcript.

    Returns:
        The token list for ``text``.

    Example:
        .. code-block:: python

            >>> f5_pinyin_g2p("hi")
            ['h', 'i']

    Note:
        Exists because ESPnet's g2p contract is one string in, one token list
        out, whereas :func:`convert_char_to_pinyin` is batch-shaped. Registered
        as ``g2p_type="f5_pinyin"`` by :func:`register_f5_pinyin_g2p`.
    """
    return convert_char_to_pinyin([text])[0]


def load_vocab_char_map(vocab_file: str) -> Dict[str, int]:
    """Load F5's ``vocab.txt`` as ``{token: index}`` (line number = index).

    Args:
        vocab_file: Path to F5's ``vocab.txt``, one token per line.

    Returns:
        Mapping from token to its line number.

    Example:
        .. code-block:: python

            >>> load_vocab_char_map("vocab.txt")   # first line is a space
            {' ': 0, 'a': 1, 'b': 2}

    Note:
        Only the trailing newline is stripped, never surrounding whitespace, so
        F5's literal space token at index 0 survives. Stripping it would shift
        every id and silently corrupt the mapping. Unlike an ESPnet token list
        this vocabulary has no ``<blank>`` / ``<unk>`` / ``<sos/eos>``.
    """
    vocab_char_map: Dict[str, int] = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab_char_map[line.rstrip("\n")] = i  # keep a literal space token
    return vocab_char_map


def text_to_pinyin_ids(text: str, vocab_char_map: Dict[str, int]) -> np.ndarray:
    """F5's exact mapping: tokenize then map each token, unknown -> 0.

    Args:
        text: One raw transcript.
        vocab_char_map: Mapping from :func:`load_vocab_char_map`.

    Returns:
        Token ids as an ``int64`` array.

    Example:
        .. code-block:: python

            >>> text_to_pinyin_ids("abz", {" ": 0, "a": 1, "b": 2})
            array([1, 2, 0])

    Note:
        F5's vocabulary has no ``<unk>``, so an unknown token maps to index 0,
        which is the literal space. That is upstream's behaviour and is kept
        deliberately for checkpoint compatibility, but it means out-of-vocab
        text degrades silently rather than raising.
    """
    tokens = convert_char_to_pinyin([text])[0]
    return np.asarray([vocab_char_map.get(t, 0) for t in tokens], dtype=np.int64)


def build_pinyin_vocab(
    texts: Iterable[str],
    add_ascii_latin: bool = False,
    polyphone: bool = True,
) -> List[str]:
    """Build an F5-style pinyin ``vocab.txt`` token list from raw transcripts.

    Reproduces the vocab construction in F5-TTS's dataset prep
    (``prepare_emilia.py`` / ``prepare_csv_wavs.py``): tokenize every transcript
    with :func:`convert_char_to_pinyin`, take the **union of unique tokens**, and
    return them **sorted** (Unicode codepoint order). No ``<blank>``/``<unk>``/
    ``<sos/eos>`` symbols are added, and a literal space lands at index 0 (it is
    the lowest-codepoint token), matching F5's convention (``vocab[" "] == 0``).

    ``texts`` is any iterable of strings, so this is dataset-agnostic. Feed it
    transcripts from any manifest or dataset.

    Args:
        texts: Iterable of raw transcripts (zh/en/mixed).
        add_ascii_latin: If True, also seed the vocab with ASCII (32..126) and
            Latin-1 (192..255) characters, F5's optional line for broadening
            coverage (e.g. finetuning to de/fr). Off by default to match the
            plain Emilia/csv prep.
        polyphone: Passed through to :func:`convert_char_to_pinyin`.

    Returns:
        Sorted list of vocab tokens (write one per line to ``vocab.txt``).

    Note:
        The exact token set depends on the corpus *and* on the ``rjieba`` /
        ``pypinyin`` versions (polyphone handling), so this will not necessarily
        be byte-identical to F5's published ``Emilia_ZH_EN_pinyin/vocab.txt``.
        For from-scratch training that is fine; for finetuning a pretrained F5
        model, use F5's shipped ``vocab.txt`` instead.

    Example:
        .. code-block:: python

            >>> build_pinyin_vocab(["ab"])
            ['a', 'b']
    """
    vocab_set = set()
    for text in texts:
        tokens = convert_char_to_pinyin([text], polyphone=polyphone)[0]
        vocab_set.update(tokens)
    if add_ascii_latin:
        vocab_set.update(
            [chr(i) for i in range(32, 127)] + [chr(i) for i in range(192, 256)]
        )
    return sorted(vocab_set)


_REGISTERED = False


def register_f5_pinyin_g2p() -> None:
    """Register ``g2p_type="f5_pinyin"`` into espnet2's PhonemeTokenizer.

    Lets ``CommonPreprocessor`` / ``build_tokenizer`` use F5's pinyin tokenizer
    via ``token_type: phn, g2p_type: f5_pinyin`` (for the from-scratch case where
    you build your own vocab including a ``<unk>`` symbol). Idempotent.

    Returns:
        ``None``. The effect is the global registration.

    Example:
        .. code-block:: yaml

            preprocessor:
              _target_: espnet2.train.preprocessor.CommonPreprocessor
              token_type: phn
              g2p_type: f5_pinyin

    Note:
        This monkey-patches ``espnet2.text.phoneme_tokenizer``, appending to
        ``g2p_choices`` and wrapping ``PhonemeTokenizer.__init__``. Repeat calls
        are no-ops, so it is safe to call from several entry points. Use this
        path only when building your own vocab with a ``<unk>``; to match F5's
        fixed ``vocab.txt`` use ``F5PinyinPreprocessor`` instead, since ESPnet's
        ``TokenIDConverter`` requires a ``<unk>`` that F5's vocab lacks.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import espnet2.text.phoneme_tokenizer as pt

    if "f5_pinyin" not in pt.g2p_choices:
        pt.g2p_choices.append("f5_pinyin")

    orig_init = pt.PhonemeTokenizer.__init__
    if getattr(orig_init, "_f5_patched", False):
        _REGISTERED = True
        return

    def __init__(
        self,
        g2p_type,
        non_linguistic_symbols=None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        if g2p_type == "f5_pinyin":
            # Reuse the original setup (g2p_type=None path sets all common attrs),
            # then swap in F5's pinyin g2p.
            orig_init(
                self,
                None,
                non_linguistic_symbols,
                space_symbol,
                remove_non_linguistic_symbols,
            )
            self.g2p = f5_pinyin_g2p
            self.g2p_type = "f5_pinyin"
        else:
            orig_init(
                self,
                g2p_type,
                non_linguistic_symbols,
                space_symbol,
                remove_non_linguistic_symbols,
            )

    __init__._f5_patched = True
    pt.PhonemeTokenizer.__init__ = __init__
    _REGISTERED = True
