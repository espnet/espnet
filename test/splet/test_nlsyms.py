#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Non-linguistic symbol removal against espnet2's tokenizers.

`asr.sh` removes nlsyms inside `espnet2.bin.tokenize_text`, and the word and
character tokenizers do it differently: one drops a whole whitespace-delimited
token, the other removes the symbol wherever it appears in the character
stream. Both are reproduced here, and where espnet2 is importable the results
are compared against it directly rather than against a reading of it.

splet/ may not import espnet2 (ci/check_splet_independence.py). A test may:
the rule is about the package, and the whole point of this one is to check
SPLET against the thing it replaces.
"""

import pytest

from splet.normalizers import build_normalizer
from splet.normalizers.basic import remove_tokens_setup

NLSYMS = ["<noise>", "<laugh>", "[APH]"]

TEXTS = [
    "hello <noise> world",
    "<noise>",
    "a <noise> b <laugh> c",
    "nothing to remove here",
    "[APH] the aphasia tag [APH]",
    "<noise><laugh>back to back",
    "",
    "   spaced   out   ",
    "inside<noise>a word",
]


def test_token_mode_drops_whole_tokens():
    remove = remove_tokens_setup(tokens=NLSYMS, match="token")
    assert remove("hello <noise> world") == "hello world"
    # Not a whole token, so WordTokenizer would keep it.
    assert remove("inside<noise>a") == "inside<noise>a"


def test_substring_mode_removes_anywhere():
    remove = remove_tokens_setup(tokens=NLSYMS, match="substring")
    assert remove("inside<noise>a") == "insidea"


def test_substring_mode_keeps_the_spaces_around_a_symbol():
    """The CER denominator counts them.

    CharTokenizer emits <space> for the space before the symbol and again for
    the one after it, so collapsing them here would drop a token from the
    reference length.
    """
    remove = remove_tokens_setup(tokens=NLSYMS, match="substring")
    assert remove("a <noise> b") == "a  b"


def test_longest_symbol_wins():
    """Deterministic where espnet2's set iteration is not."""
    remove = remove_tokens_setup(tokens=["<n>", "<noise>"], match="substring")
    assert remove("x<noise>y") == "xy"


def test_symbols_can_come_from_an_nlsyms_file(tmp_path):
    """The file shape asr.sh passes as --non_linguistic_symbols."""
    path = tmp_path / "nlsyms.txt"
    path.write_text("<noise>\n<laugh>\n\n[APH]\n", encoding="utf-8")
    remove = remove_tokens_setup(tokens_file=str(path), match="token")
    assert remove("a <noise> b [APH] c") == "a b c"


def test_a_blank_line_in_the_file_is_not_an_empty_symbol(tmp_path):
    """espnet2 puts "" in the set, which matches at every position."""
    path = tmp_path / "nlsyms.txt"
    path.write_text("<noise>\n\n\n", encoding="utf-8")
    remove = remove_tokens_setup(tokens_file=str(path), match="substring")
    assert remove("abc<noise>def") == "abcdef"


def test_a_missing_file_is_an_error_not_a_warning():
    """espnet2 warns and scores with an empty set, which is a silent no-op."""
    with pytest.raises(FileNotFoundError):
        remove_tokens_setup(tokens_file="/nonexistent/nlsyms.txt")


def test_unknown_match_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown match"):
        remove_tokens_setup(tokens=NLSYMS, match="fuzzy")


def test_config_travels_with_the_pipeline():
    entry = {"name": "remove_tokens", "tokens": NLSYMS, "match": "substring"}
    normalizer = build_normalizer([entry])
    assert normalizer.config == [entry]


@pytest.mark.parametrize("text", TEXTS)
def test_token_mode_matches_espnet2_word_tokenizer(text):
    """Against the implementation, not against a reading of it."""
    word_tokenizer = pytest.importorskip("espnet2.text.word_tokenizer")
    theirs = word_tokenizer.WordTokenizer(
        non_linguistic_symbols=NLSYMS, remove_non_linguistic_symbols=True
    )
    ours = remove_tokens_setup(tokens=NLSYMS, match="token")
    assert ours(text).split() == theirs.text2tokens(text)


@pytest.mark.parametrize("text", TEXTS)
def test_substring_mode_matches_espnet2_char_tokenizer(text):
    char_tokenizer = pytest.importorskip("espnet2.text.char_tokenizer")
    theirs = char_tokenizer.CharTokenizer(
        non_linguistic_symbols=NLSYMS, remove_non_linguistic_symbols=True
    )
    ours = remove_tokens_setup(tokens=NLSYMS, match="substring")
    mine = ["<space>" if character == " " else character for character in ours(text)]
    assert mine == theirs.text2tokens(text)
