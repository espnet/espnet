"""SPLET's normalize-then-tokenize pipeline against espnet2's tokenizers.

``egs2/TEMPLATE/asr1/asr.sh`` scores by piping both sides through
``espnet2.bin.tokenize_text`` and handing the result to sclite. SPLET
replaces that pipe, so what it produces has to be what the pipe produced,
character for character -- a denominator that differs by one token moves
every rate computed from it.

The two tokenizers remove non-speech markers differently, which is why
:func:`splet.normalizers.basic.remove_tokens_setup` has a ``match`` mode at
all. These tests pin the pairing: ``match="token"`` with word tokenization,
``match="substring"`` with char.
"""

import pytest

from splet.normalizers.basic import remove_tokens_setup
from splet.utterance_metrics.error_rate import TOKENIZER_CHOICES

pytest.importorskip(
    "espnet2", reason="splet/ is extractable; these tests are not part of it"
)

from espnet2.text.char_tokenizer import CharTokenizer  # noqa: E402
from espnet2.text.word_tokenizer import WordTokenizer  # noqa: E402

SYMBOLS = ["<noise>", "<unk>", "[laughter]"]

TEXTS = [
    "a <noise> b",
    "hello <unk> world",
    "<noise>start",
    "end<noise>",
    "a <noise><unk> b",
    "[laughter] ha",
    "no markers here",
    # Nothing is removed here, so CharTokenizer leaves the doubled space
    # alone: _merge_spaces runs under `if removed`.
    "double  space kept",
    "<noise> only",
    "a <noise> ",
    "  <unk>  x  ",
    # espnet/espnet#6863: a marker strictly inside a word leaves no space
    # behind at all, which is a different path through _merge_spaces than
    # a marker surrounded by them.
    "hel<noise>lo world",
]


@pytest.mark.parametrize("text", TEXTS)
def test_char_pipeline_matches_char_tokenizer(text):
    """match="substring" + char == CharTokenizer(remove_non_linguistic...)."""
    expected = CharTokenizer(
        non_linguistic_symbols=SYMBOLS, remove_non_linguistic_symbols=True
    ).text2tokens(text)
    normalize = remove_tokens_setup(SYMBOLS, match="substring")
    tokenize = TOKENIZER_CHOICES["char"](space_symbol="<space>")
    assert tokenize(normalize(text)) == expected


@pytest.mark.parametrize("text", TEXTS)
def test_word_pipeline_matches_word_tokenizer(text):
    """match="token" + word == WordTokenizer(remove_non_linguistic...)."""
    expected = WordTokenizer(
        non_linguistic_symbols=SYMBOLS, remove_non_linguistic_symbols=True
    ).text2tokens(text)
    normalize = remove_tokens_setup(SYMBOLS, match="token")
    tokenize = TOKENIZER_CHOICES["word"]()
    assert tokenize(normalize(text)) == expected


def test_the_two_match_modes_are_not_interchangeable():
    """Pin the disagreement, so the pairing above is not decoration.

    A marker glued to a word is removed by one and kept by the other, which
    is a whole token of denominator either way.
    """
    token = remove_tokens_setup(SYMBOLS, match="token")
    substring = remove_tokens_setup(SYMBOLS, match="substring")
    assert token("end<noise>") == "end<noise>"
    assert substring("end<noise>") == "end"


def test_substring_mode_tidies_spaces_only_when_it_removed_something():
    """CharTokenizer._merge_spaces runs under `if removed`; so does this.

    Tidying unconditionally would quietly change the denominator of every
    utterance that has a doubled space and no marker.
    """
    substring = remove_tokens_setup(SYMBOLS, match="substring")
    assert substring("double  space kept") == "double  space kept"
    assert substring("a <noise>  b") == "a b"


def test_markers_come_from_a_file_the_way_espnet_reads_one(tmp_path):
    """nlsyms_txt: one marker per line, trailing whitespace stripped."""
    path = tmp_path / "nlsyms.txt"
    path.write_text("<noise>\n<unk>  \n\n", encoding="utf-8")
    normalize = remove_tokens_setup(tokens_file=str(path), match="token")
    assert normalize("a <noise> b <unk> c") == "a b c"
    # A blank line would otherwise contribute an empty marker, which matches
    # at every position.
    assert normalize("abc") == "abc"


def test_a_file_and_an_explicit_list_combine(tmp_path):
    path = tmp_path / "nlsyms.txt"
    path.write_text("<noise>\n", encoding="utf-8")
    normalize = remove_tokens_setup(
        tokens=["[laughter]"], tokens_file=str(path), match="token"
    )
    assert normalize("a <noise> [laughter] b") == "a b"


def test_an_unknown_match_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown match"):
        remove_tokens_setup(SYMBOLS, match="regex")


def test_removing_punctuation_before_markers_destroys_them():
    """The ordering hazard, pinned so a config author meets it in a test.

    remove_punctuation strips the angle brackets, after which <noise> is the
    ordinary word "noise" and no marker list will match it.
    """
    from splet.normalizers import build_normalizer

    wrong = build_normalizer(
        [{"name": "remove_punctuation"}, {"name": "remove_tokens"}]
    )
    assert "noise" in wrong("a <noise> b")

    right = build_normalizer(
        [{"name": "remove_tokens"}, {"name": "remove_punctuation"}]
    )
    assert "noise" not in right("a <noise> b")


def test_satisfies_the_assertions_of_espnet_6863():
    """The upstream fix this reproduces, checked against its own test.

    espnet/espnet#6863 made CharTokenizer merge the spaces a removed marker
    leaves behind, because the character error rate was charging one error
    per tag in the reference while the word error rate of the same pair was
    zero. SPLET has to agree with the fixed tokenizer, not the old one, so
    its four assertions are pinned here as well: the same expectation that
    it checks, reached through the SPLET pipeline.
    """
    expected = CharTokenizer(
        non_linguistic_symbols=["<noise>"], remove_non_linguistic_symbols=True
    ).text2tokens("hello world")
    normalize = remove_tokens_setup(["<noise>"], match="substring")
    tokenize = TOKENIZER_CHOICES["char"](space_symbol="<space>")

    for text in (
        "hello <noise> world",
        "<noise> hello world <noise>",
        "hello <noise><noise> world",
        "hel<noise>lo world",
    ):
        assert tokenize(normalize(text)) == expected, text
