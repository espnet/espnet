"""Tests for the st.sh case conventions and the Moses/espnet Perl ports."""

import pytest

from espnet3.systems.st.normalization import (
    CASES,
    apply_case,
    is_punctuation,
    normalize_punctuation,
    remove_punctuation,
)


@pytest.mark.parametrize(
    "char, expected",
    [
        (".", True),
        (",", True),
        ("'", True),
        ("-", True),
        ("¿", True),
        ("„", True),
        # Unicode calls these Symbol, not Punctuation, but Perl's [[:punct:]]
        # matches them, so the port has to as well.
        ("$", True),
        ("+", True),
        ("<", True),
        ("=", True),
        ("^", True),
        ("`", True),
        ("|", True),
        ("~", True),
        ("a", False),
        ("7", False),
        (" ", False),
        ("ü", False),
    ],
)
def test_is_punctuation_matches_perl_posix_class(char, expected):
    assert is_punctuation(char) is expected


def test_remove_punctuation_keeps_apostrophes():
    # The Perl script protects apostrophes with a placeholder; everything else
    # in [[:punct:]] goes.
    assert remove_punctuation("don't stop, please!") == "don't stop please"


def test_remove_punctuation_protects_space_marker():
    # "<space>" survives even though <, > and the word are punctuation-adjacent.
    assert remove_punctuation("a <space> b.") == "a <space> b"


def test_remove_punctuation_collapses_whitespace():
    assert remove_punctuation("  a  --  b  ") == "a b"


def test_remove_punctuation_placeholder_collision_matches_perl():
    """The placeholder trick rewrites literal "apostrophe"/"spacemark".

    remove_punctuation.pl swaps those two words in and back out to protect
    apostrophes and the <space> marker, so input that already contains them
    comes out changed. The Perl has the identical flaw; this is asserted so
    the quirk is a recorded property of the port rather than a surprise.
    """
    assert remove_punctuation("apostrophe and spacemark") == "' and <space>"


def test_remove_punctuation_on_empty_string():
    assert remove_punctuation("") == ""


@pytest.mark.parametrize(
    "case, expected",
    [
        ("tc", "Hello, World!"),
        ("lc", "hello, world!"),
        ("lc.rm", "hello world"),
    ],
)
def test_apply_case(case, expected):
    assert apply_case("Hello, World!", case) == expected


def test_apply_case_covers_every_declared_case():
    for case in CASES:
        apply_case("Some Text.", case)


def test_apply_case_rejects_unknown_case():
    with pytest.raises(ValueError, match="Unknown case"):
        apply_case("text", "uc")


def test_normalize_punctuation_unifies_quotes_and_dashes():
    out = normalize_punctuation("„quoted“ and –dashed–")
    assert "„" not in out and "“" not in out
    assert out.count('"') == 2
    assert "–" not in out


def test_normalize_punctuation_ellipsis_and_backtick():
    assert normalize_punctuation("wait… now") == "wait... now"
    assert normalize_punctuation("`quoted'") == "'quoted'"


def test_normalize_punctuation_handles_nbsp_pseudo_spaces():
    # Moses writes U+00A0 on the left of these rules, not an ASCII space.
    assert normalize_punctuation("50 %") == "50%"
    assert normalize_punctuation("really ?") == "really?"
    assert normalize_punctuation("20 cm") == "20 cm"


def test_normalize_punctuation_french_guillemets():
    assert '"' in normalize_punctuation("« bonjour »")


def test_normalize_punctuation_digit_separator_is_language_specific():
    # German groups digits with a comma, English with a period.
    assert normalize_punctuation("1 000", language="de") == "1,000"
    assert normalize_punctuation("1 000", language="en") == "1.000"


def test_normalize_punctuation_english_moves_quote_after_punctuation():
    # The en rule rewrites '"' followed by commas/periods as the punctuation
    # first, then the quote.
    assert normalize_punctuation('he said ","', language="en") == 'he said ,""'


def test_normalize_punctuation_strips_carriage_returns():
    assert "\r" not in normalize_punctuation("line\r")


def test_normalize_punctuation_is_idempotent_on_plain_text():
    text = "A plain sentence with nothing unusual."
    assert normalize_punctuation(text) == text
