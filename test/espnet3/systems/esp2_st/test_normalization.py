"""Tests for the st.sh case conventions and the remove_punctuation port."""

import pytest

from espnet3.systems.esp2_st.src.normalization import (
    CASES,
    apply_case,
    is_punctuation,
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
