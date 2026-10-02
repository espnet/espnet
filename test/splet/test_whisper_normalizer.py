#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""The vendored Whisper normalizer against the real openai-whisper.

splet/normalizers/whisper/ is a copy of openai-whisper's normalizers, taken
because importing them would pull in torch. A copy is only worth having if it
is checked, so these tests assert string equality with the installed package
wherever it is available, and pin the behaviour that a reimplementation would
get wrong wherever it is not.
"""

import pathlib
import random

import pytest

from splet.normalizers import build_normalizer
from splet.normalizers.whisper import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
    whisper_setup,
)

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "test_utils" / "splet"

# Chosen to hit each of the four passes and the places they interact: the
# spelling table, the number formatter, contractions, currency and units.
SENTENCES = [
    "Mr. O'Brien paid $1,250.50 for 2 items in 1990s.",
    "I'd've thought he'd be here by now, wouldn't you?",
    "The temperature was -40 degrees and it cost £5.99 per kg.",
    "She finished 3rd out of 21 in the 100m on March 4th, 2019.",
    "Dr. Smith's practice is at 1600 Pennsylvania Ave, Apt. 3B.",
    "It's 25% off, down from seventy-five dollars to $56.25.",
    "we spent 1 1/2 hours on it and then another twenty minutes",
    "[BLANK_AUDIO] the recognised colour was grey, not gray",
    "call me at five five five, one two one two",
    "naïve café résumé façade",
    "one thousand two hundred and thirty four",
    "he said quote unquote it's fine",
    "",
    "   ",
    "a",
]


def corpus_sentences():
    """The parity fixtures, reused so the comparison sees real transcripts."""
    lines = []
    for name in ("ref.scp", "hyp_good.scp", "hyp_poor.scp"):
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) > 1:
                lines.append(fields[1])
    return lines


def random_sentences(count=200):
    """Recombinations of the fixed set, to catch pass-ordering differences."""
    random.seed(20260930)
    pieces = " ".join(SENTENCES).split()
    out = []
    for _ in range(count):
        length = random.randint(1, 12)
        out.append(" ".join(random.choice(pieces) for _ in range(length)))
    return out


@pytest.mark.execution_timeout(120.0)
def test_english_matches_openai_whisper():
    """Every string, exactly, against the package we copied from."""
    upstream = pytest.importorskip("whisper.normalizers")
    theirs = upstream.EnglishTextNormalizer()
    ours = EnglishTextNormalizer()
    for text in SENTENCES + corpus_sentences() + random_sentences():
        assert ours(text) == theirs(text), text


@pytest.mark.execution_timeout(120.0)
@pytest.mark.parametrize("remove_diacritics", [False, True])
def test_basic_matches_openai_whisper(remove_diacritics):
    upstream = pytest.importorskip("whisper.normalizers")
    theirs = upstream.BasicTextNormalizer(remove_diacritics=remove_diacritics)
    ours = BasicTextNormalizer(remove_diacritics=remove_diacritics)
    for text in SENTENCES + corpus_sentences() + random_sentences():
        assert ours(text) == theirs(text), text


def test_inlined_windowed_matches_more_itertools():
    """The one upstream dependency this copy replaced."""
    more_itertools = pytest.importorskip("more_itertools")
    from splet.normalizers.whisper._english import windowed

    for length in range(0, 6):
        words = [str(i) for i in range(length)]
        padded = [None] + words + [None]
        assert list(windowed(padded, 3)) == list(more_itertools.windowed(padded, 3))


def test_the_quirk_that_rules_out_reimplementation():
    """Pinned without needing openai-whisper installed.

    The number pass reads a standalone "o" as a zero, so "O'Brien" becomes
    "0 brien". Nobody would write that on purpose, and a recipe scored
    against real Whisper output depends on it.
    """
    assert EnglishTextNormalizer()("Mr. O'Brien") == "mister 0 brien"
    assert BasicTextNormalizer()("Mr. O'Brien") == "mr o brien"


def test_selected_through_the_normalizer_config():
    """The config a recipe would actually write, and it travels with it."""
    normalizer = build_normalizer([{"name": "whisper"}])
    assert normalizer("It's 25% off.") == EnglishTextNormalizer()("It's 25% off.")
    assert normalizer.config == [{"name": "whisper"}]

    basic = build_normalizer([{"name": "whisper", "language": "whisper_basic"}])
    assert basic("Naïve") == BasicTextNormalizer()("Naïve")


def test_unknown_language_is_refused():
    """Whisper has an English normalizer and a basic one, and no others."""
    with pytest.raises(ValueError, match="unknown whisper normalizer"):
        whisper_setup(language="fr")


def test_english_normalizer_takes_no_options():
    """Silently ignoring them would report a normalization that did not run."""
    with pytest.raises(ValueError, match="takes no options"):
        whisper_setup(language="en", remove_diacritics=True)
