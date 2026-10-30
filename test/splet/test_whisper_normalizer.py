"""Whisper-style normalization, and the claim that it is OpenAI's.

The point of depending on whisper-normalizer rather than vendoring is that
it is a redistribution of the normalizers in openai-whisper. That is a
claim about two packages, so it is tested as one: where both are installed,
every string must normalize identically. Where neither or only one is, the
rest of the file still runs.
"""

import pathlib

import pytest

from splet.normalizers import build_normalizer
from splet.normalizers.whisper import VARIANT_CHOICES, whisper_setup

whisper_normalizer = pytest.importorskip(
    "whisper_normalizer", reason="in the splet extra, which CI does not install"
)

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "test_utils" / "splet"

# Built to reach each stage of EnglishTextNormalizer: the spelling table, the
# number formatter, the contraction expander and the currency/unit handler.
PROBES = [
    "I paid $20.50 for 3 items",
    "it's 1980's style",
    "Dr. Smith went to St. Louis",
    "we don't wanna go",
    "the temperature is -5 degrees",
    "Mr. O'Brien's car",
    "100%",
    "1/2 cup",
    "a.m. and p.m.",
    "COVID-19",
    "twenty-first century",
    "£5 and €10",
    "No. 7",
    "Hello, World!",
    "THE QUICK BROWN FOX",
    "   leading and trailing   ",
    "",
    "1,234,567",
    'she said "hello"',
    "ain't gonna",
    "1st 2nd 3rd",
    "你好 world",
]


def _fixture_strings():
    strings = []
    for name in ("ref.scp", "hyp_good.scp", "hyp_poor.scp", "ref_cs.scp"):
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                strings.append(parts[1])
    return strings


@pytest.mark.parametrize("variant", sorted(VARIANT_CHOICES))
def test_reproduces_openai_whisper(variant):
    """The dependency is a redistribution; prove it on this machine."""
    # whisper.normalizers is a submodule, not an attribute of the package:
    # importing "whisper" alone leaves it unbound.
    pytest.importorskip("whisper.normalizers")
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

    reference = EnglishTextNormalizer() if variant == "en" else BasicTextNormalizer()
    ours = whisper_setup(variant=variant)
    differences = [
        text for text in PROBES + _fixture_strings() if ours(text) != reference(text)
    ]
    assert differences == [], f"{variant}: {differences[:3]}"


def test_the_two_variants_are_the_two_espnet2_cleaners():
    """espnet2.text.cleaner builds exactly these, with default arguments."""
    assert VARIANT_CHOICES == {"en": "whisper_en", "basic": "whisper_basic"}


def test_english_normalization_runs_from_a_config():
    normalize = build_normalizer([{"name": "whisper", "variant": "en"}])
    assert normalize("Dr. Smith!") == "doctor smith"


def test_basic_normalization_keeps_non_english():
    normalize = build_normalizer([{"name": "whisper", "variant": "basic"}])
    assert normalize("Hello, World! 你好") == "hello world 你好"


def test_split_letters_belongs_to_the_basic_variant():
    spaced = whisper_setup(variant="basic", split_letters=True)("abc")
    assert spaced.split() == ["a", "b", "c"]
    with pytest.raises(ValueError, match="variant='basic' only"):
        whisper_setup(variant="en", split_letters=True)


def test_an_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown whisper variant"):
        whisper_setup(variant="whisper_en")


def test_the_config_is_echoed_back():
    """A number is not reproducible without the normalization that made it."""
    normalize = build_normalizer([{"name": "whisper", "variant": "basic"}])
    assert normalize.config == [{"name": "whisper", "variant": "basic"}]
