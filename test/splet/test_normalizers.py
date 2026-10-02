"""Tests for the normalization pipeline."""

import pytest

from splet.normalizers import build_normalizer


def test_empty_config_is_the_identity():
    """SPLET never normalizes text the caller did not ask it to normalize."""
    normalizer = build_normalizer(None)
    assert normalizer("  Hello, World!  ") == "  Hello, World!  "
    assert len(normalizer) == 0


def test_steps_apply_in_order():
    normalizer = build_normalizer(
        [
            {"name": "lowercase"},
            {"name": "remove_punctuation", "keep": "'"},
            {"name": "whitespace"},
        ]
    )
    assert normalizer("  Don't  STOP, please!  ") == "don't stop please"


def test_punctuation_removal_covers_non_ascii():
    """Matching by Unicode category is what makes this work outside English."""
    normalizer = build_normalizer([{"name": "remove_punctuation"}])
    assert normalizer("こんにちは、世界！") == "こんにちは世界"


def test_punctuation_can_be_replaced_rather_than_deleted():
    deleted = build_normalizer([{"name": "remove_punctuation"}])
    spaced = build_normalizer(
        [{"name": "remove_punctuation", "replace_with": " "}, {"name": "whitespace"}]
    )
    assert deleted("a-b") == "ab"
    assert spaced("a-b") == "a b"


def test_remove_tokens_only_removes_whole_tokens():
    normalizer = build_normalizer([{"name": "remove_tokens"}])
    assert normalizer("hello <unk> unknown world") == "hello unknown world"


def test_unicode_folds_full_width():
    normalizer = build_normalizer([{"name": "unicode", "form": "NFKC"}])
    assert normalizer("ＡＢＣ") == "ABC"


def test_config_is_kept_verbatim():
    config = [{"name": "remove_punctuation", "keep": "'"}]
    normalizer = build_normalizer(config)
    assert normalizer.config == config
    # A copy, so that mutating the caller's config cannot change what a
    # reported score claims it was produced with.
    config[0]["keep"] = ""
    assert normalizer.config[0]["keep"] == "'"


def test_unknown_step_is_rejected():
    with pytest.raises(ValueError, match="unknown normalization step"):
        build_normalizer([{"name": "no_such_step"}])


def test_step_without_a_name_is_rejected():
    with pytest.raises(ValueError, match="has no name"):
        build_normalizer([{"keep": "'"}])


def test_whisper_normalization_is_the_real_thing_not_an_approximation():
    """A wrong normalizer is worse than a missing one: it moves every score.

    So the whisper step is openai-whisper's own code, vendored. This checks
    that it is wired up and behaving; test_whisper_normalizer.py checks it
    against the upstream package string for string.
    """
    normalizer = build_normalizer([{"name": "whisper"}])
    assert normalizer("Mr. O'Brien paid $1,250.50") == "mister 0 brien paid $1250.50"
