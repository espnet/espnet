import pytest

from espnet2.asr.postprocess.sot_postprocess import (
    count_ngrams,
    truncate_at_repeating_ngram,
)


def test_count_ngrams_basic():
    text = "the cat sat on the mat"
    counts = count_ngrams(text, min_n=2, max_n=2)
    assert counts["the cat"] == 1
    assert counts["the mat"] == 1


def test_count_ngrams_skips_identical_words():
    text = "yes yes yes yes"
    counts = count_ngrams(text, min_n=2, max_n=3)
    # All-same-word n-grams should be skipped
    assert "yes yes" not in counts
    assert "yes yes yes" not in counts


def test_truncate_short_text():
    text = "hello world"
    result = truncate_at_repeating_ngram(text)
    assert result == text  # below min_word_threshold, unchanged


def test_truncate_no_repetition():
    words = [f"word{i}" for i in range(40)]
    text = " ".join(words)
    result = truncate_at_repeating_ngram(text)
    assert result == text  # no repetition, unchanged


def test_truncate_unigram_repetition():
    text = "start " + "hello " * 15 + "end"
    result = truncate_at_repeating_ngram(text, min_word_threshold=5)
    # Should truncate at the repeating unigram
    assert len(result.split()) < len(text.split())


def test_truncate_ngram_repetition():
    base = " ".join(f"w{i}" for i in range(5))
    text = " ".join([base] * 15)
    result = truncate_at_repeating_ngram(text, min_word_threshold=5, repeat_threshold=3)
    assert len(result.split()) < len(text.split())
