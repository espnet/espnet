"""Tests for the error-rate metric and the way it is summarized."""

import random

import pytest

from splet.metric_registry import load_metrics, measure_utterances, summarize
from splet.utterance_metrics import error_rate_metric, error_rate_setup


def test_wer_counts_one_substitution():
    state = error_rate_setup()
    result = error_rate_metric(state, "the quik brown fox", "the quick brown fox")
    assert result["wer"] == 0.25
    assert result["wer_errors"] == 1
    assert result["wer_sub"] == 1
    assert result["wer_ref_len"] == 4


def test_cer_counts_characters_including_spaces():
    state = error_rate_setup(name="cer", tokenizer="char")
    result = error_rate_metric(state, "ab cd", "ab cd")
    assert result["cer_ref_len"] == 5
    assert result["cer"] == 0.0


def test_cer_can_ignore_spaces():
    state = error_rate_setup(
        name="cer", tokenizer="char", tokenizer_conf={"remove_space": True}
    )
    result = error_rate_metric(state, "ab cd", "ab cd")
    assert result["cer_ref_len"] == 4


def test_normalization_is_applied_to_both_sides():
    state = error_rate_setup(
        normalize=[{"name": "lowercase"}, {"name": "remove_punctuation"}]
    )
    assert error_rate_metric(state, "Hello, world!", "hello world")["wer"] == 0.0


def test_normalization_config_is_recoverable_from_the_state():
    """A result is only reproducible if its normalization travels with it."""
    state = error_rate_setup(normalize=[{"name": "remove_punctuation", "keep": "'"}])
    assert state["normalizer"].config == [{"name": "remove_punctuation", "keep": "'"}]


def test_summary_pools_counts_rather_than_averaging_rates():
    """The corpus rate is sum(errors) / sum(ref_len), as SCTK reports it.

    The two differ whenever utterances have different lengths, and the
    difference is not small: here the pooled rate is 1/9 and the mean of the
    per-utterance rates is 1/4.
    """
    modules = load_metrics([{"name": "wer"}])
    results = measure_utterances(
        {"short": "x", "long": "a b c d e f g h"},
        modules,
        {"short": "y", "long": "a b c d e f g h"},
    )
    summary = summarize(results)
    assert summary["wer_errors"] == 1
    assert summary["wer_ref_len"] == 9
    assert summary["wer"] == pytest.approx(1 / 9)
    assert summary["num_utterances"] == 2


def test_summary_of_nothing():
    assert summarize([]) == {"num_utterances": 0}


def test_missing_reference_is_an_error_not_a_skipped_utterance():
    """Dropping an unmatched utterance would improve the result silently."""
    modules = load_metrics([{"name": "wer"}])
    with pytest.raises(KeyError, match="utt2"):
        measure_utterances({"utt1": "a", "utt2": "b"}, modules, {"utt1": "a"})


def test_missing_hypothesis_is_an_error_not_a_shorter_denominator():
    """A reference nobody answered must not vanish from the denominator."""
    modules = load_metrics([{"name": "wer"}])
    with pytest.raises(KeyError, match="no hypothesis for reference 'utt2'"):
        measure_utterances({"utt1": "a"}, modules, {"utt1": "a", "utt2": "b c"})


def test_empty_hypothesis_counts_every_reference_word_as_deleted():
    """No output is a legitimate hypothesis, and it is all deletions."""
    modules = load_metrics([{"name": "wer"}])
    (result,) = measure_utterances({"utt1": ""}, modules, {"utt1": "a b c"})
    assert result["wer"] == 1.0
    assert (result["wer_sub"], result["wer_del"], result["wer_ins"]) == (0, 3, 0)


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown metric"):
        load_metrics([{"name": "no_such_metric"}])


def test_unknown_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        error_rate_setup(tokenizer="no_such_tokenizer")


@pytest.mark.parametrize("unit", ["word", "char"])
def test_matches_jiwer(unit):
    """Regression against the implementation espnet3 uses today.

    espnet3's WER and CER call jiwer, so SPLET's must agree with it on the
    corpus figure before it can replace them anywhere.
    """
    jiwer = pytest.importorskip("jiwer")

    random.seed(1234)
    vocabulary = "the quick brown fox jumps over a lazy dog".split()
    references, hypotheses = [], []
    for _ in range(40):
        reference = [random.choice(vocabulary) for _ in range(random.randint(1, 15))]
        hypothesis = [
            word for word in reference if random.random() > 0.2  # deletions
        ] + [random.choice(vocabulary) for _ in range(random.randint(0, 3))]
        random.shuffle(hypothesis)
        references.append(" ".join(reference))
        hypotheses.append(" ".join(hypothesis) or "x")

    modules = load_metrics([{"name": "wer" if unit == "word" else "cer"}])
    results = measure_utterances(
        {str(i): hypothesis for i, hypothesis in enumerate(hypotheses)},
        modules,
        {str(i): reference for i, reference in enumerate(references)},
    )
    summary = summarize(results)

    expected = (
        jiwer.wer(references, hypotheses)
        if unit == "word"
        else jiwer.cer(references, hypotheses)
    )
    assert summary["wer" if unit == "word" else "cer"] == pytest.approx(expected)
