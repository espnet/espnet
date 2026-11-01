"""Tests for splet/corpus_metrics/bleu.py, the sacrebleu-backed corpus metrics."""

import pytest

from splet.corpus_metrics import (
    bleu_setup,
    chrf_setup,
    sacrebleu_metric,
    translation_edit_rate_setup,
)

pytest.importorskip("sacrebleu")

REFS = [
    "The cat sat on the mat.",
    "Hello, world!",
    "It is raining again today, isn't it?",
    "We'll meet at the station at 5 p.m.",
]
HYPS = [
    "the cat sat on a mat",
    "Hello world!",
    "It rains again today, isn't it?",
    "We will meet at the station at 5 pm.",
]


def test_corpus_bleu_is_not_the_mean_of_sentence_bleus():
    state = bleu_setup()
    corpus = sacrebleu_metric(state, HYPS, REFS)["bleu"]
    sentences = [sacrebleu_metric(state, [h], [r])["bleu"] for h, r in zip(HYPS, REFS)]
    assert corpus != pytest.approx(sum(sentences) / len(sentences), abs=0.1)


def test_bleu_reports_brevity_penalty_and_lengths():
    result = sacrebleu_metric(bleu_setup(), ["a b c"], ["a b c d e f"])
    assert result["bleu_hyp_len"] == 3
    assert result["bleu_ref_len"] == 6
    assert result["bleu_ratio"] == pytest.approx(0.5)
    assert result["bleu_bp"] < 1.0


def test_normalization_is_applied_to_both_sides():
    normalize = [{"name": "lowercase"}, {"name": "remove_punctuation", "keep": "'"}]
    hyps, refs = ["hello WORLD it's fine"], ["Hello, world! It's fine."]
    assert sacrebleu_metric(bleu_setup(), hyps, refs)["bleu"] < 100.0
    for setup, name, perfect in (
        (bleu_setup, "bleu", 100.0),
        (chrf_setup, "chrf", 100.0),
        (translation_edit_rate_setup, "translation_edit_rate", 0.0),
    ):
        result = sacrebleu_metric(setup(normalize=normalize), hyps, refs)
        assert result[name] == pytest.approx(perfect)


def test_bleu_needs_references():
    with pytest.raises(ValueError, match="references"):
        sacrebleu_metric(bleu_setup(), ["a"], None)


def test_unequal_hypothesis_and_reference_counts_are_an_error():
    # sacrebleu scores only the overlap, so a missing reference would
    # silently change the corpus figure.
    with pytest.raises(ValueError, match="2 hypotheses but 1 references"):
        sacrebleu_metric(chrf_setup(), ["good", "bad"], ["good"])
