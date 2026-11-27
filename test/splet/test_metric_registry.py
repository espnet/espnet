"""Tests for the corpus tier of splet/metric_registry.py."""

import pytest

from splet.corpus_metrics import bleu_setup, sacrebleu_metric
from splet.metric_registry import load_corpus_metrics, measure_corpus

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


def test_utterances_are_matched_by_key_not_by_order():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in reversed(list(enumerate(HYPS)))}
    in_order = sacrebleu_metric(bleu_setup(), HYPS, REFS)["bleu"]
    assert measure_corpus(pred, metrics, gt)["bleu"] == pytest.approx(in_order)


def test_hypothesis_without_reference_is_an_error():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    with pytest.raises(KeyError, match="utt9"):
        measure_corpus({"utt0": "a", "utt9": "b"}, metrics, {"utt0": "a"})


def test_reference_without_hypothesis_is_an_error():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    with pytest.raises(KeyError, match="utt1' and 2 more"):
        measure_corpus({"utt0": HYPS[0]}, metrics, gt)


def test_empty_hypothesis_is_measured_not_dropped():
    metrics = load_corpus_metrics([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in enumerate(HYPS)}
    full = measure_corpus(pred, metrics, gt)
    pred["utt3"] = ""
    empty = measure_corpus(pred, metrics, gt)
    assert empty["bleu_ref_len"] == full["bleu_ref_len"]
    assert empty["bleu"] < full["bleu"]


def test_corpus_tier_checks_for_a_reference_before_measuring():
    metrics = load_corpus_metrics([{"name": "chrf"}])
    with pytest.raises(ValueError, match="metric 'chrf' requires a reference"):
        measure_corpus({"utt0": "a"}, metrics, None)
