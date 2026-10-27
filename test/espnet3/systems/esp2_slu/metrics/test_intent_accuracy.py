"""Tests for the esp2_slu intent accuracy metric."""

from pathlib import Path

import pytest

from espnet3.systems.esp2_slu.metrics.intent_accuracy import IntentAccuracy


def test_accuracy_pools_counts_across_updates():
    """`compute` scores every utterance seen since the last reset."""
    metric = IntentAccuracy()
    metric.update(ref=["news_query", "alarm_set"], hyp=["news_query", "alarm_set"])
    metric.update(ref=["audio_volume_mute"], hyp=["calendar_query"])

    assert metric.compute() == {"IntentAccuracy": 66.67}


def test_labels_are_compared_after_stripping():
    metric = IntentAccuracy()
    metric.update(ref=[" news_query "], hyp=["news_query"])

    assert metric.compute() == {"IntentAccuracy": 100.0}


def test_scoring_nothing_is_an_error():
    """An empty test set is a mistake, not an accuracy of zero."""
    with pytest.raises(RuntimeError, match="No utterance to score"):
        IntentAccuracy().compute()


def test_reset_drops_counts_and_errors(tmp_path: Path):
    metric = IntentAccuracy()
    metric.collect_artifacts = True
    metric.update(ref=["news_query"], hyp=["alarm_set"])
    metric.reset()
    metric.update(ref=["news_query"], hyp=["news_query"])

    assert metric.compute() == {"IntentAccuracy": 100.0}
    metric.write_artifacts(tmp_path)
    assert (tmp_path / "intent_errors").read_text(encoding="utf-8").splitlines() == [
        "idx\tref_intent\thyp_intent"
    ]


def test_misclassified_rows_are_numbered_by_dataset_index(tmp_path: Path):
    """`update` gets no IDs, so rows are numbered by the running total.

    The recipe's IDs are dataset indices, so a running counter reproduces them
    as long as batches arrive in dataset order.
    """
    metric = IntentAccuracy()
    metric.collect_artifacts = True
    metric.update(ref=["news_query", "alarm_set"], hyp=["news_query", "alarm_query"])
    metric.update(ref=["audio_volume_mute"], hyp=["calendar_query"])

    metric.write_artifacts(tmp_path)

    rows = (tmp_path / "intent_errors").read_text(encoding="utf-8").splitlines()
    assert rows == [
        "idx\tref_intent\thyp_intent",
        "1\talarm_set\talarm_query",
        "2\taudio_volume_mute\tcalendar_query",
    ]


def test_errors_are_not_collected_when_not_asked_for(tmp_path: Path):
    """`collect_artifacts` is off outside a run that writes them."""
    metric = IntentAccuracy()
    metric.update(ref=["news_query"], hyp=["alarm_set"])

    metric.write_artifacts(tmp_path)

    assert (tmp_path / "intent_errors").read_text(encoding="utf-8").splitlines() == [
        "idx\tref_intent\thyp_intent"
    ]


def test_write_errors_false_writes_nothing(tmp_path: Path):
    metric = IntentAccuracy(write_errors=False)
    metric.collect_artifacts = True
    metric.update(ref=["news_query"], hyp=["alarm_set"])

    metric.write_artifacts(tmp_path)

    assert not (tmp_path / "intent_errors").exists()
