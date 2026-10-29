"""Tests for ARECHO metric collation."""

import random

import numpy as np
import pytest

from espnet2.aqa.ar_universa.data import ARMetricCollateFn


@pytest.mark.parametrize("randomize", [False, True])
def test_collate_sparse_metrics_preserves_pairs(randomize, monkeypatch):
    """Shuffle whole pairs and pad missing metrics without modifying inputs."""
    monkeypatch.setattr("random.shuffle", lambda pairs: pairs.reverse())
    labels = {"mos": (4, 7), "language": (10, 12)}
    samples = [
        ("a", dict(audio=np.ones((5, 8), dtype=np.float32), metrics=labels)),
        ("b", dict(audio=np.ones((3, 8), dtype=np.float32))),
    ]
    keys, batch = ARMetricCollateFn(randomize=randomize)(samples)
    assert keys == ["a", "b"]
    assert batch["audio_lengths"].tolist() == [5, 3]
    assert batch["metrics"]["metric_token_lengths"].tolist() == [4, 0]
    expected = [10, 12, 4, 7] if randomize else [4, 7, 10, 12]
    assert batch["metrics"]["metric_token"].tolist() == [expected, [0, 0, 0, 0]]
    assert list(labels.values()) == [(4, 7), (10, 12)]
    _, unlabelled = ARMetricCollateFn()([samples[1]])
    assert unlabelled["metrics"]["metric_token"].shape == (1, 0)
    assert unlabelled["metrics"]["metric_token_lengths"].tolist() == [0]


def _collate_labels(collate, labels):
    """Return the label/value pairs of one collated sample."""
    sample = dict(audio=np.ones((2, 8), dtype=np.float32), metrics=labels)
    tokens = collate([("a", sample)])[1]["metrics"]["metric_token"][0].tolist()
    return list(zip(tokens[::2], tokens[1::2]))


@pytest.mark.parametrize("randomize", [False, True])
def test_defer_full_meta_orders_whole_clip_pairs_last(randomize, monkeypatch):
    """Shuffle within prefix and whole-clip groups, never across them."""
    labels = {
        "mos_full": (14, 15),
        "mos": (4, 5),
        "language_full": (20, 21),
        "language": (10, 11),
        "noise": (24, 25),
    }
    collate = ARMetricCollateFn(randomize=randomize, defer_full_meta=True)
    monkeypatch.setattr("random.shuffle", random.Random(0).shuffle)
    orders = set()
    for _ in range(20):
        pairs = _collate_labels(collate, labels)
        assert sorted(pairs) == sorted(labels.values())
        assert {label for label, _ in pairs[3:]} == {14, 20}
        orders.add(tuple(pairs))
    if randomize:
        assert len(orders) > 1
    else:
        # Input order is kept within each group.
        assert orders == {((4, 5), (10, 11), (24, 25), (14, 15), (20, 21))}


@pytest.mark.parametrize("defer_full_meta", [False, True])
def test_defer_full_meta_is_opt_in(defer_full_meta, monkeypatch):
    """Without the option, whole-clip pairs are shuffled like any other pair."""
    monkeypatch.setattr("random.shuffle", lambda pairs: pairs.reverse())
    labels = {"mos": (4, 5), "mos_full": (14, 15), "language": (10, 11)}
    collate = ARMetricCollateFn(randomize=True, defer_full_meta=defer_full_meta)
    expected = [(10, 11), (4, 5), (14, 15)]
    if not defer_full_meta:
        expected = [(10, 11), (14, 15), (4, 5)]
    assert _collate_labels(collate, labels) == expected
    assert ARMetricCollateFn().defer_full_meta is False
