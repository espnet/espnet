"""Tests for ARECHO metric collation."""

import numpy as np
import pytest

from espnet2.universa.ar_universa.data import ARMetricCollateFn


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
