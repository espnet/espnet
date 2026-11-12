"""TSOS padding and threshold regression from the original local tests."""

from unittest.mock import patch

import pytest
import torch

from espnet3.systems.esp2_tse.metrics.tsos import TSOS


def _import_tsos():
    return TSOS


def test_tsos_ignores_padded_frames_and_uses_threshold():
    """Thresholded oversuppression should ignore padded STFT frames when averaging."""
    with patch("espnet3.systems.esp2_tse.metrics.tsos.STFTEncoder"):
        try:
            TSOS = _import_tsos()
        except pytest.skip.Exception:
            pytest.skip("espnet2 not available")
        metric = TSOS.__new__(TSOS)
        metric.power = 1.0
        metric.threshold = 0.1

    ref_spec = torch.tensor(
        [
            [[10.0], [10.0]],
            [[10.0], [10.0]],
        ],
        dtype=torch.float32,
    )
    inf_spec = torch.tensor(
        [
            [[0.0], [0.0]],
            [[0.0], [10.0]],
        ],
        dtype=torch.float32,
    )
    flens = torch.tensor([2, 1], dtype=torch.long)

    scores = metric._compute_tsos(ref_spec, inf_spec, flens)

    assert scores == pytest.approx([1.0, 1.0])
