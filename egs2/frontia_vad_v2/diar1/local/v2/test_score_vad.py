"""Tests of the frame time of local/v2/score_vad.py (latency 0 at look-ahead 0)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_vad  # noqa: E402


def test_frame_interval_ends_at_window_end():
    # frame i reads samples [128 i, 128 i + 512); its scored interval ends at 128 i + 512
    for i in (0, 1, 7, 100):
        prob = np.zeros(i + 3, dtype=np.float32)
        prob[i] = 1.0
        ((start, duration),) = score_vad.segments(prob, 0.5, 1)
        assert np.isclose(start + duration, (128 * i + 512) / 16000)
        assert np.isclose(duration, 128 / 16000)


def test_explicit_offset_overrides_default():
    prob = np.array([1.0, 0.0], dtype=np.float32)
    ((start, duration),) = score_vad.segments(prob, 0.5, 1, shift=0.01, offset=0.005)
    assert np.isclose(start, 0.0) and np.isclose(duration, 0.01)
