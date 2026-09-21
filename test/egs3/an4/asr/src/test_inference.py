"""Regression tests for full AN4 preparation and migration settings."""

from pathlib import Path

import pytest

from egs3.an4.asr.src.inference import build_output

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/an4/asr"


def test_output_pairs_single_and_batched_references():
    """Keep hypotheses aligned with their references in both modes."""
    data = {"text": "HELLO"}
    hypothesis = [("WORLD", None, None, None)]
    assert build_output(data, hypothesis, 3) == {
        "utt_id": "3",
        "ref": "HELLO",
        "hyp": "WORLD",
    }
    outputs = build_output([data, data], [hypothesis, [(None,)]], [3, 4])
    assert outputs[1] == {"utt_id": "4", "ref": "HELLO", "hyp": ""}
    with pytest.raises(ValueError):
        build_output([data, data], [hypothesis], [3, 4])
