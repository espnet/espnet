"""Check output alignment for shared ASR inference."""

import pytest

from egs3.voices.esp2_asr.src.inference import build_output


def test_output_alignment_and_empty_prediction():
    """Keep every reference, including when a smoke model predicts no text."""
    output = build_output([{"text": "A"}, {"text": "B"}], [[("C",)], [(None,)]], [3, 4])
    assert output == [
        dict(utt_id="3", ref="A", hyp="C"),
        dict(utt_id="4", ref="B", hyp=""),
    ]
    with pytest.raises(ValueError):
        build_output([{"text": "A"}], [], [0])
