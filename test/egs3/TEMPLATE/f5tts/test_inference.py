"""Tests for the template's ``src/inference.py`` output formatter."""

import numpy as np
import pytest
import torch

from egs3.TEMPLATE.f5tts.src.inference import build_output
from espnet3.api.inference import Audio


def test_build_output_formats_one_sample() -> None:
    output = build_output(
        {"text": "hello"}, {"wav": np.zeros((1, 4), dtype=np.float64)}, 7
    )

    assert output["utt_id"] == "7"
    assert output["text"] == "hello"
    assert output["wav"].dtype == np.float32
    assert output["wav"].shape == (4,)


def test_build_output_prefers_the_sample_utt_id() -> None:
    output = build_output(
        {"utt_id": "spk_0001", "text": "hello"}, {"wav": np.zeros(2)}, 0
    )

    assert output["utt_id"] == "spk_0001"


def test_build_output_takes_the_samples_of_an_audio() -> None:
    """``Inference`` returns an Audio; the wav writer wants the bare samples."""
    samples = np.linspace(-1, 1, 8, dtype=np.float32)

    output = build_output({"text": "hello"}, {"wav": Audio(samples, 24000)}, 0)

    assert isinstance(output["wav"], np.ndarray)
    np.testing.assert_array_equal(output["wav"], samples)


def test_build_output_detaches_a_tensor() -> None:
    wav = torch.ones(3, requires_grad=True)

    output = build_output({"text": "hello"}, {"wav": wav}, 0)

    assert isinstance(output["wav"], np.ndarray)
    assert output["wav"].tolist() == [1.0, 1.0, 1.0]


def test_build_output_requires_a_wav() -> None:
    with pytest.raises(RuntimeError, match="does not contain 'wav'"):
        build_output({"text": "hello"}, {}, 0)


@pytest.mark.parametrize(
    "model_output",
    [
        # One mapping per item, as ``Inference`` returns for a batch.
        [{"wav": np.zeros(2)}, {"wav": np.ones(3)}],
        # One mapping holding a list, as the bare ``F5TTSInference`` returns.
        {"wav": [np.zeros(2), np.ones(3)]},
    ],
)
def test_build_output_formats_a_batch(model_output) -> None:
    outputs = build_output(
        [{"text": "first"}, {"text": "second"}], model_output, [4, 5]
    )

    assert [output["utt_id"] for output in outputs] == ["4", "5"]
    assert [output["text"] for output in outputs] == ["first", "second"]
    assert [output["wav"].shape for output in outputs] == [(2,), (3,)]
