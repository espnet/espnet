"""Decoded enrollment audio must retain native TSE crop semantics."""

import numpy as np
import pytest

from espnet2.train.preprocessor import TSEPreprocessor


@pytest.mark.parametrize("train", [False, True])
def test_decoded_enrollment_is_supported(train):
    preprocessor = TSEPreprocessor(
        train=train, load_all_speakers=True, num_spk=1, enroll_segment=32
    )
    audio = np.random.default_rng(0).normal(size=128).astype(np.float32)
    sample = {
        "speech_mix": audio.copy(),
        "speech_ref1": audio.copy(),
        "enroll_ref1": audio.copy(),
    }
    result = preprocessor("u1", sample)
    assert result["enroll_ref1"].shape == (32,)
    assert np.isfinite(result["enroll_ref1"]).all()


def test_empty_decoded_enrollment_rejected():
    preprocessor = TSEPreprocessor(train=False, enroll_segment=32)
    with pytest.raises(ValueError, match="nonempty"):
        preprocessor._read_audio_segment(np.array([], dtype=np.float32), 32)
