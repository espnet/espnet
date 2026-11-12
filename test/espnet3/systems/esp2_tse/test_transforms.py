"""Tests for the LibriMix TSE recipe per-sample transforms."""

import numpy as np
import pytest
import soundfile as sf

from espnet3.systems.esp2_tse.src.transforms import TSEAudioLoadTransform


def _load_transform_class():
    return TSEAudioLoadTransform


@pytest.fixture(scope="module")
def transform_cls():
    return _load_transform_class()


@pytest.fixture()
def wav_path(tmp_path):
    path = tmp_path / "utt.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    return str(path)


def test_decodes_speech_fields(transform_cls, wav_path):
    """Mixture and reference paths are decoded into waveform arrays."""
    transform = transform_cls()
    out = transform({"speech_mix": wav_path, "speech_ref1": wav_path})
    assert isinstance(out["speech_mix"], np.ndarray)
    assert isinstance(out["speech_ref1"], np.ndarray)
    assert out["speech_mix"].shape == (1600,)


def test_key_set_is_preserved(transform_cls, wav_path):
    """The transform must not add or drop keys (CombinedDataset asserts this)."""
    sample = {"speech_mix": wav_path, "enroll_ref1": wav_path, "num_spk": 2}
    out = transform_cls(load_enrollment_as_audio=True)(sample)
    assert set(out) == set(sample)
    assert out["num_spk"] == 2


def test_enrollment_left_alone_by_default(transform_cls, wav_path):
    """With load_enrollment=False the enrollment path is passed through."""
    out = transform_cls()({"speech_mix": wav_path, "enroll_ref1": wav_path})
    assert out["enroll_ref1"] == wav_path


def test_enrollment_decoded_when_requested(transform_cls, wav_path):
    """With load_enrollment_as_audio=True a real enrollment path is decoded."""
    out = transform_cls(load_enrollment_as_audio=True)({"enroll_ref1": wav_path})
    assert isinstance(out["enroll_ref1"], np.ndarray)


def test_training_placeholder_is_passed_through(transform_cls):
    """'*UTT_ID SPEAKER_ID' is resolved by TSEPreprocessor, not here."""
    placeholder = "*1272-128104-0000 1272"
    out = transform_cls(load_enrollment_as_audio=True)({"enroll_ref1": placeholder})
    assert out["enroll_ref1"] == placeholder


def test_already_decoded_array_is_idempotent(transform_cls):
    """A field decoded by the Dataset is passed through unchanged."""
    audio = np.zeros(1600, dtype=np.float32)
    out = transform_cls()({"speech_mix": audio})
    assert out["speech_mix"] is audio


def test_rejects_unsupported_field_type(transform_cls):
    """A non-path, non-array audio field raises TypeError."""
    with pytest.raises(TypeError, match="speech_mix"):
        transform_cls()({"speech_mix": 42})


def test_rejects_mixed_sampling_rates(transform_cls, tmp_path):
    """Fields decoded from one sample must share a sampling rate."""
    path_16k = tmp_path / "a.wav"
    path_8k = tmp_path / "b.wav"
    sf.write(path_16k, np.zeros(1600, dtype=np.float32), 16000)
    sf.write(path_8k, np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(ValueError, match="sampling rate"):
        transform_cls()({"speech_mix": str(path_16k), "speech_ref1": str(path_8k)})
