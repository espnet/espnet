"""The codec preprocessor hands the `speech` field to the model as `audio`."""

import numpy as np
import pytest

from espnet3.systems.esp2_codec.src.preprocessor import CodecPreprocessor


def _make(**kwargs):
    """Build a preprocessor with the template's settings plus overrides."""
    options = dict(
        train=False,
        token_type=None,
        speech_name="speech",
        force_single_channel=True,
        audio_pad_value=0.0,
    )
    options.update(kwargs)
    return CodecPreprocessor(**options)


def test_speech_is_handed_over_as_audio():
    """`speech` leaves the item and the same samples arrive under `audio`."""
    waveform = np.arange(4, dtype=np.float32)
    out = _make()("utt1", {"speech": waveform})
    assert sorted(out) == ["audio"]
    np.testing.assert_array_equal(out["audio"], waveform)


def test_common_processing_runs_before_the_rename():
    """`force_single_channel` averages a stereo item, then it is renamed."""
    stereo = np.stack([np.ones(4), np.zeros(4)], axis=1).astype(np.float32)
    out = _make()("utt1", {"speech": stereo})
    assert out["audio"].shape == (4,)
    np.testing.assert_allclose(out["audio"], 0.5)


def test_other_fields_pass_through():
    """Fields the preprocessor does not own are untouched."""
    out = _make()("utt1", {"speech": np.zeros(2, dtype=np.float32), "utt_id": "utt1"})
    assert sorted(out) == ["audio", "utt_id"]
    assert out["utt_id"] == "utt1"


def test_item_without_speech_is_untouched():
    """An item that carries no `speech` is returned as is."""
    other = np.zeros(2, dtype=np.float32)
    out = _make()("utt1", {"other": other})
    assert sorted(out) == ["other"]
    assert out["other"] is other


def test_model_input_name_is_configurable():
    """The target keyword follows `model_input_name`."""
    out = _make(model_input_name="wav")("utt1", {"speech": np.zeros(2, np.float32)})
    assert sorted(out) == ["wav"]


def test_same_name_means_no_rename():
    """With `speech_name` equal to `model_input_name` nothing is renamed."""
    out = _make(speech_name="audio")("utt1", {"audio": np.zeros(2, np.float32)})
    assert sorted(out) == ["audio"]


def test_both_names_present_is_an_error():
    """An item carrying both keys would lose one silently, so it is refused."""
    item = {"speech": np.zeros(2, np.float32), "audio": np.ones(2, np.float32)}
    with pytest.raises(KeyError, match="both 'speech' and 'audio'"):
        _make()("utt1", item)
