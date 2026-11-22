"""PCM scaling at the shared audio input boundary."""

import numpy as np
import pytest

from espnet3.contract.kinds.audio import Audio


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_signed_pcm_uses_full_negative_range(dtype):
    """Signed PCM extrema scale correctly without modifying the source."""
    limits = np.iinfo(dtype)
    scale = 1 << (limits.bits - 1)
    source = np.array(
        [limits.min, -(scale // 2), -1, 0, 1, scale // 2, limits.max],
        dtype=dtype,
    )
    original = source.copy()

    audio = Audio(source, 16000)

    expected = np.array(
        [-1.0, -0.5, -1.0 / scale, 0.0, 1.0 / scale, 0.5, 1.0 - 1.0 / scale],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(audio.array, expected)
    np.testing.assert_array_equal(source, original)
    assert audio.array.dtype == np.float32
    assert audio.rate == 16000


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_unsigned_pcm_remains_centered_at_midpoint(dtype):
    """Unsigned PCM keeps its existing midpoint-based scaling."""
    limits = np.iinfo(dtype)
    midpoint = 1 << (limits.bits - 1)
    source = np.array([0, midpoint, limits.max], dtype=dtype)
    original = source.copy()

    audio = Audio(source, 16000)

    expected = np.array([-1.0, 0.0, 1.0 - 1.0 / midpoint], dtype=np.float32)
    np.testing.assert_array_equal(audio.array, expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_audio_is_not_rescaled_or_clipped(dtype):
    """Float waveforms are cast to float32 without amplitude normalization."""
    source = np.array([-1.25, -0.5, 0.0, 0.5, 1.25], dtype=dtype)
    original = source.copy()

    audio = Audio(source, 16000)

    np.testing.assert_array_equal(audio.array, source.astype(np.float32))
    np.testing.assert_array_equal(source, original)


def test_pcm_rate_pair_uses_the_same_scaling():
    """A rate/PCM pair follows the same conversion as direct construction."""
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)

    audio = Audio.coerce((16000, pcm), rate=None)

    np.testing.assert_array_equal(
        audio.array, np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32)
    )
    assert audio.rate == 16000


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e300])
def test_audio_rejects_nonfinite_float32_samples(value):
    """Invalid samples, including overflow during conversion, fail early."""
    with pytest.raises(ValueError, match="finite"):
        Audio(np.array([0.0, value]), 16000)


@pytest.mark.parametrize("value", [["1", "2"], [True, False], [1j, 2j]])
def test_audio_rejects_non_audio_sample_types(value):
    """Text, booleans and complex values are not PCM waveforms."""
    with pytest.raises(TypeError, match="real numeric"):
        Audio(np.array(value), 16000)


@pytest.mark.parametrize("shape", [(2, 1), (1, 2), (2, 2)])
def test_ambiguous_short_audio_defaults_to_channels_first(shape):
    """Short mono/stereo data follows the canonical axis convention."""
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    audio = Audio(source, 16000)
    np.testing.assert_array_equal(audio.array, source[0] if shape[0] == 1 else source)
    assert audio.channels == shape[0]


def test_empty_final_chunk_remains_representable():
    """An empty final streaming flush is distinct from invalid sample values."""
    audio = Audio(np.empty(0, dtype=np.float32), 16000)
    assert audio.array.shape == (0,)


def test_existing_audio_cannot_bypass_finite_sample_validation():
    audio = Audio(np.ones(4, dtype=np.float32), 16000)
    audio.array[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Audio.coerce(audio, 16000)


@pytest.mark.parametrize("rate", [0, -1])
def test_audio_rejects_nonpositive_rate(rate):
    """An invalid source rate fails before resampling or inference."""
    with pytest.raises(ValueError, match="sample rate"):
        Audio(np.zeros(16), rate)


@pytest.mark.parametrize("shape", [(), (1, 2, 3)])
def test_audio_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="1-D or 2-D"):
        Audio(np.zeros(shape), 16000)


def test_audio_representations_and_noncontiguous_views_agree(tmp_path):
    import soundfile as sf
    import torch

    pcm = np.arange(-32, 32, dtype=np.int16)
    expected = pcm.astype(np.float32) / 32768
    path = tmp_path / "pcm.wav"
    sf.write(path, pcm, 16000, subtype="PCM_16")
    view = np.column_stack((expected, expected))[:, 0]
    assert not view.flags.c_contiguous
    for source in (path, (16000, pcm), Audio(pcm, 16000), view, torch.from_numpy(pcm)):
        audio = Audio.coerce(source, 16000)
        np.testing.assert_array_equal(audio.array, expected)
        assert audio.array.flags.c_contiguous
        assert audio.array.dtype == np.float32


@pytest.mark.parametrize("shape", [(160,), (1, 160), (160, 1), (2, 160), (160, 2)])
def test_silent_waveform_layouts_preserve_channels(shape):
    audio = Audio(np.zeros(shape, dtype=np.float32), 16000)
    assert audio.array.shape == ((2, 160) if 2 in shape else (160,))
    assert audio.duration == 0.01
    assert not audio.array.any()
