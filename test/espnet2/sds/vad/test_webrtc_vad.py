import numpy as np
import pytest
import soundfile

from espnet2.sds.vad.webrtc_vad import WebrtcVADModel

pytest.importorskip("webrtcvad")


def test_forward():
    vad_model = WebrtcVADModel()
    vad_model.warmup()
    x, rate = soundfile.read("test_utils/ctc_align_test.wav", dtype="int16")
    vad_model.forward(x, rate)
    vad_model.forward(x, rate, binary=True)


class _LoudIsSpeech:
    """Stand-in for webrtcvad.Vad: a frame is speech when it is not zero."""

    def __init__(self, mode=3):
        pass

    def is_speech(self, frame, sample_rate):
        return any(frame)


def _vad(monkeypatch, *args, **kwargs):
    from espnet2.sds.vad import webrtc_vad

    monkeypatch.setattr(webrtc_vad.webrtcvad, "Vad", _LoudIsSpeech)
    return WebrtcVADModel(*args, **kwargs)


def _feed(vad, signal, chunk_s, rate=16000, binary=True):
    """Feed signal in chunks; return (turn, last_turn) of each turn returned."""
    step = int(chunk_s * rate)
    turns = []
    for i in range(0, len(signal), step):
        out = vad(signal[i : i + step], rate, binary=binary)
        if out is not None:
            turns.append((out, dict(vad.last_turn)))
    return turns


def _speech_then_silence(speech_s, silence_s, rate=16000):
    return np.concatenate(
        [
            np.zeros(int(0.5 * rate), dtype=np.int16),
            np.full(int(speech_s * rate), 1000, dtype=np.int16),
            np.zeros(int(silence_s * rate), dtype=np.int16),
        ]
    )


@pytest.mark.parametrize("chunk_s", [0.02, 0.1, 0.5])
def test_any_chunk_length_gives_the_same_turn(monkeypatch, chunk_s):
    """20 ms chunks work: the old per-chunk rule needed chunks of 0.26 s."""
    vad = _vad(monkeypatch)
    turns = _feed(vad, _speech_then_silence(1.0, 1.5), chunk_s)
    assert len(turns) == 1
    turn, info = turns[0]
    assert info["speech_start_s"] == pytest.approx(0.5, abs=0.02)
    assert info["endpoint_s"] == pytest.approx(1.5, abs=0.02)
    # 300 ms of padding on each side
    assert len(turn) == pytest.approx(1.6 * 16000, abs=0.04 * 16000)


def test_a_short_turn_is_dropped_and_forgotten(monkeypatch):
    vad = _vad(monkeypatch, min_speech_ms=500)
    signal = np.concatenate(
        [_speech_then_silence(0.3, 1.0), _speech_then_silence(1.0, 1.0)]
    )
    turns = _feed(vad, signal, 0.02)
    # the 1 s turn alone, not glued to the dropped 300 ms one
    assert len(turns) == 1
    assert turns[0][1]["endpoint_s"] - turns[0][1]["speech_start_s"] == (
        pytest.approx(1.0, abs=0.02)
    )


def test_a_long_turn_is_split_and_memory_stays_bounded(monkeypatch):
    vad = _vad(monkeypatch, max_turn_ms=2000)
    turns = _feed(vad, np.full(7 * 16000, 1000, dtype=np.int16), 0.1)
    assert len(turns) == 3 and all(info["forced"] for _, info in turns)
    assert len(vad.audio) < 3 * 16000


def test_deprecated_arguments_warn(monkeypatch):
    with pytest.warns(DeprecationWarning, match="speakup_threshold"):
        _vad(monkeypatch, speakup_threshold=12)
    with pytest.warns(DeprecationWarning, match="max_speech_ms"):
        vad = _vad(monkeypatch, max_speech_ms=4000.0)
    assert vad.max_turn_frames == 200


def test_reset_forgets_the_turn_in_progress(monkeypatch):
    vad = _vad(monkeypatch)
    _feed(vad, np.full(16000, 1000, dtype=np.int16), 0.1)
    vad.reset()
    assert _feed(vad, np.zeros(16000, dtype=np.int16), 0.1) == []


def test_audio_at_another_rate_is_resampled_for_webrtc(monkeypatch):
    vad = _vad(monkeypatch)
    rate = 22050
    turns = _feed(vad, _speech_then_silence(1.0, 1.5, rate), 0.1, rate)
    assert len(turns) == 1
    assert turns[0][1]["endpoint_s"] == pytest.approx(1.5, abs=0.03)


def test_memory_bounded_over_many_turns(monkeypatch):
    from test.memory_growth import assert_memory_bounded

    vad = _vad(monkeypatch)
    period = _speech_then_silence(1.0, 1.0)
    chunk = [0]

    def step():
        i = chunk[0] % (len(period) // 1600)
        vad(period[i * 1600 : (i + 1) * 1600], 16000)
        chunk[0] += 1

    assert_memory_bounded(vad, step, steps=6000)  # 10 minutes in 0.1 s chunks


def test_old_positional_arguments_keep_their_meaning(monkeypatch):
    """WebrtcVADModel(speakup, continue, min_speech, max_speech, target_sr)."""
    with pytest.warns(DeprecationWarning):
        vad = _vad(monkeypatch, 3, 3, 500, 4000.0, 8000)
    assert vad.min_speech_frames == 25  # 500 ms of 20 ms frames
    assert vad.max_turn_frames == 200  # max_speech_ms 4000 -> max_turn_ms
    assert vad.target_sr == 8000


def test_new_controls_are_keyword_only(monkeypatch):
    from espnet2.sds.vad import webrtc_vad

    monkeypatch.setattr(webrtc_vad.webrtcvad, "Vad", _LoudIsSpeech)
    with pytest.raises(TypeError):
        WebrtcVADModel(None, None, 500.0, None, 16000, 2)
    assert WebrtcVADModel(mode=2, median=3).median == 3
