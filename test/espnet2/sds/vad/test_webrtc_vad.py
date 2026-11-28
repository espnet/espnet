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
    """Stand-in for webrtcvad.Vad: a 20 ms frame is speech when it is not zero."""

    def set_mode(self, mode):
        pass

    def is_speech(self, frame, sample_rate):
        return any(frame)


def _vad(monkeypatch, **kwargs):
    from espnet2.sds.vad import webrtc_vad

    monkeypatch.setattr(webrtc_vad.webrtcvad, "Vad", _LoudIsSpeech)
    return WebrtcVADModel(**kwargs)


def _chunk(seconds, speech, rate=16000):
    return np.full(int(seconds * rate), 1000 if speech else 0, dtype=np.int16)


def test_turn_ends_with_the_first_silent_chunk(monkeypatch):
    vad = _vad(monkeypatch)
    assert vad(_chunk(0.5, True), 16000) is None
    assert vad(_chunk(0.5, True), 16000) is None
    turn = vad(_chunk(0.5, False), 16000, binary=True)
    assert turn is not None and len(turn) == 16000


def test_a_dropped_short_turn_is_forgotten(monkeypatch):
    vad = _vad(monkeypatch, min_speech_ms=500)
    assert vad(_chunk(0.3, True), 16000) is None
    assert vad(_chunk(0.3, False), 16000) is None  # 300 ms: dropped
    assert vad.vad_output is None
    vad(_chunk(1.0, True), 16000)
    turn = vad(_chunk(0.5, False), 16000, binary=True)
    # the next turn alone, not glued to the dropped one
    assert len(turn) == 16000


def test_a_turn_longer_than_max_is_dropped_at_once(monkeypatch):
    vad = _vad(monkeypatch, max_speech_ms=1000)
    for _ in range(10):  # 5 s of speech: never more than 1 s held
        assert vad(_chunk(0.5, True), 16000) is None
        assert vad.vad_output is None or len(vad.vad_output) <= 2
    vad(_chunk(0.5, False), 16000)
    vad(_chunk(0.5, True), 16000)
    assert vad(_chunk(0.5, False), 16000) is not None


def test_reset_forgets_the_turn_in_progress(monkeypatch):
    vad = _vad(monkeypatch)
    vad(_chunk(0.5, True), 16000)
    vad.reset()
    assert vad(_chunk(0.5, False), 16000) is None
