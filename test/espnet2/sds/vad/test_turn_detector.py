"""TurnDetector over a stand-in frame VAD whose scores depend only on time.

Because the stand-in scores a frame by its start time, the same speech must
give the same turns at any input rate, chunk length, frame shift, median and
look-ahead; these tests check that, and that the memory held is released.
"""

import numpy as np
import pytest

from espnet2.sds.vad.abs_frame_vad import AbsFrameVAD
from espnet2.sds.vad.turn_detector import TurnDetector

# speech in [1.0, 2.0) s and [3.5, 4.2) s; the 1.5 s pause ends the first turn
SPEECH = [(1.0, 2.0), (3.5, 4.2)]
LENGTH_S = 8.0  # long enough for a turn to return with 1 s look-ahead


class _TimeFrameVAD(AbsFrameVAD):
    """Frame i starts at i * shift and is ready window_s later.

    It scores 0.9 inside the speech intervals, repeated every period seconds,
    and 0.1 elsewhere. window_s above the frame shift stands for a model that
    looks ahead.
    """

    def __init__(self, shift=0.008, window_s=None, speech=SPEECH, period=LENGTH_S):
        self.shift = shift
        self.window = shift if window_s is None else window_s
        self.speech = speech
        self.period = period
        self.reset()

    def _is_speech(self, t):
        t = t % self.period
        return any(a <= t < b for a, b in self.speech)

    @property
    def frame_shift(self):
        return self.shift

    @property
    def window_s(self):
        return self.window

    def reset(self):
        self.seconds = 0.0
        self.emitted = 0

    def accept(self, audio, sample_rate):
        self.seconds += len(audio) / sample_rate
        ready = int(np.floor((self.seconds - self.window) / self.shift + 1e-9)) + 1
        ready = max(ready, 0)
        out = [
            0.9 if self._is_speech(i * self.shift) else 0.1
            for i in range(self.emitted, ready)
        ]
        self.emitted = max(self.emitted, ready)
        return np.asarray(out, dtype=np.float32)


def _first_frame_s(t, shift):
    """The start of the first frame that starts at or after t."""
    i = 0
    while i * shift < t:
        i += 1
    return i * shift


def _signal(rate):
    """int16 ramp, so that a slice shows where it came from."""
    n = int(LENGTH_S * rate)
    return (np.arange(n) % 30000).astype(np.int16)


def _chunks(n, rate, chunk):
    if chunk == "random":
        rng = np.random.default_rng(rate)
        i = 0
        while i < n:
            step = int(rng.integers(1, int(0.3 * rate)))
            yield i, i + step
            i += step
    else:
        step = max(1, int(round(chunk * rate)))
        for i in range(0, n, step):
            yield i, i + step


def _run(det, rate, chunk, binary=True):
    x = _signal(rate)
    turns = []
    for a, b in _chunks(len(x), rate, chunk):
        out = det.forward(x[a:b], rate, binary=binary)
        if out is not None:
            turns.append((out, dict(det.last_turn), b / rate))
    return x, turns


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("chunk", [0.01, 0.02, 0.1, 0.5, "random"])
def test_same_turns_at_any_rate_and_chunk_length(rate, chunk):
    det = TurnDetector(_TimeFrameVAD(), threshold=0.5, median=25)
    x, turns = _run(det, rate, chunk)
    assert [(t["speech_start_s"], t["endpoint_s"]) for _, t, _ in turns] == [
        pytest.approx((_first_frame_s(a, 0.008), _first_frame_s(b, 0.008)))
        for a, b in SPEECH
    ]
    for out, t, _ in turns:
        # binary=True returns the input samples of [start_s, end_s) exactly
        a, b = int(round(t["start_s"] * rate)), int(round(t["end_s"] * rate))
        np.testing.assert_array_equal(out, x[a:b])
        assert t["end_s"] - t["endpoint_s"] == pytest.approx(0.3, abs=1 / rate)


@pytest.mark.parametrize("rate", [8000, 44100])
def test_float_output_is_at_target_sr(rate):
    det = TurnDetector(_TimeFrameVAD(), target_sr=16000)
    _, turns = _run(det, rate, 0.02, binary=False)
    for out, t, _ in turns:
        assert out.dtype == np.float32
        assert len(out) == pytest.approx((t["end_s"] - t["start_s"]) * 16000, abs=2)


@pytest.mark.parametrize("median", [1, 3, 11, 25])
@pytest.mark.parametrize("shift", [0.008, 0.02, 0.032])
@pytest.mark.parametrize("look_ahead_s", [0.0, 0.1, 1.0])
def test_turn_returns_endpoint_delay_s_after_its_endpoint(median, shift, look_ahead_s):
    """Fed 1 ms at a time, each turn comes back endpoint_delay_s after it.

    The median, the frame shift and a frame VAD's look-ahead (window_s) all
    move the return time; endpoint_delay_s must account for each.
    """
    det = TurnDetector(
        _TimeFrameVAD(shift, window_s=shift + look_ahead_s), median=median
    )
    _, turns = _run(det, 8000, 0.001)
    assert len(turns) == 2
    for _, t, returned_s in turns:
        late = returned_s - t["endpoint_s"] - det.endpoint_delay_s
        assert -1e-9 <= late < 0.001 + 1e-9  # within the 1 ms chunk
    # a clean step survives every median: the times do not move
    assert [(t["speech_start_s"], t["endpoint_s"]) for _, t, _ in turns] == [
        pytest.approx((_first_frame_s(a, shift), _first_frame_s(b, shift)))
        for a, b in SPEECH
    ]


def test_a_frame_vad_with_look_ahead_delays_but_does_not_change_turns():
    results = []
    for look_ahead_s in (0.0, 0.5, 1.0):
        det = TurnDetector(_TimeFrameVAD(window_s=0.008 + look_ahead_s), median=25)
        _, turns = _run(det, 16000, 0.02)
        results.append([(t["speech_start_s"], t["endpoint_s"]) for _, t, _ in turns])
    assert results[0] == results[1] == results[2]


def _held_bytes(det):
    """Bytes of the audio and frame buffers, not counting last_turn."""
    raw = 0 if det.raw is None else det.raw.nbytes
    return raw + det.audio.nbytes + det.probs.nbytes


@pytest.mark.parametrize("rate", [16000, 48000])
def test_memory_is_released_after_a_turn(rate):
    """After a 20 s turn returns, the buffers fall back to what silence holds."""
    # 3 s of silence, 20 s of speech, then silence
    frame_vad = _TimeFrameVAD(speech=[(3.0, 23.0)], period=1e9)
    det = TurnDetector(frame_vad, max_turn_ms=60000)
    chunk = np.zeros(int(0.1 * rate), dtype=np.int16)
    for _ in range(30):
        det.forward(chunk, rate)
    idle = _held_bytes(det)
    peak, returned = 0, False
    for _ in range(240):
        returned = det.forward(chunk, rate) is not None
        peak = max(peak, _held_bytes(det))
        if returned:
            break
    assert returned
    for _ in range(30):  # 3 s more silence
        det.forward(chunk, rate)
    assert peak > 20 * idle  # the turn's audio was held while it lasted
    assert _held_bytes(det) <= idle + chunk.nbytes * 3


def test_reset_releases_everything():
    det = TurnDetector(_TimeFrameVAD())
    fresh = _held_bytes(det)
    _run(det, 16000, 0.1)
    det.forward(np.ones(16000 * 2, dtype=np.int16), 16000)
    det.reset()
    assert _held_bytes(det) == fresh
    assert det.frame_vad.seconds == 0 and det.frame_vad.emitted == 0


@pytest.mark.parametrize("rate", [8000, 44100])
def test_memory_bounded_over_ten_minutes_at_any_rate(rate):
    from test.memory_growth import assert_memory_bounded

    det = TurnDetector(_TimeFrameVAD(), median=25)
    # two turns every 8 s, by the stand-in's period
    chunk = np.zeros(int(0.1 * rate), dtype=np.int16)

    def run():
        det.forward(chunk, rate)

    assert_memory_bounded(det, run, steps=6000)
