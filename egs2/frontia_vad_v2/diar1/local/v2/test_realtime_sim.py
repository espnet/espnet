"""Tests of local/v2/realtime_sim.py with toy systems of known look-ahead."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import realtime_sim as rs  # noqa: E402

FS = rs.FS


class ToyStream(rs.StreamingSystem):
    """Frames of 160 samples; frame i is the sum of the samples up to its end + look-ahead
    and is returned as soon as those samples are in (early=True: one frame before its
    samples are in, a system that dates a decision too early)."""

    shift, offset = 160 / FS, 80 / FS

    def __init__(self, look_ahead=0, early=False):
        self.look_ahead, self.early = look_ahead, early

    def reset(self):
        self.audio = np.zeros(0)
        self.done = 0

    def accept(self, samples):
        self.audio = np.concatenate([self.audio, samples])
        out = []
        while True:
            need = (self.done + (0 if self.early else 1)) * 160 + self.look_ahead
            if need > len(self.audio):
                break
            out.append(self.audio[: (self.done + 1) * 160 + self.look_ahead].sum())
            self.done += 1
        return np.asarray(out)


def offline(wav, look_ahead):
    n = len(wav) // 160
    return np.array([wav[: (i + 1) * 160 + look_ahead].sum() for i in range(n)])


@pytest.mark.parametrize("look_ahead", [0, 1, 320, 1601])
def test_latency_equals_look_ahead(look_ahead):
    wav = np.random.default_rng(0).standard_normal(FS * 2)
    probs, emit = rs.simulate(ToyStream(look_ahead), wav, block=1)
    lat = rs.latencies(emit, ToyStream.shift, ToyStream.offset)
    assert np.isclose(lat.max(), look_ahead / FS)
    n = len(probs)
    assert np.allclose(probs, offline(wav, look_ahead)[:n], rtol=1e-5, atol=1e-4)


def test_latency_at_most_block():
    wav = np.random.default_rng(1).standard_normal(FS * 2)
    block = 320
    probs, emit = rs.simulate(ToyStream(0), wav, block=block)
    lat = rs.latencies(emit, ToyStream.shift, ToyStream.offset)
    assert lat.min() >= -1e-12 and lat.max() <= block / FS + 1e-12
    assert np.allclose(probs, offline(wav, 0)[: len(probs)], rtol=1e-5, atol=1e-4)


def test_a_frame_returned_before_its_audio_is_refused():
    wav = np.random.default_rng(2).standard_normal(FS)
    with pytest.raises(rs.CausalityError):
        rs.simulate(ToyStream(0, early=True), wav, block=100)


class ToyOffline:
    """Offline model of 160-sample frames that reads `look_ahead` samples past each frame
    (zero padding at the end) and `context` samples before it."""

    def __init__(self, look_ahead, context=480):
        self.look_ahead, self.context = look_ahead, context

    def __call__(self, x):
        out = []
        for w in x:
            p = np.r_[w, np.zeros(self.look_ahead)]
            n = len(w) // 160
            out.append(
                np.array(
                    [
                        p[
                            max(0, i * 160 - self.context) : (i + 1) * 160
                            + self.look_ahead
                        ].sum()
                        for i in range(n)
                    ]
                )
            )
        return out


@pytest.mark.parametrize("look_ahead", [0, 800])
def test_windowed_is_causal_and_batched_equals_sequential(look_ahead):
    wav = np.random.default_rng(3).standard_normal(FS * 3 + 77)
    sys_ = rs.WindowedOffline(ToyOffline(look_ahead), 160, 160 / FS, 80 / FS, FS)
    seq, emit_seq = rs.simulate(sys_, wav, block=1600)
    bat, emit_bat = rs.windowed_probs(sys_, wav, block=1600, batch=4)
    assert np.allclose(seq, bat, rtol=1e-5, atol=1e-4) and np.allclose(
        emit_seq, emit_bat
    )
    lat = rs.latencies(emit_seq, 160 / FS, 80 / FS)
    assert lat.min() >= -1e-12 and lat.max() <= 0.1 + 1e-12
    full = ToyOffline(look_ahead)([wav])[0][: len(seq)]
    if look_ahead == 0:
        # a causal model with a receptive field inside the window gives its offline output
        assert np.allclose(seq, full, rtol=1e-5, atol=1e-4)
    else:
        # a model that reads ahead sees zeros instead of the audio not yet arrived
        assert not np.allclose(seq, full, rtol=1e-5, atol=1e-4)


def test_window_shorter_than_block_is_refused():
    wav = np.random.default_rng(4).standard_normal(FS)
    sys_ = rs.WindowedOffline(ToyOffline(0), 160, 160 / FS, 80 / FS, 320)
    with pytest.raises(ValueError):
        rs.simulate(sys_, wav, block=1600)
