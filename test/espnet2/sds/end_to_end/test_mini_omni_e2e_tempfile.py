"""The wav handed to the client must not be left behind on disk.

`forward` and `warmup` write the user turn to a NamedTemporaryFile with
delete=False, because run_AT_batch_stream is a generator and does not open the
path until it is drained, so the file has to outlive the with block. It still
has to be removed once the generator is exhausted. sds1 runs as a long-lived
Gradio app, so one leaked wav per turn grows without bound.

These stub the client and the SNAC decoder so no checkpoint, GPU or ffmpeg is
needed, and count what is left in a temp directory of their own.
"""

import tempfile

import numpy as np
import pytest
import torch

from espnet2.sds.end_to_end import mini_omni_e2e as mod

SNAC_FRAME = 2048
N_FRAMES = 12
STREAM_STRIDE = 4


class _StubSnac:
    """Stands in for snac_24khz."""

    def decode(self, codes):
        """Return one frame of silence per coarse code."""
        return torch.zeros(1, 1, codes[0].shape[-1] * SNAC_FRAME)


class _StubClient:
    """Records every path it is asked to read, and reads it like the real one."""

    def __init__(self):
        """Build the stub with a fake SNAC decoder on CPU."""
        self.snacmodel = _StubSnac()
        self.device = torch.device("cpu")
        self.seen = []

    def run_AT_batch_stream(self, audio_path, stream_stride, max_tokens, **kwargs):
        """Read the wav, then yield chunks and return the token stream."""
        # the real client opens the path here, inside the generator body, which
        # is why the file cannot be removed when the with block closes
        with open(audio_path, "rb") as fh:
            fh.read()
        self.seen.append(audio_path)
        tokens = [[1] * (N_FRAMES + 7) for _ in range(8)]
        for _ in range(N_FRAMES // stream_stride):
            yield b"\x00\x00" * (stream_stride * SNAC_FRAME)
        yield "a text response"
        return tokens


class _RecordingSegment:
    """Stands in for pydub.AudioSegment so no encoder is needed."""

    def __init__(self, data, frame_rate=None, sample_width=None, channels=None):
        """Keep the bytes it was handed."""
        self._data = data

    def export(self, buf, **kwargs):
        """Write the kept bytes straight out."""
        buf.write(self._data)
        return buf


def _build_model():
    """Build the model with __new__ so no checkpoint is fetched."""
    model = mod.MiniOmniE2EModel.__new__(mod.MiniOmniE2EModel)
    model.client = _StubClient()
    model.stream_stride = STREAM_STRIDE
    model.max_tokens = 2048
    model.temperature = 0.9
    model.top_k = 1
    model.top_p = 1.0
    model.OUT_CHANNELS = 1
    model.OUT_RATE = 24000
    model.OUT_SAMPLE_WIDTH = 2
    model.device = "cpu"
    model.dtype = "float16"
    return model


@pytest.fixture
def isolated_tmp(monkeypatch, tmp_path):
    """Point tempfile at a directory of our own so the count is unambiguous."""
    monkeypatch.setattr(mod, "AudioSegment", _RecordingSegment, raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    yield tmp_path
    monkeypatch.setattr(tempfile, "tempdir", None)


def test_forward_removes_the_wav_it_writes(isolated_tmp):
    model = _build_model()
    for _ in range(3):
        model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    assert len(model.client.seen) == 3, "the client should have been called 3 times"
    left = list(isolated_tmp.iterdir())
    assert left == [], f"{len(left)} temp file(s) left behind: {[p.name for p in left]}"


def test_warmup_removes_the_wav_it_writes(isolated_tmp):
    model = _build_model()
    model.warmup()

    left = list(isolated_tmp.iterdir())
    assert left == [], f"{len(left)} temp file(s) left behind: {[p.name for p in left]}"


def test_wav_is_removed_even_when_generation_raises(isolated_tmp):
    """A failed turn must not leak either, which the old code could not do."""
    model = _build_model()

    class _Failing(_StubClient):
        def run_AT_batch_stream(self, audio_path, *a, **k):
            with open(audio_path, "rb") as fh:
                fh.read()
            raise RuntimeError("generation blew up")
            yield  # pragma: no cover - makes this a generator

    model.client = _Failing()
    with pytest.raises(RuntimeError):
        model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    left = list(isolated_tmp.iterdir())
    assert left == [], f"{len(left)} temp file(s) left behind: {[p.name for p in left]}"


def test_nothing_is_left_when_writing_the_wav_fails(isolated_tmp, monkeypatch):
    """A write failure must not leave a partial file either.

    NamedTemporaryFile creates the file before write() is called, so the earlier
    shape of this code could leave a partial wav behind if the write raised.
    A TemporaryDirectory covers that, since the directory is what gets removed.
    """
    model = _build_model()
    real_open = open

    def _failing_open(path, mode="r", *a, **k):
        if str(path).endswith("turn.wav") and "w" in mode:
            fh = real_open(path, mode, *a, **k)
            fh.close()

            class _Boom:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def write(self, _):
                    raise OSError("no space left on device")

            return _Boom()
        return real_open(path, mode, *a, **k)

    monkeypatch.setitem(mod.__builtins__, "open", _failing_open)
    with pytest.raises(OSError):
        model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    left = list(isolated_tmp.iterdir())
    assert left == [], f"{len(left)} left behind: {[p.name for p in left]}"
