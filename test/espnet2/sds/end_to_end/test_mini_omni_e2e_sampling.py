"""CPU-only tests for the generation settings of MiniOmniE2EModel.

The OmniInference client and the SNAC decoder are stubbed and pydub's
AudioSegment is replaced, so these need no checkpoint, no GPU and no ffmpeg.
The assertions are on what reaches run_AT_batch_stream, which is the part this
module owns.
"""

import numpy as np
import pytest
import torch

from espnet2.sds.end_to_end import mini_omni_e2e as mod

SNAC_FRAME = 2048
N_FRAMES = 8


class _StubSnac:
    def decode(self, codes):
        return torch.zeros(1, 1, codes[0].shape[-1] * SNAC_FRAME)


class _RecordingClient:
    """Records the arguments run_AT_batch_stream is called with."""

    def __init__(self):
        self.snacmodel = _StubSnac()
        self.device = torch.device("cpu")
        self.calls = []

    def run_AT_batch_stream(
        self,
        audio_path,
        stream_stride=4,
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        top_p=1.0,
    ):
        self.calls.append(
            dict(
                stream_stride=stream_stride,
                max_returned_tokens=max_returned_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )
        tokens = [[1] * (N_FRAMES + 7) for _ in range(8)]
        for _ in range(N_FRAMES // stream_stride):
            yield b"\x00\x00" * (stream_stride * SNAC_FRAME)
        yield "a text response"
        return tokens


class _StubSegment:
    def __init__(self, data, **kwargs):
        self._data = data

    def export(self, buf, **kwargs):
        buf.write(self._data)
        return buf


def _build_model(**kwargs):
    """Build a MiniOmniE2EModel, skipping __init__ so no weights are fetched."""
    model = mod.MiniOmniE2EModel.__new__(mod.MiniOmniE2EModel)
    model.client = _RecordingClient()
    model.stream_stride = kwargs.get("stream_stride", 4)
    model.max_tokens = kwargs.get("max_tokens", 2048)
    model.temperature = kwargs.get("temperature", 0.9)
    model.top_k = kwargs.get("top_k", 1)
    model.top_p = kwargs.get("top_p", 1.0)
    model.OUT_CHANNELS = 1
    model.OUT_RATE = 24000
    model.OUT_SAMPLE_WIDTH = 2
    model.device = "cpu"
    model.dtype = "float16"
    return model


@pytest.fixture
def stub_segment(monkeypatch):
    """Replace pydub's AudioSegment so no encoder is needed."""
    monkeypatch.setattr(mod, "AudioSegment", _StubSegment, raising=False)


def test_defaults_are_unchanged(stub_segment):
    """The shipped behaviour must not move: greedy decoding, stride 4."""
    model = _build_model()
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    call = model.client.calls[-1]
    assert call["stream_stride"] == 4
    assert call["max_returned_tokens"] == 2048
    assert call["temperature"] == 0.9
    assert call["top_k"] == 1
    assert call["top_p"] == 1.0


def test_constructor_settings_are_honoured(stub_segment):
    """stream_stride and max_tokens were stored but never read before."""
    model = _build_model(stream_stride=8, max_tokens=512, temperature=1.1, top_k=20)
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    call = model.client.calls[-1]
    assert call["stream_stride"] == 8
    assert call["max_returned_tokens"] == 512
    assert call["temperature"] == 1.1
    assert call["top_k"] == 20


def test_per_call_override_reaches_the_client(stub_segment):
    """Sampling several responses for one input needs a per-call override."""
    model = _build_model()
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000, top_k=20)
    model.forward(
        np.zeros(1600, dtype=np.int16), orig_sr=16000, temperature=1.2, top_p=0.9
    )

    first, second = model.client.calls
    assert first["top_k"] == 20
    assert first["temperature"] == 0.9, "unset overrides must fall back"
    assert second["temperature"] == 1.2
    assert second["top_p"] == 0.9
    assert second["top_k"] == 1, "unset overrides must fall back"


def test_top_k_none_disables_filtering_from_the_constructor(stub_segment):
    """None reaches the sampler, which reads it as "no top-k filtering".

    forward() cannot express this, because there None means "keep the
    constructor value", so the constructor is the place to disable it.
    """
    model = _build_model(top_k=None)
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    assert model.client.calls[-1]["top_k"] is None


def test_override_does_not_persist(stub_segment):
    """An override applies to one call only."""
    model = _build_model()
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000, top_k=20)
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    assert model.client.calls[0]["top_k"] == 20
    assert model.client.calls[1]["top_k"] == 1
    assert model.top_k == 1
