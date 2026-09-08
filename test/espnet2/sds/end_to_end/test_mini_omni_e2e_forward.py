"""CPU-only tests for how MiniOmniE2EModel.forward assembles its audio.

These deliberately avoid the real checkpoint, a GPU and ffmpeg so that they run
in the normal unit test job: the OmniInference client and the SNAC decoder are
stubbed, and pydub's AudioSegment is replaced by a recorder. That keeps the
assertions on the part this module actually owns, which is how the streamed
output is turned into the bytes forward() returns.
"""

import numpy as np
import pytest
import torch

from espnet2.sds.end_to_end import mini_omni_e2e as mod

# 24 kHz samples that snac_24khz reconstructs per coarse code.
SNAC_FRAME = 2048
# Number of coarse frames the stubbed client produces. Must be a multiple of
# forward()'s stream_stride (4) so the stub can yield whole chunks.
N_FRAMES = 12
STREAM_STRIDE = 4


class _StubSnac:
    """Returns one frame of silence per coarse code, like snac_24khz."""

    def decode(self, codes):
        return torch.zeros(1, 1, codes[0].shape[-1] * SNAC_FRAME)


class _StubClient:
    """Mimics OmniInference.run_AT_batch_stream.

    It yields per-chunk audio, then the text response, and returns the complete
    8-layer token stream, which is what the real generator does.
    """

    def __init__(self):
        self.snacmodel = _StubSnac()
        self.device = torch.device("cpu")

    def run_AT_batch_stream(self, audio_path, stream_stride, max_tokens, **kwargs):
        # kwargs absorbs the generation settings forward() passes through. What
        # reaches the client is asserted in test_mini_omni_e2e_sampling.py; this
        # file is only about how the streamed output is assembled.
        # reconscruct_snac drops the text layer and trims layer i by i + 1, so a
        # stream of N_FRAMES + 7 steps leaves N_FRAMES coarse frames.
        tokens = [[1] * (N_FRAMES + 7) for _ in range(8)]
        for _ in range(N_FRAMES // stream_stride):
            yield b"\x00\x00" * (stream_stride * SNAC_FRAME)
        yield "a text response"
        return tokens


class _RecordingSegment:
    """Stands in for pydub.AudioSegment so no encoder is needed."""

    calls = []

    def __init__(self, data, frame_rate=None, sample_width=None, channels=None):
        type(self).calls.append(data)
        self._data = data

    def export(self, buf, **kwargs):
        buf.write(self._data)
        return buf


def _build_model():
    """Build a MiniOmniE2EModel, skipping __init__ so no weights are fetched."""
    model = mod.MiniOmniE2EModel.__new__(mod.MiniOmniE2EModel)
    model.client = _StubClient()
    model.stream_stride = STREAM_STRIDE
    model.max_tokens = 2048
    # forward() reads these, so skipping __init__ means setting them here. The
    # values are the shipped defaults.
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
def recorder(monkeypatch):
    """Replace pydub's AudioSegment and record what it is handed."""
    monkeypatch.setattr(mod, "AudioSegment", _RecordingSegment, raising=False)
    _RecordingSegment.calls = []
    yield _RecordingSegment
    _RecordingSegment.calls = []


def test_forward_encodes_the_utterance_as_one_segment(recorder):
    """The whole utterance must be encoded once, not once per streamed chunk.

    Encoding each chunk separately produced one encoder delay and one block of
    padding per chunk boundary, which showed up as inserted silence.
    """
    model = _build_model()
    speech = np.zeros(1600, dtype=np.int16)

    text_str, audio_output = model.forward(speech, orig_sr=16000)

    assert text_str == "a text response"
    assert isinstance(audio_output, bytes)

    # calls[0] is the input wav forward() builds; the rest are output segments.
    assert len(recorder.calls) == 2, (
        "expected exactly one output AudioSegment, got " f"{len(recorder.calls) - 1}"
    )


def test_forward_keeps_every_generated_frame(recorder):
    """No frames may be dropped.

    The streamed path only emits once its counter reaches stream_stride, so a
    trailing partial group never reached the caller.
    """
    model = _build_model()
    model.client = _StubClient()
    speech = np.zeros(1600, dtype=np.int16)

    _, audio_output = model.forward(speech, orig_sr=16000)

    expected = N_FRAMES * SNAC_FRAME * model.OUT_SAMPLE_WIDTH
    assert len(recorder.calls[-1]) == expected, (
        f"expected {N_FRAMES} frames ({expected} bytes) of audio, got "
        f"{len(recorder.calls[-1])} bytes"
    )
    assert len(audio_output) == expected


def test_forward_uses_the_client_device_for_decoding(recorder):
    """The SNAC decode must run on the device the client was built with."""
    model = _build_model()
    seen = {}

    class _DeviceCheckingSnac(_StubSnac):
        def decode(self, codes):
            seen["device"] = codes[0].device
            return super().decode(codes)

    model.client.snacmodel = _DeviceCheckingSnac()
    model.forward(np.zeros(1600, dtype=np.int16), orig_sr=16000)

    assert seen["device"] == torch.device("cpu")
