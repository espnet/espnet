"""Check that ASR and ST streaming inference share one apply_frontend behaviour."""

from types import SimpleNamespace

import pytest
import torch

from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.bin.asr_inference_streaming import (
    Speech2TextStreaming as ASRSpeech2TextStreaming,
)
from espnet2.bin.st_inference_streaming import (
    Speech2TextStreaming as STSpeech2TextStreaming,
)


def _make_streamer(cls, model_attr, frontend, n_fft, win_length, hop_length):
    """Build a streamer that only has the attributes apply_frontend reads."""

    def extract_feats(speech, speech_lengths):
        return frontend(speech, speech_lengths)

    streamer = object.__new__(cls)
    streamer.n_fft = n_fft
    streamer.win_length = win_length
    streamer.hop_length = hop_length
    streamer.dtype = "float32"
    streamer.device = "cpu"
    model = SimpleNamespace(_extract_feats=extract_feats, normalize=None)
    setattr(streamer, model_attr, model)
    return streamer


def _chunk_sizes(total, chunk):
    """Split total samples into chunk sizes; chunk=None gives irregular sizes."""
    if chunk is not None:
        return [min(chunk, total - start) for start in range(0, total, chunk)]
    generator = torch.Generator().manual_seed(0)
    sizes, remaining = [], total
    while remaining > 0:
        size = min(int(torch.randint(1, 700, (1,), generator=generator)), remaining)
        sizes.append(size)
        remaining -= size
    return sizes


@pytest.mark.parametrize("total", [300, 450, 4000, 4050])
@pytest.mark.parametrize("chunk", [1, 100, 255, 401, 479, 1600, None])
@pytest.mark.parametrize(
    "n_fft,win_length,hop_length", [(512, 512, 128), (512, 400, 160), (256, 256, 128)]
)
def test_asr_and_st_apply_frontend_agree(n_fft, win_length, hop_length, chunk, total):
    frontend = DefaultFrontend(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    frontend.eval()
    asr = _make_streamer(
        ASRSpeech2TextStreaming, "asr_model", frontend, n_fft, win_length, hop_length
    )
    st = _make_streamer(
        STSpeech2TextStreaming, "st_model", frontend, n_fft, win_length, hop_length
    )

    speech = torch.randn(total, generator=torch.Generator().manual_seed(1))
    asr_states, st_states, start = None, None, 0
    with torch.no_grad():
        for size in _chunk_sizes(total, chunk):
            piece = speech[start : start + size]
            start += size
            is_final = start >= total
            asr_feats, asr_lens, asr_states = asr.apply_frontend(
                piece, asr_states, is_final=is_final
            )
            st_feats, st_lens, st_states = st.apply_frontend(
                piece, st_states, is_final=is_final
            )

            assert (asr_feats is None) == (st_feats is None)
            if asr_feats is not None:
                assert torch.equal(asr_feats, st_feats)
                assert torch.equal(asr_lens, st_lens)
            assert (asr_states is None) == (st_states is None)
            if asr_states is not None:
                assert asr_states.keys() == st_states.keys()
                for key in asr_states:
                    if isinstance(asr_states[key], torch.Tensor):
                        assert torch.equal(asr_states[key], st_states[key])
                    else:
                        assert asr_states[key] == st_states[key]
