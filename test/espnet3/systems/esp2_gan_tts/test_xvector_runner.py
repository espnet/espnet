"""Tests for the x-vector runner used by compute_xvectors."""

from test.espnet3.systems.esp2_gan_tts._xvector_helpers import (
    make_provider,
    write_wav,
)

import numpy as np
import pytest
import soundfile as sf
import torch

from espnet3.systems.esp2_gan_tts.xvector_provider import XVectorProvider
from espnet3.systems.esp2_gan_tts.xvector_runner import XVectorRunner

# ===============================================================
# Test Case Summary
# ===============================================================
#
# extraction and persistence
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_forward_writes_embedding               | forward() saves <utt>.pt and |
# |                                             | reports status 'ok'.         |
# | test_forward_skips_existing                 | An existing .pt is skipped   |
# |                                             | without re-reading audio.    |
# | test_forward_accepts_iterable               | An iterable index yields a   |
# |                                             | list of status dicts.        |
# | test_forward_converts_embedding_types       | ndarray/tensor/list are all  |
# |                                             | stored as float32 tensors.   |
# | test_load_audio_mixes_stereo_to_mono        | Multi-channel input is mixed |
# |                                             | down to mono float32.        |
# | test_load_audio_keeps_mono_untouched        | Mono input keeps its shape,  |
# |                                             | dtype and native rate.       |
# | test_extract_embedding_dispatches_toolkit   | Each toolkit reaches its own |
# |                                             | extractor; unknown raises.   |
# | test_extract_speechbrain_uses_encode_batch  | The speechbrain path batches |
# |                          | the signal and returns a 1-D embedding.          |
# | test_extract_espnet_resamples_and_mixes     | The espnet path resamples    |
# |                                             | and mixes down to mono.      |
# | test_extract_rawnet_pads_short_audio        | RawNet pads short clips into |
# |                                             | ten 3-second segments.       |
# | test_extract_rawnet_resamples               | Non-16 kHz input is          |
# |                                             | resampled before segmenting. |


def test_forward_writes_embedding(manifest, tmp_path, stub_model):
    env = make_provider(manifest, tmp_path).build_env_local()

    result = XVectorRunner.forward(0, **env)

    assert result == {"utt_id": "u1", "status": "ok"}
    saved = torch.load(str(env["output_dir"] / "u1.pt"))
    assert saved.dtype == torch.float32
    assert saved.shape == (192,)


def test_forward_skips_existing(manifest, tmp_path, stub_model, monkeypatch):
    env = make_provider(manifest, tmp_path).build_env_local()
    XVectorRunner.forward(0, **env)

    def _fail(*args, **kwargs):
        raise AssertionError("audio must not be re-read for an existing .pt")

    monkeypatch.setattr(XVectorRunner, "_load_audio", staticmethod(_fail))

    assert XVectorRunner.forward(0, **env) == {"utt_id": "u1", "status": "skipped"}


def test_forward_accepts_iterable(manifest, tmp_path, stub_model):
    env = make_provider(manifest, tmp_path).build_env_local()

    results = XVectorRunner.forward(range(3), **env)

    assert [r["utt_id"] for r in results] == ["u1", "u2", "u3"]
    assert {r["status"] for r in results} == {"ok"}


@pytest.mark.parametrize(
    "embedding",
    [
        np.zeros(4, dtype=np.float64),
        torch.zeros(4, dtype=torch.float64),
        [0.0, 0.0, 0.0, 0.0],
    ],
)
def test_forward_converts_embedding_types(manifest, tmp_path, monkeypatch, embedding):
    monkeypatch.setattr(
        XVectorProvider, "_build_model", staticmethod(lambda *a, **k: "MODEL")
    )
    monkeypatch.setattr(
        XVectorRunner,
        "_extract_embedding",
        staticmethod(lambda wav, sr, model, toolkit, device: embedding),
    )
    env = make_provider(manifest, tmp_path).build_env_local()

    XVectorRunner.forward(0, **env)

    saved = torch.load(str(env["output_dir"] / "u1.pt"))
    assert saved.dtype == torch.float32
    assert saved.shape == (4,)


def test_load_audio_mixes_stereo_to_mono(tmp_path):
    """_load_audio owns the mono mix-down that librosa.load used to provide."""
    path = tmp_path / "stereo.wav"
    stereo = np.stack(
        [np.full(16, 1.0, dtype=np.float32), np.full(16, -0.5, dtype=np.float32)],
        axis=-1,
    )
    sf.write(path, stereo, 16000)

    wav, in_sr = XVectorRunner._load_audio(path)

    assert in_sr == 16000
    assert wav.ndim == 1
    assert wav.shape == (16,)
    assert wav.dtype == np.float32
    np.testing.assert_allclose(wav, 0.25, atol=1e-4)


def test_load_audio_keeps_mono_untouched(tmp_path):
    wav, in_sr = XVectorRunner._load_audio(write_wav(tmp_path, seconds=0.5, sr=8000))

    assert (wav.ndim, in_sr, wav.shape) == (1, 8000, (4000,))
    assert wav.dtype == np.float32


def test_extract_embedding_dispatches_toolkit(monkeypatch):
    seen = []
    for name in ("espnet", "speechbrain", "rawnet"):
        monkeypatch.setattr(
            XVectorRunner,
            f"_extract_{name}",
            staticmethod(lambda wav, sr, model, device, n=name: seen.append(n)),
        )

    wav = np.zeros(16, dtype=np.float32)
    for name in ("espnet", "speechbrain", "rawnet"):
        XVectorRunner._extract_embedding(wav, 16000, "MODEL", name, "cpu")
    assert seen == ["espnet", "speechbrain", "rawnet"]

    with pytest.raises(ValueError, match="Unknown toolkit: nope"):
        XVectorRunner._extract_embedding(wav, 16000, "MODEL", "nope", "cpu")


def test_extract_speechbrain_uses_encode_batch(fake_speechbrain):
    """speechbrain wants (batch, time) in and gives (batch, 1, emb) out.

    The runner must add the batch axis and flatten the result so the default
    toolkit persists the same 1-D shape as the espnet and rawnet paths.
    """

    class _Model:
        def __init__(self):
            self.seen_shapes = []

        def encode_batch(self, wav_tensor):
            self.seen_shapes.append(tuple(wav_tensor.shape))
            return torch.zeros(1, 1, 192)

    model = _Model()
    out = XVectorRunner._extract_speechbrain(
        np.zeros(16000, dtype=np.float32), 16000, model, "cpu"
    )

    assert model.seen_shapes == [(1, 16000)]
    assert out.shape == (192,)


def test_extract_espnet_resamples_and_mixes():
    seen = {}

    def _model(wav_tensor):
        seen["shape"] = tuple(wav_tensor.shape)
        return torch.zeros(192)

    # 8 kHz stereo input must arrive as mono at the espnet default of 16 kHz.
    out = XVectorRunner._extract_espnet(
        np.zeros((2, 8000), dtype=np.float32), 8000, _model, "cpu"
    )

    assert len(seen["shape"]) == 1
    assert seen["shape"][0] == 16000
    assert out.shape == (192,)


def test_extract_rawnet_pads_short_audio():
    seen = {}

    def _model(audios):
        seen["shape"] = tuple(audios.shape)
        return torch.zeros(10, 256)

    # A 0.5 s clip is shorter than RawNet3's 3 s window and must be padded.
    out = XVectorRunner._extract_rawnet(
        np.zeros(8000, dtype=np.float32), 16000, _model, "cpu"
    )

    assert seen["shape"] == (10, 48000)
    assert out.shape == (256,)


def test_extract_rawnet_resamples():
    seen = {}

    def _model(audios):
        seen["shape"] = tuple(audios.shape)
        return torch.zeros(10, 256)

    # 8 kHz in must be resampled to 16 kHz before the 3 s windows are cut.
    out = XVectorRunner._extract_rawnet(
        np.zeros(80000, dtype=np.float32), 8000, _model, "cpu"
    )

    assert seen["shape"] == (10, 48000)
    assert out.shape == (256,)
