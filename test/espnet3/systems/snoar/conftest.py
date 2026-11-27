"""Shared fixtures: a tiny TF-GridNet config and a few short wav files."""

import numpy as np
import pytest
import soundfile as sf

FS = 16000


def tiny_net(n_srcs):
    """Return an ESPnet2 TF-GridNetV2 config small enough for CPU tests."""
    return {
        "separator": "tfgridnetv2",
        "separator_conf": {
            "n_srcs": n_srcs,
            "n_fft": 64,
            "stride": 32,
            "n_layers": 1,
            "emb_dim": 4,
            "lstm_hidden_units": 8,
            "attn_n_head": 1,
        },
    }


@pytest.fixture
def wav_dir(tmp_path):
    """Write three 1.5 s recordings with speech-like bursts and 0.6 s gaps."""
    rng = np.random.default_rng(0)
    d = tmp_path / "wav"
    d.mkdir()
    t = np.arange(int(1.5 * FS)) / FS
    for i in range(3):
        env = ((t > 0.6) & (t < 1.0)).astype(np.float32)
        x = 0.3 * env * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(len(t))
        sf.write(d / f"rec{i}.wav", x.astype(np.float32), FS)
    return d
