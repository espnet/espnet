import random
from argparse import ArgumentParser

import numpy as np
import pytest
import soundfile as sf
import torch

from espnet2.tasks.rst import (
    RestorationTask,
    SidonCollateFn,
    _band_limit,
    _clip,
    _noise,
    _packet_loss,
    _reverb,
    degrade_waveform,
)
from espnet2.tasks.rst_vocoder import RestorationVocoderTask, SidonVocoderCollateFn


@pytest.fixture
def pools(tmp_path):
    rng = np.random.RandomState(0)
    noise_dir, rir_dir = tmp_path / "noise", tmp_path / "rir"
    noise_dir.mkdir()
    rir_dir.mkdir()
    for i in range(2):
        sf.write(
            noise_dir / f"n{i}.wav", rng.randn(16000).astype(np.float32) * 0.1, 16000
        )
        # decaying random tail well below the direct-path tap, written as float
        # so nothing is clipped to the tap's magnitude (a tie would make the
        # peak-based trimming depend on argmax tie-breaking)
        rir = rng.randn(3200).astype(np.float32) * 0.1 * np.exp(-np.arange(3200) / 400)
        rir[:200] = 0.0  # leading propagation delay that _reverb must trim
        rir[200] = 1.0
        sf.write(rir_dir / f"r{i}.wav", rir, 16000, subtype="FLOAT")
    return str(noise_dir), str(rir_dir)


def test_parsers():
    assert isinstance(RestorationTask.get_parser(), ArgumentParser)
    assert isinstance(RestorationVocoderTask.get_parser(), ArgumentParser)


def test_single_degradations(pools):
    noise_dir, rir_dir = pools
    from espnet2.tasks.rst import _audio_files

    noise_files, rir_files = _audio_files(noise_dir), _audio_files(rir_dir)
    assert len(noise_files) == 2 and len(rir_files) == 2
    wav = torch.randn(16000) * 0.1
    for name, fn in (
        ("reverb", lambda x: _reverb(x, 16000, rir_files)),
        ("noise", lambda x: _noise(x, 16000, noise_files)),
        ("band_limit", lambda x: _band_limit(x, 16000)),
        ("clip", _clip),
        ("packet_loss", lambda x: _packet_loss(x, 16000)),
    ):
        changed = False
        for seed in range(8):  # band_limit is a no-op when it draws the native rate
            random.seed(seed)
            torch.manual_seed(seed)
            out = fn(wav.clone())
            assert torch.isfinite(out).all(), name
            assert out.numel() >= wav.numel() - 1, name
            changed = changed or not torch.equal(out[: wav.numel()], wav)
        assert changed, name


def test_reverb_keeps_alignment(pools):
    _, rir_dir = pools
    from espnet2.tasks.rst import _audio_files

    impulse = torch.zeros(16000)
    impulse[1000] = 1.0
    out = _reverb(impulse, 16000, _audio_files(rir_dir))
    # the RIR's leading delay is removed, so the direct path stays at 1000
    assert int(torch.argmax(out.abs())) == 1000


def test_degrade_waveform_probability(pools):
    noise_dir, rir_dir = pools
    from espnet2.tasks.rst import _audio_files

    wav = torch.randn(8000) * 0.1
    same = degrade_waveform(
        wav, 16000, _audio_files(noise_dir), _audio_files(rir_dir), 0.0
    )
    torch.testing.assert_close(same, wav)
    torch.manual_seed(0)
    changed = degrade_waveform(
        wav, 16000, _audio_files(noise_dir), _audio_files(rir_dir), 1.0
    )
    assert torch.isfinite(changed).all()
    assert changed.abs().max() <= 1.0 + 1e-6


def test_collate_fn_is_deterministic_in_validation(pools):
    noise_dir, rir_dir = pools
    collate = SidonCollateFn(
        max_samples=8000,
        input_sr=16000,
        noise_dir=noise_dir,
        rir_dir=rir_dir,
        degrade_prob=1.0,
        online_degradation=True,
        train=False,
    )
    rng = np.random.RandomState(1)
    data = [
        ("utt1", {"speech_ref1": rng.randn(12000).astype(np.float32) * 0.1}),
        ("utt2", {"speech_ref1": rng.randn(5000).astype(np.float32) * 0.1}),
    ]
    keys, batch = collate(data)
    assert list(keys) == ["utt1", "utt2"]
    for name in (
        "speech_ref1",
        "noisy_speech",
        "speech_ref1_lengths",
        "noisy_speech_lengths",
    ):
        assert name in batch
    assert batch["speech_ref1"].shape[1] <= 8000
    assert int(batch["speech_ref1_lengths"][1]) == 5000
    torch.manual_seed(123)  # the validation crop and degradation ignore the global RNG
    _, again = collate(data)
    torch.testing.assert_close(again["noisy_speech"], batch["noisy_speech"])
    torch.testing.assert_close(again["speech_ref1"], batch["speech_ref1"])


def test_vocoder_collate_fn_crop_start():
    collate = SidonVocoderCollateFn(
        context_samples=48000,
        segment_frames=5,
        hop=960,
        input_sr=16000,
        output_sr=48000,
        use_predicted_feat=False,
        noise_dir="",
        rir_dir="",
        degrade_prob=0.0,
        online_degradation=False,
        train=True,
    )
    data = [
        ("u", {"speech_ref1": np.random.RandomState(0).randn(96000).astype(np.float32)})
    ]
    keys, batch = collate(data)
    assert list(keys) == ["u"]
    assert batch["speech_ref1"].shape[1] <= 48000
    frames = batch["speech_ref1"].shape[1] // 960
    assert 0 <= int(batch["vocoder_crop_start"][0]) <= max(0, frames - 5)
