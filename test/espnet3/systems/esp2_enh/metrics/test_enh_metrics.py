import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import espnet3.systems.esp2_enh.metrics.pesq as pesq_module
import espnet3.systems.esp2_enh.metrics.stoi as stoi_module
from espnet3.systems.esp2_enh.metrics import PESQ, SISNR, STOI
from espnet3.systems.esp2_enh.metrics.sisnr import load_audio, si_snr

# A zero-mean reference and a zero-mean noise orthogonal to it, with equal
# energy per sample. Adding the noise at amplitude `a` gives an SI-SNR of
# exactly -20 * log10(a) dB, which is where the expected values below come from.
REF = np.tile([0.5, -0.5, 0.5, -0.5], 4000).astype(np.float32)
NOISE = np.tile([0.5, 0.5, -0.5, -0.5], 4000).astype(np.float32)


def _speech_like(seconds: float = 3.0, fs: int = 16000) -> np.ndarray:
    """Amplitude-modulated harmonics, enough for PESQ to find utterances."""
    t = np.arange(int(seconds * fs)) / fs
    voiced = sum(
        np.sin(2 * np.pi * f0 * k * t) / k for f0 in (140,) for k in range(1, 8)
    )
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    return (0.2 * voiced * envelope).astype(np.float32)


def _write_pairs(tmp_path: Path, pairs, fs: int = 16000) -> dict[str, Path]:
    ref_lines, hyp_lines = [], []
    for i, (ref, hyp) in enumerate(pairs):
        ref_path, hyp_path = tmp_path / f"ref{i}.wav", tmp_path / f"hyp{i}.wav"
        sf.write(ref_path, ref, fs, subtype="FLOAT")
        sf.write(hyp_path, hyp, fs, subtype="FLOAT")
        ref_lines.append(f"utt{i} {ref_path}")
        hyp_lines.append(f"utt{i} {hyp_path}")
    (tmp_path / "ref.scp").write_text("\n".join(ref_lines), encoding="utf-8")
    (tmp_path / "hyp.scp").write_text("\n".join(hyp_lines), encoding="utf-8")
    return {"reference": tmp_path / "ref.scp", "enhanced": tmp_path / "hyp.scp"}


def test_si_snr_matches_worked_examples():
    assert si_snr(REF, REF + NOISE) == pytest.approx(0.0, abs=1e-4)
    assert si_snr(REF, REF + 0.1 * NOISE) == pytest.approx(20.0, abs=1e-4)
    # Scale invariant: rescaling the estimate does not change the score.
    assert si_snr(REF, 3.0 * (REF + 0.1 * NOISE)) == pytest.approx(20.0, abs=1e-4)
    # The longer signal is cut to the shorter one before scoring.
    padded = np.concatenate([REF + 0.1 * NOISE, np.ones(100, dtype=np.float32)])
    assert si_snr(REF, padded) == pytest.approx(20.0, abs=1e-4)


def test_sisnr_metric_averages_over_utterances(tmp_path):
    data = _write_pairs(tmp_path, [(REF, REF + NOISE), (REF, REF + 0.1 * NOISE)])

    assert SISNR()(data, "test", tmp_path) == {"SI-SNR": pytest.approx(10.0, abs=1e-3)}


def test_sisnr_metric_uses_configured_keys(tmp_path):
    data = _write_pairs(tmp_path, [(REF, REF + 0.1 * NOISE)])
    data = {"clean": data["reference"], "denoised": data["enhanced"]}

    result = SISNR(ref_key="clean", hyp_key="denoised")(data, "test", tmp_path)

    assert result == {"SI-SNR": pytest.approx(20.0, abs=1e-3)}


def test_load_audio_takes_first_channel_and_resamples(tmp_path):
    left = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 8000, subtype="FLOAT")

    mono = load_audio(path)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, left)
    assert len(load_audio(path, 16000)) == 16000


def test_pesq_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError, match="8000 or 16000"):
        PESQ(fs=44100)


def test_pesq_requires_package(tmp_path, monkeypatch):
    monkeypatch.setattr(pesq_module, "pesq_fn", None)
    data = _write_pairs(tmp_path, [(REF, REF)])
    with pytest.raises(RuntimeError, match="pip install pesq"):
        PESQ()(data, "test", tmp_path)


@pytest.mark.parametrize("fs, mode", [(16000, "wb"), (8000, "nb")])
def test_pesq_passes_rate_and_mode_and_averages(tmp_path, monkeypatch, fs, mode):
    seen = []

    def fake_pesq(rate, ref, deg, band):
        seen.append((rate, band, len(ref), len(deg)))
        return 2.0 if len(seen) == 1 else 3.0

    monkeypatch.setattr(pesq_module, "pesq_fn", fake_pesq)
    # Second estimate is longer; scoring must trim both to the shorter length.
    data = _write_pairs(
        tmp_path, [(REF, REF), (REF, np.concatenate([REF, REF]))], fs=fs
    )

    assert PESQ(fs=fs)(data, "test", tmp_path) == {"PESQ": 2.5}
    assert seen == [(fs, mode, len(REF), len(REF))] * 2


def test_pesq_scores_clean_above_noisy(tmp_path):
    pytest.importorskip("pesq")
    clean = _speech_like()
    noisy = clean + 0.05 * np.random.default_rng(0).standard_normal(len(clean))
    data = _write_pairs(tmp_path, [(clean, clean)])
    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    noisy_data = _write_pairs(noisy_dir, [(clean, noisy.astype(np.float32))])

    clean_score = PESQ()(data, "test", tmp_path)["PESQ"]
    noisy_score = PESQ()(noisy_data, "test", noisy_dir)["PESQ"]

    assert clean_score > 4.0
    assert noisy_score < clean_score


def test_stoi_requires_package(tmp_path, monkeypatch):
    monkeypatch.setattr(stoi_module, "stoi_fn", None)
    data = _write_pairs(tmp_path, [(REF, REF)])
    with pytest.raises(RuntimeError, match="pip install pystoi"):
        STOI()(data, "test", tmp_path)


@pytest.mark.parametrize("extended, label", [(False, "STOI"), (True, "ESTOI")])
def test_stoi_passes_options_and_labels_result(tmp_path, monkeypatch, extended, label):
    seen = []

    def fake_stoi(ref, deg, rate, extended):
        seen.append((len(ref), len(deg), rate, extended))
        return 0.8

    monkeypatch.setattr(stoi_module, "stoi_fn", fake_stoi)
    data = _write_pairs(tmp_path, [(REF, np.concatenate([REF, REF]))])

    assert STOI(extended=extended)(data, "test", tmp_path) == {label: 0.8}
    assert seen == [(len(REF), len(REF), 16000, extended)]


def test_stoi_scores_identical_signals_near_one(tmp_path):
    pytest.importorskip("pystoi")
    clean = _speech_like()
    noisy = clean + 0.2 * np.random.default_rng(0).standard_normal(len(clean))
    data = _write_pairs(tmp_path, [(clean, clean)])
    noisy_dir = tmp_path / "noisy"
    noisy_dir.mkdir()
    noisy_data = _write_pairs(noisy_dir, [(clean, noisy.astype(np.float32))])

    clean_score = STOI()(data, "test", tmp_path)["STOI"]
    noisy_score = STOI()(noisy_data, "test", noisy_dir)["STOI"]

    assert clean_score == pytest.approx(1.0, abs=1e-3)
    assert noisy_score < clean_score


def test_metrics_return_nan_for_empty_test_set(tmp_path):
    (tmp_path / "ref.scp").write_text("", encoding="utf-8")
    (tmp_path / "hyp.scp").write_text("", encoding="utf-8")
    data = {"reference": tmp_path / "ref.scp", "enhanced": tmp_path / "hyp.scp"}

    assert math.isnan(SISNR()(data, "test", tmp_path)["SI-SNR"])
