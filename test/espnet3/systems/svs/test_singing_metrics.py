"""Tests for the objective SVS metrics."""

import numpy as np
import pytest
import soundfile as sf

from espnet3.systems.svs.metrics.singing import METRIC_KEYS, SingingMetrics

pytest.importorskip("pyworld")
pytest.importorskip("pysptk")
pytest.importorskip("fastdtw")

FS = 16000


def _tone(path, f0, seconds=0.6, fs=FS):
    t = np.arange(int(seconds * fs)) / fs
    # A harmonic tone with an unvoiced tail so V/UV has both classes.
    wav = sum(0.3 / k * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 6))
    wav[int(0.45 * fs) :] = 0.01 * np.random.RandomState(0).randn(
        len(t) - int(0.45 * fs)
    )
    sf.write(path, wav.astype(np.float32), fs)


def _scp(path, rows):
    path.write_text("".join(f"{utt} {wav}\n" for utt, wav in rows), encoding="utf-8")
    return path


def _score(tmp_path, gen_f0, ref_f0, gen_fs=FS):
    gen_wav = tmp_path / "gen.wav"
    ref_wav = tmp_path / "ref.wav"
    _tone(gen_wav, gen_f0, fs=gen_fs)
    _tone(ref_wav, ref_f0)
    data = {
        "wav": _scp(tmp_path / "wav.scp", [("utt1", gen_wav)]),
        "ref": _scp(tmp_path / "ref.scp", [("utt1", ref_wav)]),
    }
    return SingingMetrics()(data, "test", tmp_path / "inference")


def test_identical_wavs_score_perfectly(tmp_path):
    result = _score(tmp_path, 220.0, 220.0)
    assert set(result) == set(METRIC_KEYS)
    assert result["mcd"] == pytest.approx(0.0, abs=1e-6)
    assert result["log_f0_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["semitone_acc"] == pytest.approx(1.0)
    assert result["vuv_err"] == pytest.approx(0.0)


def test_pitch_shift_is_detected(tmp_path):
    result = _score(tmp_path, 262.0, 220.0)
    assert result["log_f0_rmse"] > 0.1
    assert result["semitone_acc"] < 0.5
    per_utt = tmp_path / "inference" / "test" / "singing_metrics.txt"
    lines = per_utt.read_text(encoding="utf-8").splitlines()
    assert lines[0].split("\t") == ["utt_id", *METRIC_KEYS]
    assert lines[1].startswith("utt1\t")


def test_sampling_rate_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="sampling rate mismatch"):
        _score(tmp_path, 220.0, 220.0, gen_fs=8000)
