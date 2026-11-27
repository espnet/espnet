import numpy as np
import pytest
import soundfile as sf

from espnet3.systems.snoar.metrics.pesq import PESQ
from espnet3.systems.snoar.metrics.sisdr import SISDR
from espnet3.systems.snoar.metrics.stoi import STOI


def test_metrics_score_identical_files(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(32000) / 16000
    x = (
        0.3 * np.sin(2 * np.pi * 300 * t) * (1 + np.sin(2 * np.pi * 3 * t))
        + 0.01 * rng.standard_normal(len(t))
    ).astype(np.float32)
    sf.write(tmp_path / "r.wav", x, 16000)
    sf.write(tmp_path / "h.wav", 0.5 * x, 16000)
    for metric, key, low in (
        (SISDR(), "si_sdr", 40),
        (PESQ(), "pesq_wb", 4.0),
        (STOI(), "stoi", 0.99),
    ):
        metric.update(ref=[str(tmp_path / "r.wav")], hyp=[str(tmp_path / "h.wav")])
        assert metric.compute()[key] >= low


def test_metric_errors(tmp_path):
    with pytest.raises(ValueError, match="No utterances"):
        SISDR().compute()
    sf.write(tmp_path / "a.wav", np.zeros(800, np.float32), 8000)
    sf.write(tmp_path / "b.wav", np.zeros(1600, np.float32), 16000)
    with pytest.raises(ValueError, match="Hz"):
        SISDR().update(ref=[str(tmp_path / "a.wav")], hyp=[str(tmp_path / "b.wav")])
