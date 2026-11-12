"""Audio metric lifecycle and corpus aggregation checks."""

from importlib import import_module

import numpy as np
import pytest
import soundfile as sf

from espnet3.contract.kinds.audio import Audio

CASES = [
    ("sisnr", "SISNR", "_compute_sisnr", ["SI_SNR"]),
    ("pesq", "PESQ", "_compute_pesq", ["PESQ"]),
    ("stoi", "STOI", "_compute_stoi", ["STOI"]),
    ("stoi", "ESTOI", "_compute_stoi", ["ESTOI"]),
    ("sdr", "SDR", "_compute_sdr_sar_sir", ["SDR", "SAR", "SIR"]),
    (
        "dnsmos",
        "DNSMOS",
        "_compute_dnsmos",
        ["DNSMOS_OVRL", "DNSMOS_SIG", "DNSMOS_BAK", "DNSMOS_P808_MOS"],
    ),
]


@pytest.mark.parametrize("module,name,method,keys", CASES)
def test_state_reset_and_chunk_parity(
    module, name, method, keys, monkeypatch, tmp_path
):
    cls = getattr(import_module(f"espnet3.systems.esp2_tse.metrics.{module}"), name)
    metric = cls()

    def score(*args):
        value = float(args[0].mean())
        if module == "dnsmos":
            return {k: value for k in ("OVRL", "SIG", "BAK", "P808_MOS")}
        return (value,) * 3 if module == "sdr" else value

    monkeypatch.setattr(metric, method, score)
    audio = [Audio(np.full(128, v, dtype=np.float32), 16000) for v in (0.25, 0.5, 0.75)]
    args = {"hyp": audio}
    if module != "dnsmos":
        args["ref"] = audio
    metric.update(**args)
    assert metric.compute() == dict.fromkeys(keys, 0.5)
    whole = metric.state()
    metric.reset()
    with pytest.raises(ValueError, match="scores|samples|empty"):
        metric.compute()
    states = []
    for idx in range(3):
        metric.reset()
        metric.update(**{k: [v[idx]] for k, v in args.items()})
        states.append(metric.state())
    metric.load_state(metric.merge_states(states))
    assert metric.state() == whole
    assert metric.compute() == dict.fromkeys(keys, 0.5)
    with pytest.raises(ValueError):
        metric.update(hyp=[12], **({"ref": audio[:1]} if module != "dnsmos" else {}))
    # SCP strings are accepted by the declared audio kind before decoding.
    wav = tmp_path / "a.wav"
    sf.write(wav, audio[0].array, 16000, subtype="FLOAT")
    scp = tmp_path / "a.scp"
    scp.write_text(f"u1 {wav}\n")
    data = {k: scp for k in args}
    assert metric(data, "test", tmp_path) == dict.fromkeys(keys, 0.25)


def test_tsos_lifecycle_and_shape_validation():
    from espnet3.systems.esp2_tse.metrics.tsos import TSOS

    metric = TSOS(n_fft=32, hop_length=8)
    wave = Audio(np.random.default_rng(0).normal(size=128).astype(np.float32), 16000)
    metric.update(ref=[wave], hyp=[wave])
    assert metric.compute() == {"TSOS": 0.0}
    with pytest.raises(ValueError, match="shape|length"):
        metric.update(ref=[wave], hyp=[Audio(np.zeros(64), 16000)])


def test_dnsmos_downloads_use_path_objects(tmp_path, monkeypatch):
    from pathlib import Path

    import espnet2.enh.layers.dnsmos as backend
    from espnet3.systems.esp2_tse.metrics import dnsmos

    def download(url, dst_path, **kwargs):
        assert isinstance(dst_path, Path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.touch()

    monkeypatch.setattr(dnsmos, "download_url", download)
    sentinel = object()
    monkeypatch.setattr(backend, "DNSMOS_local", lambda **kwargs: sentinel)
    metric = dnsmos.DNSMOS(dnsmos_dir=str(tmp_path / "models"))
    assert metric._ensure_dnsmos() is sentinel
