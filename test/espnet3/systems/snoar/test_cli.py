import numpy as np
import soundfile as sf

from espnet2.bin.cli import main as espnet_main
from espnet3.systems.snoar.src.cli import main
from espnet3.systems.snoar.src.data.manifest import read_manifest
from espnet3.systems.snoar.src.model import SNOARModel
from espnet3.systems.snoar.src.net import save_net

from .conftest import tiny_net


def test_enhance_harvest_manifest(wav_dir, tmp_path):
    model = SNOARModel("dnf", net=tiny_net(2))
    save_net(model.inference_net(), tmp_path / "m", {"readout": "dnf", "fs": 16000})
    assert (
        main(
            [
                "snoar",
                "enhance",
                str(wav_dir / "rec0.wav"),
                "-o",
                str(tmp_path / "o.wav"),
                "--model",
                str(tmp_path / "m"),
                "--device",
                "cpu",
                "--chunk-seconds",
                "0.5",
            ]
        )
        == 0
    )
    y, fs = sf.read(tmp_path / "o.wav")
    assert fs == 16000 and len(y) == 24000 and np.isfinite(y).all()
    assert (
        main(
            [
                "snoar",
                "enhance",
                str(wav_dir),
                "-o",
                str(tmp_path / "dir"),
                "--model",
                str(tmp_path / "m"),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert len(list((tmp_path / "dir").glob("*.wav"))) == 3
    assert (
        main(["snoar", "harvest", str(wav_dir), "-o", str(tmp_path / "pool.jsonl")])
        == 0
    )
    assert read_manifest(tmp_path / "pool.jsonl")
    assert (
        main(["snoar", "manifest", str(wav_dir), "-o", str(tmp_path / "rec.jsonl")])
        == 0
    )
    assert len(read_manifest(tmp_path / "rec.jsonl")) == 3


def test_errors_are_reported_without_traceback(tmp_path, capsys):
    assert (
        main(["snoar", "adapt", str(tmp_path / "none"), "-o", str(tmp_path / "o")]) == 2
    )
    assert "no such file or directory" in capsys.readouterr().err
    sf.write(tmp_path / "x8k.wav", np.zeros(8000, np.float32), 8000)
    assert (
        espnet_main(
            ["snoar", "adapt", str(tmp_path / "x8k.wav"), "-o", str(tmp_path / "o")]
        )
        == 1
    )
    assert "not 16000 Hz" in capsys.readouterr().err


def test_enhancer_auto_device(tmp_path, monkeypatch):
    import pytest
    import torch

    from espnet3.systems.snoar.inference import SNOAREnhancer

    model = SNOARModel("dnf", net=tiny_net(2))
    save_net(model.inference_net(), tmp_path / "m", {"readout": "dnf", "fs": 16000})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    enhancer = SNOAREnhancer(str(tmp_path / "m"), device="auto")
    assert enhancer.device == "cpu" and enhancer.readout == "dnf"
    y = enhancer(np.random.default_rng(0).normal(0, 0.1, 8000).astype(np.float32))
    assert y.shape == (8000,) and np.isfinite(y).all()
    with pytest.raises(FileNotFoundError, match="not an exported model"):
        SNOAREnhancer(str(tmp_path), device="auto")


def test_runner_rejects_batches():
    import pytest

    from espnet3.systems.snoar.inference import SNOARInferenceRunner

    with pytest.raises(ValueError, match="one recording at a time"):
        SNOARInferenceRunner.prepare_model_inputs({"speech": [None, None]})
