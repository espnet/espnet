from argparse import ArgumentParser

import numpy as np
import pytest
import soundfile as sf

import espnet2.bin.rst_inference as rst_inference


def test_get_parser():
    assert isinstance(rst_inference.get_parser(), ArgumentParser)


def test_read_audio_resamples_to_16k(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros((24000, 2), np.float32), 48000)
    waveform = rst_inference._read_audio(str(path))
    assert waveform.ndim == 1
    assert abs(len(waveform) - 8000) <= 1


@pytest.mark.parametrize("overlap_sec", [0.0, 0.5])
def test_restore_crossfade_reconstructs_identity(monkeypatch, overlap_sec):
    # a chunk restorer that upsamples 3x without changing content: the
    # overlap-add with fades must then reproduce the input exactly
    monkeypatch.setattr(
        rst_inference,
        "_restore_chunk",
        lambda piece, *a: np.repeat(piece.astype(np.float32), 3),
    )
    waveform = np.random.RandomState(0).randn(16000 * 5).astype(np.float32)
    out = rst_inference._restore(waveform, None, None, "cpu", 2.0, overlap_sec)
    ref = np.repeat(waveform, 3)
    n = min(len(out), len(ref))
    assert n > 0
    np.testing.assert_allclose(out[:n], ref[:n], atol=1e-5)


def test_restore_short_input_single_chunk(monkeypatch):
    monkeypatch.setattr(
        rst_inference, "_restore_chunk", lambda piece, *a: np.repeat(piece, 3)
    )
    waveform = np.ones(1000, np.float32)
    out = rst_inference._restore(waveform, None, None, "cpu", 2.0, 0.5)
    assert len(out) == 3000


def test_main_writes_manifest_and_resumes(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        rst_inference,
        "_load_feature_predictor",
        lambda *a: type("M", (), {"ssl_encoder": type("E", (), {"ssl_dim": 8})()})(),
    )
    monkeypatch.setattr(rst_inference, "_load_vocoder", lambda *a: None)
    monkeypatch.setattr(
        rst_inference, "_read_audio", lambda s: np.zeros(1600, np.float32)
    )

    def fake_restore(waveform, *a):
        calls.append(1)
        return np.zeros(4800, np.float32)

    monkeypatch.setattr(rst_inference, "_restore", fake_restore)
    scp = tmp_path / "in.scp"
    scp.write_text("u1 /x/1.wav\nu2 /x/2.wav\nu3 /x/3.wav\n")
    out = tmp_path / "out"
    (out / "wav").mkdir(parents=True)
    # an earlier, interrupted run finished u1 only
    sf.write(out / "wav" / "u1.wav", np.zeros(4800, np.float32), 48000)
    (out / "wav.scp.partial").write_text(f"u1 {out / 'wav' / 'u1.wav'}\n")
    rst_inference.main(
        [
            "--train_config",
            "x",
            "--model_file",
            "x",
            "--sidon_vocoder",
            "x",
            "--wav_scp",
            str(scp),
            "--output_dir",
            str(out),
        ]
    )
    assert len(calls) == 2
    lines = (out / "wav.scp").read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["u1", "u2", "u3"]
    assert len((out / "wav.scp.partial").read_text().splitlines()) == 3


def test_vocoder_args_are_exclusive(tmp_path):
    parser = rst_inference.get_parser()
    args = parser.parse_args(
        [
            "--train_config",
            "x",
            "--model_file",
            "x",
            "--sidon_vocoder",
            "x",
            "--vocoder_train_config",
            "y",
            "--vocoder_model_file",
            "z",
            "--wav_scp",
            "w",
            "--output_dir",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError):
        rst_inference._load_vocoder(args, 8, "cpu")
    args.sidon_vocoder = None
    args.vocoder_train_config = None
    with pytest.raises(ValueError):
        rst_inference._load_vocoder(args, 8, "cpu")
