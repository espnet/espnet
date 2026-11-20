"""Tests of the external test sets: external_set.py, vad_infer.SetReader, the UEM of
score_vad.py.

python3 -m pytest local/v2/test_external_eval.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score_vad  # noqa: E402
from vad_infer import SetReader  # noqa: E402


def _kaldi_dir(tmp_path):
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    soundfile.write(wavs / "a.wav", np.zeros(32000, dtype=np.float32), 16000)
    soundfile.write(wavs / "b.flac", np.zeros((8000, 2), dtype=np.float32), 8000)
    d = tmp_path / "kaldi"
    d.mkdir()
    # paths written on another machine; --wav_dir points at the files here
    (d / "wav.scp").write_text("a /ocean/x/a.wav\nb /ocean/x/b.flac\n")
    (d / "rttm").write_text(
        "SPEAKER a 1 0.10 0.50 <NA> <NA> s1 <NA> <NA>\n"
        "SPEAKER a 1 0.40 0.40 <NA> <NA> s2 <NA> <NA>\n"
        "SPEAKER a 1 1.50 0.20 <NA> <NA> s1 <NA> <NA>\n"
    )
    (d / "all.uem").write_text("a 1 0.00 1.80\nc 1 0.00 5.00\n")
    return d, wavs


def test_external_set(tmp_path):
    d, wavs = _kaldi_dir(tmp_path)
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(HERE / "external_set.py"), str(d), str(out)]
        + ["--wav_dir", str(wavs)],
        check=True,
        capture_output=True,
    )
    assert (out / "wav.scp").read_text() == f"a {wavs}/a.wav\nb {wavs}/b.flac\n"
    assert (out / "utt2num_samples").read_text() == "a 32000\nb 16000\n"
    assert (out / "rttm").read_text().splitlines() == [
        "SPEAKER a 1 0.100 0.700 <NA> <NA> speech <NA> <NA>",
        "SPEAKER a 1 1.500 0.200 <NA> <NA> speech <NA> <NA>",
    ]
    assert (out / "uem").read_text() == "a 1 0.00 1.80\n"  # c is not in wav.scp


def test_set_reader_plain_files(tmp_path):
    _, wavs = _kaldi_dir(tmp_path)
    scp = tmp_path / "wav.scp"
    scp.write_text(f"a {wavs}/a.wav\nb {wavs}/b.flac\n")
    r = SetReader(str(scp))
    assert not r.omniio
    assert r["a"][1].shape == (32000,)
    rate, x = r["b"]  # 8 kHz stereo -> 16 kHz, first channel
    assert rate == 16000 and x.shape == (16000,)


def test_uem_regions(tmp_path):
    uem = tmp_path / "all.uem"
    uem.write_text("a.1 1 0.5 1.0\na.1 1 2.0 3.0\nz 1 0 1\n")
    out = tmp_path / "uem.md-eval"
    score_vad.write_uem(out, {"a.1": 0, "b": 0}, {"a.1": 64000, "b": 16000}, uem)
    assert out.read_text().splitlines() == [
        "a_1 1 0.500000 1.000000",
        "a_1 1 2.000000 3.000000",
        "b 1 0.000000 1.000000",
    ]
