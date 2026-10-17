"""Exercise the GeoLID data scripts without downloading speech corpora."""

import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[3] / "egs2/geolid/lid1/local"


def make_dump(path, utterance, samples=160000):
    path.mkdir()
    (path / "wav.scp").write_text(f"{utterance} /audio/{utterance}.wav\n")
    (path / "utt2lang").write_text(f"{utterance} eng\n")
    (path / "utt2num_samples").write_text(f"{utterance} {samples}\n")
    (path / "feats_type").write_text("raw\n")
    (path / "audio_format").write_text("wav\n")


def test_filter_and_combine(tmp_path):
    babel = tmp_path / "train_babel_lang"
    make_dump(babel, "long")
    for name, value in (
        ("wav.scp", "/audio/short.wav"),
        ("utt2lang", "fra"),
        ("utt2num_samples", "159999"),
    ):
        with (babel / name).open("a") as stream:
            stream.write(f"short {value}\n")
    before = {p.name: p.read_bytes() for p in babel.iterdir()}
    filtered = tmp_path / "train_babel_over_10s_lang"
    subprocess.run(
        [
            sys.executable,
            LOCAL / "filter_babel_over_10s.py",
            "--babel_dir",
            babel,
            "--babel_over_10s_dir",
            filtered,
        ],
        check=True,
    )
    assert (filtered / "utt2lang").read_text() == "long eng\n"
    assert (filtered / "category2utt").read_text() == "eng long\n"
    other = tmp_path / "train_fleurs_lang"
    make_dump(other, "fleurs")
    combined = tmp_path / "combined"
    command = [
        sys.executable,
        LOCAL / "combine_lid_data.py",
        "--output-dir",
        combined,
        filtered,
        other,
    ]
    subprocess.run(command, check=True)
    assert (combined / "lang2utt").read_text() == "eng fleurs long\n"
    assert (
        combined / "utt2dataset"
    ).read_text() == "fleurs train_fleurs_lang\nlong train_babel_over_10s_lang\n"
    assert before == {p.name: p.read_bytes() for p in babel.iterdir()}
    duplicate = subprocess.run(command + [filtered], capture_output=True, text=True)
    assert duplicate.returncode != 0
    assert "Duplicate utterance long" in duplicate.stderr
