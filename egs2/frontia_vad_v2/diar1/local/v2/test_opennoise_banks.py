"""Tests of local/v2/opennoise_banks.py on a synthetic manifest."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "opennoise_banks.py"
HEAD = "id\tpath\tkind\tsplit\tsource\tlicense\tduration_s\tmay_have_vocals\tcontains_speech\n"


def run(tmp_path, rows, *extra):
    m = tmp_path / "manifest.tsv"
    m.write_text(HEAD + "".join("\t".join(r) + "\n" for r in rows))
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(m),
            "--audio_root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


ROWS = [
    ("n1", "a/n1.flac", "noise", "train", "fsd50k", "CC0-1.0", "30", "0", "0"),
    (
        "n2",
        "a/n2.flac",
        "noise",
        "train",
        "fsd50k",
        "CC-BY-4.0",
        "30",
        "0",
        "1",
    ),  # speech
    (
        "n3",
        "a/n3.flac",
        "noise",
        "dev",
        "fsd50k",
        "CC0-1.0",
        "30",
        "0",
        "0",
    ),  # not train
    ("m1", "a/m1.flac", "music", "train", "musicnet", "CC-BY-4.0", "30", "0", "0"),
    (
        "m2",
        "a/m2.flac",
        "music",
        "train",
        "fma",
        "CC-BY-4.0",
        "30",
        "1",
        "0",
    ),  # singing
    ("r1", "a/r1.flac", "rir", "train", "but", "CC-BY-4.0", "1", "0", "0"),
    ("b1", "a/b1.flac", "babble", "train", "realman", "CC-BY-4.0", "30", "0", "1"),
]


def ids(path):
    return [x.split()[0] for x in path.read_text().splitlines()]


def test_flagged_rows_never_in_noise_or_music(tmp_path):
    r = run(tmp_path, ROWS)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    assert ids(out / "noise.scp") == ["n1"]
    assert ids(out / "music.scp") == ["m1"]
    assert ids(out / "rir.scp") == ["r1"]
    assert ids(out / "babble.scp") == ["b1"]
    assert ids(out / "noise_babble.scp") == ["n1", "b1"]


def test_licence_outside_the_list_stops(tmp_path):
    rows = ROWS + [
        ("n9", "a/n9.flac", "noise", "train", "x", "CC-BY-SA-4.0", "30", "0", "0")
    ]
    r = run(tmp_path, rows)
    assert r.returncode != 0 and "CC-BY-SA-4.0" in r.stderr


def test_cap_draws_sources_in_turn(tmp_path):
    rows = [
        (f"a{i}", f"a/a{i}.flac", "noise", "train", "big", "CC0-1.0", "3600", "0", "0")
        for i in range(10)
    ]
    rows += [
        ("s1", "a/s1.flac", "noise", "train", "small", "CC0-1.0", "3600", "0", "0")
    ]
    r = run(tmp_path, rows, "--cap_hours", "noise=2")
    assert r.returncode == 0, r.stderr
    got = ids(tmp_path / "out" / "noise.scp")
    assert len(got) == 2 and "s1" in got


@pytest.mark.parametrize("col", ["may_have_vocals", "contains_speech"])
def test_missing_columns_stop(tmp_path, col):
    m = tmp_path / "manifest.tsv"
    m.write_text("id\tpath\tkind\n")
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(m),
            "--audio_root",
            str(tmp_path),
            "--out",
            str(tmp_path / "o"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0
