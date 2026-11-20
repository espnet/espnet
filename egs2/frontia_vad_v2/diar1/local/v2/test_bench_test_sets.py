"""Tests of local/v2/bench_test_sets.py (TextGrid reading, offsets, the written set).

python3 -m pytest local/v2/test_bench_test_sets.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_test_sets as b  # noqa: E402

TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "P0127"
        xmin = 0
        xmax = 10
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = ""
        intervals [2]:
            xmin = 1.5
            xmax = 4.25
            text = "你好 世界"
        intervals [3]:
            xmin = 4.25
            xmax = 5
            text = "<NOISE>"
        intervals [4]:
            xmin = 5
            xmax = 10
            text = "[laugh] 对"
"""


def test_textgrid_intervals_and_speech():
    iv = b.textgrid_intervals(TEXTGRID)
    assert [(s, e) for s, e, _ in iv] == [(0, 1.5), (1.5, 4.25), (4.25, 5), (5, 10)]
    assert [b.is_speech(t) for _, _, t in iv] == [False, True, False, True]
    assert not b.is_speech("  [laugh] <sil> ")


def test_lag_seconds_finds_the_delay():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(5 * b.FS).astype(np.float32)
    rec = np.concatenate([np.zeros(800, dtype=np.float32), ref])[: len(ref)]
    assert abs(b.lag_seconds(ref, rec) - 800 / b.FS) < 1e-9


def test_write_set(tmp_path):
    pytest.importorskip("espnet2.fileio.omniio_audio")
    x = np.zeros(3 * b.FS, dtype=np.float32)
    b.write_set(
        tmp_path / "s",
        [("r1", x, [(0.5, 1.0), (0.9, 1.5), (2.8, 9.0)])],
        "test reference",
    )
    rttm = (tmp_path / "s" / "rttm").read_text().splitlines()
    assert rttm == [
        "SPEAKER r1 1 0.500 1.000 <NA> <NA> speech <NA> <NA>",
        "SPEAKER r1 1 2.800 0.200 <NA> <NA> speech <NA> <NA>",
    ]
    assert (tmp_path / "s" / "utt2num_samples").read_text() == f"r1 {3 * b.FS}\n"
    assert (tmp_path / "s" / "wav.scp").read_text().startswith("r1 ")


def test_write_set_per_speaker_with_uem(tmp_path):
    pytest.importorskip("espnet2.fileio.omniio_audio")
    x = np.zeros(4 * b.FS, dtype=np.float32)
    segs = [(0.5, 1.0, "P1"), (0.8, 1.5, "P2"), (0.9, 1.2, "P1"), (2.0, 9.0, "P2")]
    b.write_set(tmp_path / "s", [("r1", x, segs)], "ref", {"r1": [(0.4, 12000.0)]})
    assert (tmp_path / "s" / "rttm").read_text().splitlines() == [
        "SPEAKER r1 1 0.500 0.700 <NA> <NA> P1 <NA> <NA>",
        "SPEAKER r1 1 0.800 0.700 <NA> <NA> P2 <NA> <NA>",
        "SPEAKER r1 1 2.000 2.000 <NA> <NA> P2 <NA> <NA>",
    ]
    assert (tmp_path / "s" / "uem").read_text() == "r1 1 0.400 4.000\n"
