"""Tests of local/v2/aug_banks.py: packing, the train-split filter and licences."""

import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aug_banks import FS, Collector, PackWriter  # noqa: E402
from registry import AUTO  # noqa: E402


def test_packs_concatenate_clips_and_record_their_spans(tmp_path):
    pw = PackWriter(tmp_path, "music", "src", pack_seconds=1.0)
    for i, n in enumerate([6000, 6000, 6000, 3000]):
        pw.add(f"c{i}", np.full(n, 0.1 * (i + 1), np.float32), "CC0", f"m{i}")
    pw.flush()
    assert [p.name for p, _ in pw.files] == ["src_00000.wav", "src_00001.wav"]
    # a pack is closed after the clip that reaches pack_seconds
    x, rate = soundfile.read(pw.files[0][0])
    assert rate == FS and len(x) == 18000
    spans = [(c, s, e) for c, s, e, _, _ in pw.files[0][1]]
    assert spans == [("c0", 0.0, 0.375), ("c1", 0.375, 0.75), ("c2", 0.75, 1.125)]
    # the clip at 0.375 s is the second one
    assert abs(x[6000] - 0.2) < 1e-3
    assert [c for c, *_ in pw.files[1][1]] == ["c3"]
    assert abs(pw.hours() - 21000 / FS / 3600) < 1e-12


def test_collector_keeps_train_clips_resampled_first_channel(tmp_path):
    pw = PackWriter(tmp_path, "effects", "src", pack_seconds=1000.0)
    col = Collector(pw, lambda r: f"lic-{r}")
    stereo = np.stack([np.ones(8000), -np.ones(8000)], 1).astype(np.float32)
    assert col["train"].add("a", stereo, 8000, channels=[0], source="m/a") == 1
    assert col["dev"].add("b", stereo, 8000) == 0
    # a split by group follows registry.hash_split, as prep.py does
    kept = sum(col[f"{AUTO}g{i}"].add(f"g{i}", stereo, 8000) for i in range(200))
    assert 180 < kept < 200
    pw.flush()
    ((path, clips),) = pw.files
    x, _ = soundfile.read(path)
    assert len(x) == 16000 * (1 + kept)  # 1 s each at 16 kHz
    # channel 0 (+1, not -1); the pack is scaled down by the resampler's overshoot
    assert x[100:15900].min() > 0.8
    assert clips[0][3:] == ["lic-a", "m/a"]
    assert col.skipped == 1 + 200 - kept


def test_collector_stops_at_max_hours(tmp_path):
    pw = PackWriter(tmp_path, "effects", "src", pack_seconds=10.0)
    col = Collector(pw, lambda r: "", max_hours=2.5 / 3600)
    added = [col["train"].add(f"c{i}", np.ones(FS, np.float32), FS) for i in range(5)]
    assert added == [1, 1, 1, 0, 0]
