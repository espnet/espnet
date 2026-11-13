"""Tests of the dev split by group (registry, prep.Writers, splits, combine).

python3 -m pytest local/v2/test_splits.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prep import Writers  # noqa: E402
from registry import Adapter, dev_groups, group_hash, split_by_hash  # noqa: E402
from splits import resolve  # noqa: E402

SPEC = Adapter("", "meeting", "H", "CC BY 4.0", None)


def test_few_groups_give_one_dev_group():
    groups = [f"meeting{i}" for i in range(7)]
    dev = dev_groups(groups)
    assert dev == {min(groups, key=lambda g: (group_hash(g), g))}
    assert dev_groups(reversed(groups)) == dev  # independent of order


def test_many_groups_give_about_two_percent():
    groups = [f"spk{i}" for i in range(5000)]
    dev = dev_groups(groups)
    assert 0.01 < len(dev) / len(groups) < 0.03


def test_no_group_in_dev_by_hash_still_gives_one():
    groups = [f"g{i}" for i in range(400) if group_hash(f"g{i}") % 100 >= 2][:60]
    assert len(groups) == 60 and len(dev_groups(groups)) == 1


def _write(tmp_path, recordings, pack_seconds=None):
    w = Writers(tmp_path, SPEC, "toy", 16000, 0, 1, pack_seconds)
    for rid, group, seconds in recordings:
        x = np.zeros(int(seconds * 16000), dtype=np.float32)
        w[split_by_hash(group)].add(rid, x, 16000, speech=[(0.1, 0.5)])
    w.close()
    return tmp_path / "toy"


def test_resolve_keeps_groups_apart(tmp_path):
    recs = [(f"r{m}_{k}", f"meeting{m}", 40.0) for m in range(6) for k in range(3)]
    final, summary = resolve(_write(tmp_path, recs))
    assert summary["groups"] == 6 and len(summary["dev_groups"]) == 1
    by_group = {}
    for rid, group, _ in recs:
        by_group.setdefault(group, set()).add(final[f"toy-{rid}"])
    assert all(len(s) == 1 for s in by_group.values())  # no group in both
    assert sum(s == {"dev"} for s in by_group.values()) == 1


def test_packed_groups_stay_together(tmp_path):
    # 10 s clips packed to 30 s: a pack holds several groups, which then form
    # units; no unit is split and nothing is dropped
    recs = [(f"c{i}", f"spk{i}", 10.0) for i in range(60)]
    final, summary = resolve(_write(tmp_path, recs, pack_seconds=30))
    assert set(final.values()) <= {"train", "dev"}
    assert (
        summary["units"] >= 20 and "dev" in final.values() and "train" in final.values()
    )
    for line in (tmp_path / "toy").glob("*/manifest.tsv"):
        for row in line.read_text().splitlines()[1:]:
            rid = row.split("\t")[0]
            assert final[rid] in ("train", "dev")


def test_one_unit_stays_in_train(tmp_path):
    recs = [(f"c{i}", "only_speaker", 40.0) for i in range(3)]
    final, summary = resolve(_write(tmp_path, recs))
    assert summary["units"] == 1 and set(final.values()) <= {"train", "dev"}
    assert (
        "dev" not in final.values() or all(v == "dev" for v in final.values()) is False
    )


def test_combine_reports_a_corpus_without_dev(tmp_path):
    w = Writers(tmp_path, SPEC, "nodev", 16000, 0, 1, None)
    w["train"].add("a", np.zeros(16000, dtype=np.float32), 16000)
    w.close()
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "combine.py"),
            str(tmp_path / "raw"),
            str(tmp_path / "nodev"),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0 and "no dev recordings in: nodev" in r.stderr
    assert (tmp_path / "raw" / "train" / "wav.scp").read_text().startswith("nodev-a ")
