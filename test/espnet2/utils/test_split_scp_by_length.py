"""pyscripts/utils/split_scp_by_length.py, used by diar.sh --split_by_length."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "egs2/TEMPLATE/asr1/pyscripts/utils/split_scp_by_length.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("split_scp_by_length", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _totals(parts, sizes, n):
    totals = [0.0] * n
    for p, s in zip(parts, sizes):
        totals[p] += s
    return totals


def test_long_entries_are_spread(module):
    # 4 hour-long recordings first, then 400 short clips: a contiguous split into
    # 4 parts puts all long recordings in part 0.
    keys = [f"a{i:03d}" for i in range(4)] + [f"b{i:03d}" for i in range(400)]
    lengths = {k: 3600.0 for k in keys[:4]}
    lengths.update({k: 10.0 for k in keys[4:]})
    parts = module.assign(keys, lengths, 4)
    assert sorted(parts[:4]) == [0, 1, 2, 3]
    totals = _totals(parts, [lengths[k] for k in keys], 4)
    assert max(totals) - min(totals) <= 10.0


def test_balance_bound_and_every_part_used(module):
    # greedy longest-first: max load <= mean load + the largest entry
    keys = [f"k{i:04d}" for i in range(1000)]
    lengths = {k: float((i * 7919) % 997 + 1) for i, k in enumerate(keys)}
    n = 16
    parts = module.assign(keys, lengths, n)
    totals = _totals(parts, [lengths[k] for k in keys], n)
    assert min(totals) > 0
    assert max(totals) <= sum(totals) / n + max(lengths.values())


def test_missing_lengths_count_as_median(module):
    # known 4, 6, 8: "m" counts as 6 and goes with "b"; as 1 (the fallback when
    # nothing is known) it would go to part 0 with "c"
    keys = ["a", "b", "c", "m"]
    parts = module.assign(keys, {"a": 4.0, "b": 6.0, "c": 8.0}, 2)
    assert parts == [0, 1, 0, 1]


def test_no_known_lengths_count_as_one(module):
    parts = module.assign(["a", "b", "c"], {}, 3)
    assert sorted(parts) == [0, 1, 2]


def test_zero_lengths_use_every_part(module):
    keys = [f"k{i}" for i in range(10)]
    parts = module.assign(keys, {k: 0.0 for k in keys}, 4)
    assert sorted(set(parts)) == [0, 1, 2, 3]
    # a few long keys and many empty ones: still one key in every part
    lengths = {k: 0.0 for k in keys}
    lengths.update({"k0": 5.0, "k1": 3.0})
    parts = module.assign(keys, lengths, 4)
    assert sorted(set(parts)) == [0, 1, 2, 3]


def test_deterministic(module):
    keys = [f"k{i}" for i in range(50)]
    lengths = {k: 5.0 for k in keys}
    assert module.assign(keys, lengths, 7) == module.assign(keys, lengths, 7)


def test_cli_keeps_lines_and_order(tmp_path):
    scp = tmp_path / "wav.scp"
    lens = tmp_path / "utt2num_samples"
    keys = [f"u{i:02d}" for i in range(20)]
    scp.write_text("".join(f"{k} /data/{k}.flac\n" for k in keys))
    lens.write_text("".join(f"{k} {(i % 5 + 1) * 16000}\n" for i, k in enumerate(keys)))
    outs = [tmp_path / f"keys.{n}.scp" for n in range(1, 4)]
    subprocess.run(
        [sys.executable, str(SCRIPT), "--lengths", str(lens), str(scp)]
        + [str(o) for o in outs],
        check=True,
    )
    got = [o.read_text().splitlines() for o in outs]
    assert all(got)
    # every input line once, and each part in input order
    assert sorted(sum(got, [])) == scp.read_text().splitlines()
    for part in got:
        assert part == sorted(part)


def test_cli_fewer_lines_than_outputs(tmp_path):
    scp = tmp_path / "wav.scp"
    lens = tmp_path / "utt2num_samples"
    scp.write_text("a x\n")
    lens.write_text("a 1\n")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--lengths", str(lens), str(scp)]
        + [str(tmp_path / f"o{n}") for n in range(2)],
    )
    assert r.returncode != 0
