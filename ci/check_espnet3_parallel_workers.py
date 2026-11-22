#!/usr/bin/env python3
"""Check that espnet3 stages give the same results on one and on two workers.

ci/test_integration_espnet3.sh runs mini_an4's collect_stats, infer and
measure stages on one CPU worker and again on two (Dask LocalCluster), and
hands the output directories here. With one worker the work is a single
shard, so only the two-worker runs split the data into shards, build the
model in separate worker processes and merge the shard outputs; this script
is what makes those paths tested rather than merely executed.

It checks, for each pair:

- infer: every ``.scp`` the stage wrote is byte-identical, and the parallel
  run really was split (at least one test set in two or more shards).
- measure: ``metrics.json`` is identical.
- collect_stats: for train and valid, the shape files hold the same
  utterances and shapes, the statistics files hold the same keys, counts are
  identical and sums agree to rounding (shards add up in a different order),
  and the parallel run was split.

A run that was not split would pass the comparisons trivially, so it is a
failure in its own right: the test data or batch size has changed so that
the test no longer exercises what it is for.

Exit status is 0 when every check passes and 1 otherwise, with each failure
printed.

Example:
    python ci/check_espnet3_parallel_workers.py \
        --inference exp/training/inference_serial \
        --inference-parallel exp/training/inference_parallel \
        --stats exp/stats_serial --stats-parallel exp/stats_parallel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# A sum over float32 features accumulated in another order: rounding, no more.
STATS_RTOL = 1e-5


def _shard_count(directory: Path) -> int:
    """Return how many shards the run planned, from its manifest."""
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return 0
    return len(json.loads(manifest.read_text(encoding="utf-8")).get("shards", []))


def _unfinished_shards(directory: Path) -> list[str]:
    """Return the shard directories that have no completion marker."""
    return sorted(
        shard.name
        for shard in directory.glob("split.*")
        if shard.is_dir() and not (shard / "done").exists()
    )


def _same_names(what: str, serial: list[str], parallel: list[str]) -> list[str]:
    """Return a failure when the two runs wrote different sets of names.

    Both directions count: an output only the parallel run wrote would
    otherwise never be compared with anything.
    """
    if sorted(serial) == sorted(parallel):
        return []
    return [
        f"{what}: one worker wrote {sorted(serial)}, two workers {sorted(parallel)}"
    ]


def _test_sets(directory: Path) -> list[str]:
    """Return the test sets an inference run wrote (those with a manifest)."""
    if not directory.is_dir():
        return []
    return sorted(
        p.name for p in directory.iterdir() if (p / "manifest.json").is_file()
    )


def check_inference(serial: Path, parallel: Path) -> list[str]:
    """Compare two inference directories; return the failures."""
    sets = _test_sets(serial)
    if not sets:
        return [f"{serial}: no test set with a manifest.json"]
    failures = _same_names("test sets", sets, _test_sets(parallel))
    split = []
    for name in sets:
        a, b = serial / name, parallel / name
        if not b.is_dir():
            failures.append(f"{b}: missing (the serial run has {name})")
            continue
        scps = sorted(p.name for p in a.glob("*.scp"))
        if not scps:
            failures.append(f"{a}: no .scp written")
        if scps != sorted(p.name for p in b.glob("*.scp")):
            failures.append(
                f"{name}: .scp files differ: {scps} vs "
                f"{sorted(p.name for p in b.glob('*.scp'))}"
            )
        for scp in scps:
            if (b / scp).is_file() and (a / scp).read_bytes() != (b / scp).read_bytes():
                failures.append(f"{name}/{scp}: differs between one and two workers")
        unfinished = _unfinished_shards(b)
        if unfinished:
            failures.append(f"{b}: shards without a done marker: {unfinished}")
        if _shard_count(b) >= 2:
            split.append(name)
    if not split:
        failures.append(
            f"{parallel}: no test set was split into two or more shards, so the "
            "parallel path was not exercised"
        )
    metrics_a, metrics_b = serial / "metrics.json", parallel / "metrics.json"
    if not (metrics_a.is_file() and metrics_b.is_file()):
        failures.append("metrics.json missing from one of the runs")
    elif json.loads(metrics_a.read_text()) != json.loads(metrics_b.read_text()):
        failures.append("metrics.json differs between one and two workers")
    return failures


def _read_shapes(path: Path) -> dict[str, str]:
    """Return a shape file as utterance id -> shape."""
    shapes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, _, value = line.partition(" ")
            shapes[key] = value.strip()
    return shapes


def check_stats(serial: Path, parallel: Path) -> list[str]:
    """Compare two collect_stats directories; return the failures."""
    failures = []
    split = []
    for mode in ("train", "valid"):
        a, b = serial / mode, parallel / mode
        if not (a.is_dir() and b.is_dir()):
            failures.append(f"{mode}: missing from one of the runs")
            continue
        shape_files = sorted(p.name for p in a.glob("*_shape"))
        if not shape_files:
            failures.append(f"{a}: no shape files written")
        failures += _same_names(
            f"{mode} shape files", shape_files, [p.name for p in b.glob("*_shape")]
        )
        for name in shape_files:
            if not (b / name).is_file():
                failures.append(f"{b / name}: missing")
            elif _read_shapes(a / name) != _read_shapes(b / name):
                failures.append(f"{mode}/{name}: differs between one and two workers")
        stats_files = sorted(p.name for p in a.glob("*_stats.npz"))
        if not stats_files:
            failures.append(f"{a}: no statistics written")
        failures += _same_names(
            f"{mode} statistics files",
            stats_files,
            [p.name for p in b.glob("*_stats.npz")],
        )
        for name in stats_files:
            if not (b / name).is_file():
                failures.append(f"{b / name}: missing")
                continue
            with np.load(a / name) as x, np.load(b / name) as y:
                if sorted(x.files) != sorted(y.files):
                    failures.append(f"{mode}/{name}: keys {x.files} vs {y.files}")
                    continue
                for key in x.files:
                    u, v = x[key], y[key]
                    same = (
                        np.array_equal(u, v)
                        if key == "count"
                        else u.shape == v.shape and np.allclose(u, v, rtol=STATS_RTOL)
                    )
                    if not same:
                        failures.append(
                            f"{mode}/{name}[{key}]: differs between one and two "
                            "workers"
                        )
        unfinished = _unfinished_shards(b)
        if unfinished:
            failures.append(f"{b}: shards without a done marker: {unfinished}")
        if _shard_count(b) >= 2:
            split.append(mode)
    if not split:
        failures.append(
            f"{parallel}: neither train nor valid was split into two or more "
            "shards, so the parallel path was not exercised"
        )
    return failures


def main(argv=None) -> int:
    """Run the checks given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--inference-parallel", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--stats-parallel", type=Path, required=True)
    args = parser.parse_args(argv)

    failures = check_inference(args.inference, args.inference_parallel)
    failures += check_stats(args.stats, args.stats_parallel)
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        return 1
    print("OK: one and two CPU workers give the same collect_stats, infer and measure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
