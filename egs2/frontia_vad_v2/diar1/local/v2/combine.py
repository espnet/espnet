#!/usr/bin/env python3
"""Combine corpus dump directories (and their shards) into one set.

    python3 local/v2/combine.py dump/raw/train dump/v2/ami dump/v2/fsd50k ...

Each input is a dump directory written by prep.py, or a directory of
shard<k>/ dump directories. The text files are concatenated and sorted; the
Omni-IO entries keep pointing at the archives of the inputs. A per-domain
table of hours is written to <out>/domains.tsv.
"""

import argparse
import collections
from pathlib import Path

TEXT = ("wav.scp", "utt2num_samples", "utt2category", "espnet_rttm")


def dump_dirs(path: Path):
    if (path / "wav.scp").exists():
        return [path]
    shards = sorted(p for p in path.glob("shard*") if (p / "wav.scp").exists())
    if not shards:
        raise FileNotFoundError(f"no dump directory in {path}")
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    dirs = [d for p in args.inputs for d in dump_dirs(p)]
    args.out.mkdir(parents=True, exist_ok=True)
    for name in TEXT:
        lines = []
        for d in dirs:
            lines += (d / name).read_text().splitlines(keepends=True)
        if name != "espnet_rttm":
            keys = [line.split(maxsplit=1)[0] for line in lines]
            dup = [k for k, c in collections.Counter(keys).items() if c > 1]
            if dup:
                raise ValueError(f"{name}: duplicate ids, e.g. {dup[:3]}")
        (args.out / name).write_text("".join(sorted(lines)))
    header, rows = None, []
    for d in dirs:
        with (d / "manifest.tsv").open() as f:
            h = f.readline()
            header = header or h
            if h != header:
                raise ValueError(f"{d}/manifest.tsv: different columns")
            rows += f.readlines()
    (args.out / "manifest.tsv").write_text(header + "".join(sorted(rows)))
    (args.out / "feats_type").write_text("raw\n")
    cols = header.rstrip("\n").split("\t")
    hours = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for row in rows:
        r = dict(zip(cols, row.rstrip("\n").split("\t")))
        h = hours[(r["domain"], r["corpus"])]
        h[0] += 1
        h[1] += float(r["seconds"]) / 3600
        h[2] += float(r["speech_seconds"]) / 3600
    with (args.out / "domains.tsv").open("w") as f:
        f.write("domain\tcorpus\trecordings\thours\tspeech_hours\n")
        for (dom, cor), (n, t, s) in sorted(hours.items()):
            f.write(f"{dom}\t{cor}\t{n}\t{t:.2f}\t{s:.2f}\n")


if __name__ == "__main__":
    main()
