#!/usr/bin/env python3
"""Records removed and added between two combined dumps, per corpus and licence class.

    python3 local/v2/dump_diff.py <old raw dir> <new raw dir> <old audit prefix> \
        <new audit prefix> [--expect_changed musan]

For train and dev: the records (rec_id) of manifest.tsv in only one of the two, by
corpus, with hours, and the licence class of their clips from the licence_audit.py
dump tables (<audit prefix>_<split>.tsv). Records of every corpus not in
--expect_changed must be the same in both (same rec_id, seconds, source and split);
otherwise the command lists the corpora and exits 1.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def manifest(d: Path):
    rows = {}
    for r in csv.DictReader(open(d / "manifest.tsv"), delimiter="\t"):
        rows[r["rec_id"]] = (r["corpus"], float(r["seconds"]), r["source"])
    return rows


def classes(audit: Path):
    out = defaultdict(set)
    if audit.exists():
        for r in csv.DictReader(open(audit), delimiter="\t"):
            out[r["record"]].add(r["class"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("old", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("old_audit")
    p.add_argument("new_audit")
    p.add_argument("--expect_changed", default="musan")
    a = p.parse_args()
    allowed = set(a.expect_changed.split(","))
    unexpected = set()
    for split in ("train", "dev"):
        old, new = manifest(a.old / split), manifest(a.new / split)
        oc = classes(Path(f"{a.old_audit}_{split}.tsv"))
        nc = classes(Path(f"{a.new_audit}_{split}.tsv"))
        agg = defaultdict(lambda: [0, 0.0])
        for rid in set(old) - set(new):
            c, s, _ = old[rid]
            agg[(c, "removed", ",".join(sorted(oc.get(rid, {"?"}))))][0] += 1
            agg[(c, "removed", ",".join(sorted(oc.get(rid, {"?"}))))][1] += s / 3600
        for rid in set(new) - set(old):
            c, s, _ = new[rid]
            agg[(c, "added", ",".join(sorted(nc.get(rid, {"?"}))))][0] += 1
            agg[(c, "added", ",".join(sorted(nc.get(rid, {"?"}))))][1] += s / 3600
        for rid in set(old) & set(new):
            if old[rid][:2] != new[rid][:2] or old[rid][2] != new[rid][2]:
                c = new[rid][0]
                agg[(c, "changed", "")][0] += 1
                agg[(c, "changed", "")][1] += new[rid][1] / 3600
        print(f"== {split}: old {len(old)} records, new {len(new)} records")
        if not agg:
            print("  identical")
        for (c, what, cls), (n, h) in sorted(agg.items()):
            flag = "" if c in allowed else "   <-- UNEXPECTED"
            print(f"  {c:24s} {what:8s} {cls:22s} {n:7d} rec {h:9.3f} h{flag}")
            if c not in allowed:
                unexpected.add(c)
    if unexpected:
        print("corpora changed that should not have:", ", ".join(sorted(unexpected)))
        sys.exit(1)
    print("only", ", ".join(sorted(allowed)), "changed")


if __name__ == "__main__":
    main()
