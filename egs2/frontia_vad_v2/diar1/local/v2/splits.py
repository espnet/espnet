"""The final train/dev split of each recording of a corpus dump.

A recording written with an official split keeps it. A recording of a corpus
without one carries its group (speaker, session, meeting) in manifest.tsv; the
dev groups of the corpus are registry.dev_groups() over all its groups, so
every such corpus has dev data (one group when it has 50 groups or fewer,
about 2 % of them otherwise) and no group is in both splits. A packed
recording lists all its groups: it goes to dev if all are dev groups, to
train if none is, and is dropped otherwise.
"""

import collections
from pathlib import Path
from typing import Dict, List, Tuple

from registry import dev_groups

SPLITS = ("train", "dev")


def corpus_dumps(corpus_dir: Path) -> List[Tuple[str, Path]]:
    """(written split, dump directory) of every dump of a corpus."""
    out = []
    for split in SPLITS:
        d = corpus_dir / split
        if (d / "wav.scp").exists():
            out.append((split, d))
        elif d.is_dir():
            out += [
                (split, p) for p in sorted(d.glob("shard*")) if (p / "wav.scp").exists()
            ]
    return out


def manifest_rows(dump: Path) -> List[Dict[str, str]]:
    with (dump / "manifest.tsv").open() as f:
        cols = f.readline().rstrip("\n").split("\t")
        return [dict(zip(cols, line.rstrip("\n").split("\t"))) for line in f]


def resolve(corpus_dir: Path) -> Tuple[Dict[str, str], dict]:
    """(recording id -> "train" | "dev" | "drop", summary) of a corpus dump."""
    rows = []
    for written, dump in corpus_dumps(corpus_dir):
        rows += [(written, r) for r in manifest_rows(dump)]
    groups = {g for _, r in rows for g in (r.get("group") or "").split(";") if g}
    dev = dev_groups(groups)
    final, moved = {}, collections.Counter()
    for written, r in rows:
        gs = [g for g in (r.get("group") or "").split(";") if g]
        if not gs:
            split = written
        elif all(g in dev for g in gs):
            split = "dev"
        elif not any(g in dev for g in gs):
            split = "train"
        else:
            split = "drop"
        final[r["rec_id"]] = split
        if split != written:
            moved[f"{written}->{split}"] += 1
    summary = dict(
        groups=len(groups),
        dev_groups=sorted(dev) if len(dev) <= 20 else len(dev),
        moved=dict(sorted(moved.items())),
    )
    return final, summary
