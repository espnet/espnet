"""The final train/dev split of each recording of a corpus dump.

A recording written with an official split keeps it. A recording of a corpus
without one carries its group (speaker, session, meeting) in manifest.tsv; the
dev groups of the corpus are registry.dev_groups() over all its groups, so
every such corpus has dev data (one unit when it has 50 or fewer, about 2 %
of them otherwise) and no group is in both splits. A packed recording lists
all its groups; groups packed together form one unit (connected component),
which goes to one split whole. A corpus whose groups all form one unit, or
that has no groups and no official dev split, gets no dev.
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


def _components(packs: List[List[str]]) -> Dict[str, str]:
    """{group: representative} of the groups joined by the packs that hold them."""
    parent: Dict[str, str] = {}

    def find(g):
        parent.setdefault(g, g)
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for gs in packs:
        for g in gs:
            find(g)
        for g in gs[1:]:
            a, b = find(gs[0]), find(g)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {g: find(g) for g in parent}


def resolve(corpus_dir: Path) -> Tuple[Dict[str, str], dict]:
    """(recording id -> "train" | "dev", summary) of a corpus dump.

    Groups that share a packed recording are one unit (a connected component
    of the groups), so a split never separates them; the dev units are
    registry.dev_groups() over the units.
    """
    rows = []
    for written, dump in corpus_dumps(corpus_dir):
        rows += [(written, r) for r in manifest_rows(dump)]
    packs = [[g for g in (r.get("group") or "").split(";") if g] for _, r in rows]
    unit_of = _components([gs for gs in packs if gs])
    units = set(unit_of.values())
    # one unit only: no dev (it would take the whole corpus)
    dev = dev_groups(units) if len(units) >= 2 else set()
    final, moved = {}, collections.Counter()
    for (written, r), gs in zip(rows, packs):
        split = written if not gs else ("dev" if unit_of[gs[0]] in dev else "train")
        final[r["rec_id"]] = split
        if split != written:
            moved[f"{written}->{split}"] += 1
    summary = dict(
        groups=len(unit_of),
        units=len(units),
        dev_groups=sorted(dev) if len(dev) <= 20 else len(dev),
        moved=dict(sorted(moved.items())),
    )
    return final, summary
