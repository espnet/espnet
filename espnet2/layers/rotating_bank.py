"""Pools of additive recordings from which BatchWavAugment draws a capped bank.

A pool is a TSV, one row per file (e.g. a pack of noise pieces of one source), with
the columns id, path, kind, source, licence, duration_s and flagged (1: the file holds
speech, singing or vocals). ``caps`` gives the hours to draw per kind, e.g.
{"noise": 150, "babble": 50}. A pool refuses a flagged row of any kind other than
"babble", and a kind it has no cap for: nothing that is speech can enter a noise or
music bank by accident.

draw(seed) returns a subset whose hours per kind reach the kind's cap, drawn source by
source in turn (in a random order within each source, from the seed), so that no source
fills the cap alone. The same seed gives the same subset.
"""

import collections
import csv
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

COLUMNS = ("id", "path", "kind", "source", "licence", "duration_s", "flagged")
SPEECH_KINDS = ("babble",)


@dataclass(frozen=True)
class PoolRow:
    """One file of a pool."""

    id: str
    path: str
    kind: str
    source: str
    licence: str
    duration_s: float


def parse_caps(caps) -> Dict[str, float]:
    """{"noise": 150} from a dict or from the string "noise=150,babble=50" (hours)."""
    if isinstance(caps, dict):
        return {str(k): float(v) for k, v in caps.items()}
    out = {}
    for item in str(caps).split(","):
        if item.strip():
            k, v = item.split("=")
            out[k.strip()] = float(v)
    return out


class Pool:
    """The files of a pool TSV and the caps in hours per kind."""

    def __init__(self, path: str, caps):
        """Read the pool at ``path`` and check its rows against ``caps``."""
        self.path = path
        self.caps = parse_caps(caps)
        if not self.caps or any(v <= 0 for v in self.caps.values()):
            raise ValueError(f"{path}: give a positive cap in hours per kind: {caps}")
        rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
        missing = [c for c in COLUMNS if not rows or c not in rows[0]]
        if missing:
            raise ValueError(f"{path}: no column {missing}")
        self.rows: List[PoolRow] = []
        for r in rows:
            flagged = str(r["flagged"]).strip().lower() in ("1", "true", "yes")
            if flagged and r["kind"] not in SPEECH_KINDS:
                raise ValueError(
                    f"{path}: {r['id']} ({r['kind']}) is flagged as speech, singing "
                    "or vocals and cannot be in a noise or music pool"
                )
            if r["kind"] not in self.caps:
                raise ValueError(f"{path}: no cap for kind {r['kind']} ({r['id']})")
            self.rows.append(
                PoolRow(
                    r["id"],
                    r["path"],
                    r["kind"],
                    r["source"],
                    r["licence"],
                    float(r["duration_s"]),
                )
            )
        if not self.rows:
            raise ValueError(f"{path}: empty pool")
        ids = [r.id for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: duplicate ids")
        self.kinds = sorted(self.caps)
        self.hours = sum(r.duration_s for r in self.rows) / 3600

    def draw(self, seed: Sequence[int]) -> List[PoolRow]:
        """Return the capped subset of every kind for this seed (a sequence of ints)."""
        out = []
        for ki, kind in enumerate(self.kinds):
            rng = np.random.default_rng(np.random.SeedSequence(list(seed) + [ki]))
            by = collections.defaultdict(list)
            for r in self.rows:
                if r.kind == kind:
                    by[r.source].append(r)
            queues = []
            for source in sorted(by):
                q = by[source]
                queues.append([q[i] for i in rng.permutation(len(q))])
            cap, total = self.caps[kind] * 3600, 0.0
            while queues and total < cap:
                nxt = []
                for q in queues:
                    if total >= cap:
                        break
                    r = q.pop()
                    out.append(r)
                    total += r.duration_s
                    if q:
                        nxt.append(q)
                queues = nxt
        return out


def composition(rows: Sequence[PoolRow]) -> Dict[tuple, List[float]]:
    """{(kind, source, licence): [files, hours]} of a subset."""
    out = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        c = out[(r.kind, r.source, r.licence)]
        c[0] += 1
        c[1] += r.duration_s / 3600
    return dict(out)
