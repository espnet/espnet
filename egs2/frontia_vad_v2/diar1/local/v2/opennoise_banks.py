#!/usr/bin/env python3
"""Augmentation banks of v2.3 from OpenNoise's per-file manifest (train split only).

    python3 local/v2/opennoise_banks.py --manifest <OpenNoise manifest.tsv> \
        --audio_root <dir the manifest paths are relative to> --out data/opennoise_v01 \
        [--cap_hours noise=150,music=75,babble=50] [--seed 0]

Reads one row per audio file (tab-separated, a header). Columns: id, path, kind (noise,
music, rir, babble, human_nonspeech), split, source, license (SPDX or LicenseRef),
duration_s, and any of the flag columns may_have_vocals, vocal, singing, contains_speech
(true: 1/true/yes). Keeps split train; refuses a licence outside ALLOWED (the manifest
should hold none). Excludes every row flagged as vocal, singing or speech from the noise
and music banks (singing is speech: never used as noise). Writes:

- noise.scp (kind noise), music.scp (kind music), rir.scp (kind rir), babble.scp (kind
  babble, which is speech by definition and goes only into noise_babble.scp), and
  noise_babble.scp = noise.scp + babble.scp; "<id> <absolute audio path>" lines;
- bank.tsv: every row written to a list, with its bank, source, license and duration;
- counts.tsv: files and hours per bank x licence and per bank x flag.

The banks are loaded onto every GPU (wav_augment keep_on_device), about 2 bytes per
sample, i.e. 115 MB per hour of audio. --cap_hours limits a bank's hours: files are
drawn source by source in turn, in a seeded random order within each source, so that no
single source fills the cap.
"""

import argparse
import collections
import csv
import random
from pathlib import Path

ALLOWED = (
    "CC0-1.0",
    "PDM-1.0",
    "LicenseRef-Public-Domain",
    "LicenseRef-US-Public-Domain",
    "CC-BY-1.0",
    "CC-BY-2.0",
    "CC-BY-2.5",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY",
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CDLA-Permissive-1.0",
    "CDLA-Permissive-2.0",
    "ODC-By-1.0",
    "LicenseRef-Open-AoE-Dataset-License",
    "LicenseRef-UIowa-MIS-free-use",
)
FLAGS = ("may_have_vocals", "vocal", "singing", "contains_speech")
TRUE = {"1", "true", "yes", "y"}


def flagged(r):
    return [f for f in FLAGS if str(r.get(f, "")).strip().lower() in TRUE]


def capped(rows, hours, seed):
    """Rows drawn source by source in turn until `hours` (None: all rows)."""
    if hours is None:
        return rows
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, total, queues = [], 0.0, [list(v) for _, v in sorted(by.items())]
    while queues and total < hours * 3600:
        nxt = []
        for q in queues:
            if total >= hours * 3600:
                break
            r = q.pop()
            out.append(r)
            total += float(r["duration_s"])
            if q:
                nxt.append(q)
        queues = nxt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--audio_root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--cap_hours", default="", help="bank=hours,... (noise, music, babble, rir)"
    )
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    caps = {k: float(v) for k, v in (x.split("=") for x in a.cap_hours.split(",") if x)}
    rows = list(csv.DictReader(a.manifest.open(), delimiter="\t"))
    need = {"id", "path", "kind", "split", "source", "license", "duration_s"}
    missing = need - set(rows[0]) if rows else need
    if missing:
        raise SystemExit(f"{a.manifest}: no column {sorted(missing)}")
    if not any(f in rows[0] for f in FLAGS):
        raise SystemExit(f"{a.manifest}: none of the flag columns {FLAGS}")
    train = [r for r in rows if r["split"] == "train"]
    bad = collections.Counter(
        r["license"] for r in train if r["license"] not in ALLOWED
    )
    if bad:
        raise SystemExit(f"licences outside the allowed list: {dict(bad)}")
    banks = {}
    dropped = collections.Counter()
    for kind in ("noise", "music", "rir", "babble"):
        keep = []
        for r in (x for x in train if x["kind"] == kind):
            f = flagged(r)
            if f and kind != "babble":
                dropped[(kind, ",".join(f))] += 1
                continue
            keep.append(r)
        banks[kind] = capped(keep, caps.get(kind), a.seed)
    banks["noise_babble"] = banks["noise"] + banks["babble"]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rs in banks.items():
        with (a.out / f"{name}.scp").open("w") as f:
            for r in rs:
                f.write(f"{r['id']} {(a.audio_root / r['path']).resolve()}\n")
    counts = collections.defaultdict(lambda: [0, 0.0])
    with (a.out / "bank.tsv").open("w") as f:
        f.write("bank\tid\tkind\tsource\tlicense\tflags\tduration_s\n")
        for name in ("noise", "music", "rir", "babble"):
            for r in banks[name]:
                fl = ",".join(flagged(r)) or "-"
                f.write(
                    f"{name}\t{r['id']}\t{r['kind']}\t{r['source']}\t{r['license']}\t{fl}"
                    f"\t{r['duration_s']}\n"
                )
                for key in (("licence", name, r["license"]), ("flag", name, fl)):
                    counts[key][0] += 1
                    counts[key][1] += float(r["duration_s"]) / 3600
    with (a.out / "counts.tsv").open("w") as f:
        f.write("what\tbank\tvalue\tfiles\thours\n")
        for (w, b, v), (n, h) in sorted(counts.items()):
            f.write(f"{w}\t{b}\t{v}\t{n}\t{h:.2f}\n")
    for name, rs in banks.items():
        h = sum(float(r["duration_s"]) for r in rs) / 3600
        print(f"{name}: {len(rs)} files, {h:.1f} h, about {h * 0.115:.1f} GB per GPU")
    for (kind, f), n in sorted(dropped.items()):
        print(f"  dropped from {kind}: {n} rows flagged {f}")
    leak = [r["id"] for k in ("noise", "music") for r in banks[k] if flagged(r)]
    if leak:
        raise SystemExit(f"flagged rows in noise/music banks: {leak[:5]}")


if __name__ == "__main__":
    main()
