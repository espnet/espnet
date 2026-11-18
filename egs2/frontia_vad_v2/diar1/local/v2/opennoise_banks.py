#!/usr/bin/env python3
"""Pools of the v2.3 rotating banks from OpenNoise v0.1's training lists.

    python3 local/v2/opennoise_banks.py --lists <OpenNoise data/lists> --out data/opennoise_v01 \
        [--keep_babble_routed_noise]

OpenNoise writes, per kind (noise, music, rir, babble, human_nonspeech) and split,
<kind>_<split>.scp ("<entry> <wav path>"; noise, music and babble entries are packs of
about 300 s of one source) and <kind>_<split>_manifest.tsv, one row per piece: entry,
start_s, end_s, kind, source, split, duration_s, spdx, licence_class, speech_flag,
vocal_flag, speech_check, and the credit columns. From the train split this writes:

- noise_pool.tsv (packs of kind noise), noise_babble_pool.tsv (noise and babble packs) and
  music_pool.tsv: one row per pack (id, path, kind, source, licence = the SPDX ids of its
  pieces joined by "+", duration_s = its last piece's end, flagged = any piece with
  speech_flag or vocal_flag, babble packs always 1), the input of BatchWavAugment's
  noise_pool / music_pool (espnet2.layers.rotating_bank.Pool);
- rir.scp: every impulse response (a small bank, loaded whole);
- pool_counts.tsv: packs, pieces and hours per pool by licence and by flag.

It stops when a noise or music piece is flagged as speech or vocals, when a licence class is
outside PD/CC0, BY, permissive and accepted, or when an scp entry has no manifest row (or
the reverse). Noise packs routed from babble candidates (a piece whose speech_check starts
with "noise ratio=": crowd recordings that passed the speech screen) are left out unless
--keep_babble_routed_noise.
"""

import argparse
import collections
import csv
from pathlib import Path

CLASSES = {"PD/CC0", "BY", "permissive", "accepted"}
TRUE = {"1", "true", "yes"}


def read_kind(lists: Path, kind: str):
    scp = dict(
        x.split(maxsplit=1)
        for x in (lists / f"{kind}_train.scp").read_text().splitlines()
        if x
    )
    pieces = collections.defaultdict(list)
    with open(lists / f"{kind}_train_manifest.tsv", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pieces[r["entry"]].append(r)
    if set(scp) != set(pieces):
        raise SystemExit(
            f"{kind}: {len(set(scp) - set(pieces))} scp entries without manifest rows, "
            f"{len(set(pieces) - set(scp))} manifest entries without scp lines"
        )
    return scp, pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lists", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--keep_babble_routed_noise", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    packs, counts, routed = {}, collections.defaultdict(lambda: [0, 0, 0.0]), 0
    for kind in ("noise", "music", "babble"):
        scp, pieces = read_kind(a.lists, kind)
        rows = []
        for entry, ps in sorted(pieces.items()):
            bad = {p["licence_class"] for p in ps} - CLASSES
            if bad:
                raise SystemExit(f"{kind} {entry}: licence class {bad}")
            flagged = any(
                str(p.get(c, "")).strip().lower() in TRUE
                for p in ps
                for c in ("speech_flag", "vocal_flag")
            )
            if flagged and kind != "babble":
                raise SystemExit(
                    f"{kind} {entry}: a piece is flagged as speech or vocals"
                )
            if kind == "noise" and any(
                p.get("speech_check", "").startswith("noise ratio=") for p in ps
            ):
                routed += 1
                if not a.keep_babble_routed_noise:
                    continue
            lic = "+".join(sorted({p["spdx"] for p in ps}))
            dur = max(float(p["end_s"]) for p in ps)
            fl = 1 if kind == "babble" else int(flagged)
            rows.append(
                (entry, scp[entry].strip(), kind, ps[0]["source"], lic, dur, fl)
            )
            for key in (("licence", lic), ("flag", str(fl))):
                c = counts[(kind,) + key]
                c[0] += 1
                c[1] += len(ps)
                c[2] += dur / 3600
        packs[kind] = rows
    head = "id\tpath\tkind\tsource\tlicence\tduration_s\tflagged\n"
    for name, kinds in (
        ("noise_pool", ("noise",)),
        ("noise_babble_pool", ("noise", "babble")),
        ("music_pool", ("music",)),
    ):
        with open(a.out / f"{name}.tsv", "w") as f:
            f.write(head)
            for k in kinds:
                for r in packs[k]:
                    f.write("\t".join(map(str, r)) + "\n")
    rir = dict(
        x.split(maxsplit=1)
        for x in (a.lists / "rir_train.scp").read_text().splitlines()
        if x
    )
    (a.out / "rir.scp").write_text(
        "".join(f"{k} {v.strip()}\n" for k, v in sorted(rir.items()))
    )
    with open(a.out / "pool_counts.tsv", "w") as f:
        f.write("kind\twhat\tvalue\tpacks\tpieces\thours\n")
        for (k, w, v), (n, p, h) in sorted(counts.items()):
            f.write(f"{k}\t{w}\t{v}\t{n}\t{p}\t{h:.2f}\n")
    for k, rows in packs.items():
        print(f"{k}: {len(rows)} packs, {sum(r[5] for r in rows) / 3600:.1f} h")
    print(f"rir: {len(rir)} impulse responses")
    print(
        f"noise packs routed from babble candidates: {routed} "
        f"({'kept' if a.keep_babble_routed_noise else 'left out'})"
    )


if __name__ == "__main__":
    main()
