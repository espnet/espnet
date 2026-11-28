#!/usr/bin/env python3
"""Transcript units of the corpora labelled by alignment (label source A).

    python3 local/v2/a_label_units.py dump/v2/<corpus> [...]

Reads only the dumps (manifest.tsv after alignment, before_align/manifest.tsv
for what alignment dropped, align_config.json): nothing in the corpora. A
transcript unit is what one transcript covers. A recording shorter than the
packing length (prep.py --pack_seconds) was packed with others into one dump
recording "<first id>+<n>" whose transcripts were joined, so the clip
boundaries inside a pack are not in the dump: a pack's n units are counted
with the pack's mean length (marked "packed, mean"). A recording that was not
packed is one unit with its exact length. Prints per corpus the unit length
distribution (p50/p90/p99/max), the speech hours in units longer than 10, 30
and 60 s, the share of speech in packs, the alignment settings, and the
first lines of the corpus reader's docstring (adapters), which say where the
transcripts and their segmentation come from.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import corpus_dumps, manifest_rows  # noqa: E402


def units(rows):
    """[(seconds, speech seconds, packed)] of the transcript units of rows."""
    out = []
    for r in rows:
        tail = r["rec_id"].rsplit("+", 1)
        n = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 1
        sec, sp = float(r["seconds"]), float(r.get("speech_seconds") or 0.0)
        out += [(sec / n, sp / n, n > 1)] * n
    return out


def reader_doc(corpus):
    try:
        import adapters

        doc = adapters.ADAPTERS[corpus].prepare.__doc__ or ""
    except Exception as e:  # noqa: BLE001 - reported, not fatal
        return f"(no reader docstring: {type(e).__name__})"
    lines = [x.strip() for x in doc.strip().splitlines() if x.strip()]
    return " ".join(lines[:3])


def main():
    for d in map(Path, sys.argv[1:]):
        kept, before, configs = [], [], set()
        for _, dump in corpus_dumps(d):
            rows = [r for r in manifest_rows(dump) if r.get("label_source") == "A"]
            if not rows:
                continue
            kept += rows
            b = dump / "before_align" / "manifest.tsv"
            if b.exists():
                before += [
                    r
                    for r in manifest_rows(dump / "before_align")
                    if r.get("label_source") == "A"
                ]
            c = dump / "align_config.json"
            if c.exists():
                cfg = json.loads(c.read_text())
                configs.add(
                    f"min_pause {cfg.get('min_pause')}, min_word_score "
                    f"{cfg.get('min_word_score')}, word_timing "
                    f"{cfg.get('word_timing', 'segmentation')}"
                )
        if not kept:
            continue
        u = units(kept)
        sec = np.array([x[0] for x in u])
        sp = np.array([x[1] for x in u])
        packed = np.array([x[2] for x in u])
        q = " / ".join(f"{np.percentile(sec, p):.1f}" for p in (50, 90, 99))
        long_h = " / ".join(f"{sp[sec > t].sum() / 3600:.2f}" for t in (10, 30, 60))
        nb = len(units(before)) if before else 0
        print(
            f"{d.name}\tunits {len(u)} (before alignment {nb}); length p50/p90/p99 "
            f"{q} s, max {sec.max():.1f} s; speech {sp.sum() / 3600:.1f} h, in units "
            f">10/>30/>60 s {long_h} h; speech in packs (packed, mean) "
            f"{100 * sp[packed].sum() / max(1e-9, sp.sum()):.0f} %; "
            f"{'; '.join(sorted(configs)) or 'no align_config.json'}\n"
            f"\treader: {reader_doc(d.name)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
