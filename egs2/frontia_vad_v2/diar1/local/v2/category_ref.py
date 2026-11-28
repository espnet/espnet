#!/usr/bin/env python3
"""A category-aware reference of a test set: speech and don't-care from a label map.

    python3 local/v2/category_ref.py --set dump/raw/ami_dev --labels data/ami_dev \
        --map conf/label_map/eval_vocal_ignore.yaml --out dump/raw/ami_dev_cat

--set is a scoring set (rttm in seconds, utt2num_samples, wav.scp, ...). --labels
holds its category labels in the formats of local/vad_data_utils.py: categories
("<rec> <start> <end> <category> <tags> <layer>"), and for a set built from a Kaldi
directory also segments and segment_tags (the speech category of each segment);
it may be the set directory itself (bench_test_sets.py writes categories there).

The speech of the set's rttm keeps its category (that of the segment it overlaps
most, else --speech_category); the categories add the annotated sounds and the
details. Under the map (local/v2/categories.py; a binary map, whose class is
"speech"), the reference speech is the union of everything mapped to speech, and
the don't-care regions the union of everything mapped to ignore, less the
speech. --out gets the set's files (linked), the new rttm, and a uem with each
recording (or the set's own uem regions) less the don't-care regions, for
score_vad.py score --uem. With the map conf/label_map/v2x.yaml the rttm has the
set's speech and the uem nothing left out.
"""

import argparse
import collections
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from categories import DETAIL, load_map, lookup  # noqa: E402
from vad_dump import merge_intervals  # noqa: E402

FS = 16000
Interval = Tuple[float, float]


def subtract(a: List[Interval], b: List[Interval]) -> List[Interval]:
    """The parts of the intervals a outside the intervals b (both merged)."""
    out = []
    for s, e in a:
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def read_rttm(path: Path) -> Dict[str, List[Interval]]:
    out = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            out[f[1]].append((float(f[3]), float(f[3]) + float(f[4])))
    return {r: merge_intervals(v) for r, v in out.items()}


def read_labels(d: Path):
    """(segment categories {rec: [(start, end, category, tags)]}, categories)."""
    seg_cat = collections.defaultdict(list)
    if (d / "segments").exists() and (d / "segment_tags").exists():
        tags = {}
        for line in (d / "segment_tags").open():
            utt, c, t = line.split()
            tags[utt] = (c, t)
        for line in (d / "segments").open():
            utt, rec, a, b = line.split()
            if utt in tags:
                seg_cat[rec].append((float(a), float(b), *tags[utt]))
    cats = collections.defaultdict(list)
    if (d / "categories").exists():
        for line in (d / "categories").open():
            rec, a, b, c, t, layer = line.split()
            cats[rec].append((float(a), float(b), c, t, int(layer)))
    return seg_cat, cats


def speech_category(segs, s: float, e: float, default: str) -> str:
    """Category of the labelled segment that overlaps [s, e) most, else default."""
    best, cat = 0.0, default
    for a, b, c, _ in segs:
        o = min(b, e) - max(a, s)
        if o > best:
            best, cat = o, c
    return cat


def reference(speech, seg_cat, cats, m, default):
    """(speech, ignore) intervals of one recording under the map m."""
    keep, ignore = [], []
    items = [(s, e, speech_category(seg_cat, s, e, default), 1) for s, e in speech]
    items += [(a, b, c, layer) for a, b, c, _, layer in cats]
    for a, b, c, layer in items:
        if layer == DETAIL and not m.details:
            continue
        target = lookup(m, c)
        if target == "speech":
            keep.append((a, b))
        elif target == "ignore":
            ignore.append((a, b))
    keep = merge_intervals(keep)
    return keep, subtract(merge_intervals(ignore), keep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--speech_category", default="speech/utterance")
    a = ap.parse_args(argv)
    m = load_map(a.map)
    if m.classes != ("speech",):
        raise SystemExit(f"{a.map}: a reference needs a binary map, not {m.classes}")
    nums = {
        r: int(n)
        for r, n in (
            line.split() for line in (a.set / "utt2num_samples").open() if line.strip()
        )
    }
    speech = read_rttm(a.set / "rttm")
    seg_cat, cats = read_labels(a.labels)
    set_uem = collections.defaultdict(list)
    if (a.set / "uem").exists():
        for line in (a.set / "uem").open():
            r, _, s, e = line.split()
            set_uem[r].append((float(s), float(e)))
    a.out.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(a.set):
        if name not in ("rttm", "uem", ".done") and not (a.out / name).exists():
            (a.out / name).symlink_to((a.set / name).resolve())
    rttm, uem = [], []
    hours = collections.Counter()
    for rec in sorted(nums):
        dur = nums[rec] / FS
        sp, ig = reference(
            speech.get(rec, []),
            seg_cat.get(rec, []),
            cats.get(rec, []),
            m,
            a.speech_category,
        )
        sp = [(max(0.0, s), min(dur, e)) for s, e in sp if min(dur, e) > max(0.0, s)]
        for s, e in sp:
            rttm.append(
                f"SPEAKER {rec} 1 {s:.3f} {e - s:.3f} <NA> <NA> speech <NA> <NA>\n"
            )
        scored = subtract(merge_intervals(set_uem.get(rec) or [(0.0, dur)]), ig)
        uem += [f"{rec} 1 {s:.3f} {min(e, dur):.3f}\n" for s, e in scored if e > s]
        hours["speech_old"] += sum(e - s for s, e in speech.get(rec, []))
        hours["speech_new"] += sum(e - s for s, e in sp)
        hours["ignored"] += sum(min(e, dur) - s for s, e in ig if min(e, dur) > s)
        hours["recordings_with_labels"] += rec in seg_cat or rec in cats
    (a.out / "rttm").write_text("".join(sorted(rttm)))
    (a.out / "uem").write_text("".join(uem))
    (a.out / "category_map.txt").write_text(f"{a.map}\nlabels {a.labels}\n")
    print(
        f"{a.out}: {len(nums)} recordings ({int(hours['recordings_with_labels'])} with "
        f"labels); speech {hours['speech_old'] / 3600:.2f} h -> "
        f"{hours['speech_new'] / 3600:.2f} h, "
        f"don't-care {hours['ignored'] / 3600:.2f} h"
    )


if __name__ == "__main__":
    main()
