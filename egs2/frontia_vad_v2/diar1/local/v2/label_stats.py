#!/usr/bin/env python3
"""Label statistics of corpus dumps, side by side for two label versions.

    python3 local/v2/label_stats.py --old dump/v2 --new dump/v2_1 [corpus ...]

For every corpus (default: those in --new) and version: the recordings kept
after alignment, their hours and speech hours, the speech share, and the
pauses inside recordings (gaps between consecutive speech segments of the
espnet_rttm): count per speech hour, median, and the shares under 0.5 s and
over 2 s. Reads only the dumps.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splits import corpus_dirs, corpus_dumps, manifest_rows  # noqa: E402


def stats(corpus_dir: Path):
    recs = hours = speech = 0.0
    pauses = []
    for _, dump in corpus_dumps(corpus_dir):
        for r in manifest_rows(dump):
            recs += 1
            hours += float(r["seconds"]) / 3600
            speech += float(r["speech_seconds"]) / 3600
        segs = {}
        for line in (dump / "espnet_rttm").open():
            p = line.split()
            if p[0] == "SPEAKER":
                segs.setdefault(p[1], []).append((int(p[3]), int(p[4])))
        for s in segs.values():
            s.sort()
            pauses += [(b[0] - a[1]) / 16000 for a, b in zip(s, s[1:])]
    g = np.array(pauses) if pauses else np.zeros(0)
    return dict(
        recordings=int(recs),
        hours=hours,
        speech_hours=speech,
        speech_share=100 * speech / max(hours, 1e-9),
        pauses_per_speech_hour=len(g) / max(speech, 1e-9),
        pause_median=float(np.median(g)) if len(g) else float("nan"),
        pauses_under_05=100 * float((g < 0.5).mean()) if len(g) else float("nan"),
        pauses_over_2=100 * float((g > 2.0).mean()) if len(g) else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("corpora", nargs="*")
    a = ap.parse_args()
    corpora = a.corpora or [d.name for d in corpus_dirs(a.new)]
    print(
        "corpus\tversion\trecordings\thours\tspeech_h\tspeech_%\t"
        "pauses/speech_h\tpause_median_s\tpauses<0.5s_%\tpauses>2s_%"
    )
    for c in corpora:
        for name, root in (("old", a.old), ("new", a.new)):
            if not (root / c).is_dir():
                continue
            s = stats(root / c)
            print(
                f"{c}\t{name}\t{s['recordings']}\t{s['hours']:.1f}\t"
                f"{s['speech_hours']:.1f}\t{s['speech_share']:.1f}\t"
                f"{s['pauses_per_speech_hour']:.0f}\t{s['pause_median']:.2f}\t"
                f"{s['pauses_under_05']:.0f}\t{s['pauses_over_2']:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
