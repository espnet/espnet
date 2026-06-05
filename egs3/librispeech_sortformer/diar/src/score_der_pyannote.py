"""Score long-form DER from hyp/ref RTTM directories with pyannote.metrics.

Run in the pyannote conda env. Computes the standard collar-based Diarization
Error Rate (default collar 0.25 s) aggregated over all sessions, plus
miss / false-alarm / confusion components.
"""

import argparse
import glob
import os

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate


def load_rttm(path):
    ann = Annotation(uri=os.path.splitext(os.path.basename(path))[0])
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0] != "SPEAKER":
                continue
            start, dur, spk = float(p[3]), float(p[4]), p[7]
            ann[Segment(start, start + dur)] = spk
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with hyp/ and ref/ RTTMs")
    ap.add_argument("--collar", type=float, default=0.25)
    ap.add_argument("--skip_overlap", action="store_true")
    args = ap.parse_args()

    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    ref_files = sorted(glob.glob(os.path.join(args.dir, "ref", "*.rttm")))
    n = 0
    for rf in ref_files:
        name = os.path.basename(rf)
        hf = os.path.join(args.dir, "hyp", name)
        if not os.path.exists(hf):
            print("missing hyp for", name)
            continue
        ref, hyp = load_rttm(rf), load_rttm(hf)
        der = metric(ref, hyp, detailed=False)
        print(f"{name[:-5]:12s} DER={100*der:6.2f}%")
        n += 1
    res = abs(metric)
    comp = metric[:]
    total = comp["total"]
    print("=" * 50)
    print(f"sessions={n}  collar={args.collar}  skip_overlap={args.skip_overlap}")
    print(f"OVERALL DER = {100*res:.2f}%")
    print(f"  miss        = {100*comp['missed detection']/total:.2f}%")
    print(f"  false alarm = {100*comp['false alarm']/total:.2f}%")
    print(f"  confusion   = {100*comp['confusion']/total:.2f}%")


if __name__ == "__main__":
    main()
