#!/usr/bin/env python3
"""Voice activity detection error of diar.sh stage 6 outputs (md-eval).

    python3 local/v2/score_vad.py score <diarized_<set> dir> <set dir> <out dir> \
        --md_eval md-eval.pl --thresholds 0.3 ... --medians 1 11 [--median_mode causal]
    python3 local/v2/score_vad.py summarize <out csv> <exp dir> [...] \
        [--tune_set ami_mh_dev] [--scoring scoring]

score: for each threshold and odd median window (frames), the frames of the
speech probability (diarize.scp, one column) above the threshold are
median-filtered, centred (scipy medfilt, (M-1)/2 future frames) or causal
(the past M frames, zeros before the start), and become one-speaker RTTM
segments, frame i covering [offset + (i - 0.5) shift, offset + (i + 0.5)
shift) with shift 128 and offset 256 samples (the model's 512-sample frames
every 128 samples, not centred). md-eval.pl -c 0 scores them against the
set's rttm over each whole recording (a UEM from utt2num_samples), or over the
regions of --uem (DIHARD style; recordings it does not list are scored whole). Writes
result_th<T>_med<M>_collar0.0, as scripts/utils/score_der.sh names them.
Recording ids with dots get underscores for md-eval, which cuts UEM names
at the first dot.

summarize: for each exp directory, the (threshold, median) with the lowest
detection error on --tune_set is applied to every set (and each set's own
best is reported next to it); detection error = (missed + false alarm
speech) / reference speech. Writes one csv row per exp and set, and prints
the macro average over the sets and the average weighted by speech hours.
"""

import argparse
import collections
import csv
import re
import subprocess
from pathlib import Path

import numpy as np

FS = 16000
SHIFT, OFFSET = 128 / FS, 256 / FS
RESULT_RE = re.compile(r"result_th([0-9.]+)_med(\d+)_collar")


def causal_median(on, m):
    """Median of the binary on[i - m + 1 .. i] (zeros before the start), m odd."""
    count = np.cumsum(np.r_[0.0, on])
    i = np.arange(len(on))
    window = count[i + 1] - count[np.maximum(i + 1 - m, 0)]
    return (2 * window > m).astype(np.float32)


def segments(prob, thr, med, mode="centred", shift=SHIFT, offset=OFFSET):
    """(start, duration) in seconds of the speech of one recording."""
    from scipy.signal import medfilt

    on = (np.asarray(prob, dtype=np.float32) > thr).astype(np.float32)
    if med > 1:
        on = medfilt(on, med) if mode == "centred" else causal_median(on, med)
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    for s, e in zip(edges[0::2], edges[1::2]):
        start = max(0.0, offset + (s - 0.5) * shift)
        yield start, offset + (e - 0.5) * shift - start


def md_id(rid):
    return rid.replace(".", "_")


def write_uem(path, rids, num_samples, uem=None):
    """The md-eval UEM: each recording whole, or the regions of a given UEM file.

    uem: lines "<recording> <channel> <start s> <end s>" (the DIHARD all.uem); only
    its regions are scored (md-eval ignores reference and system speech outside
    them). A recording it does not list is scored whole.
    """
    regions = collections.defaultdict(list)
    if uem:
        for line in open(uem):
            p = line.split()
            if len(p) >= 4 and p[0] in rids:
                regions[p[0]].append((float(p[2]), float(p[3])))
    with open(path, "w") as f:
        for rid in sorted(rids):
            for s, e in regions.get(rid) or [(0.0, num_samples[rid] / FS)]:
                f.write(f"{md_id(rid)} 1 {s:.6f} {e:.6f}\n")


def cmd_score(a):
    inf, data, out = Path(a.inference_dir), Path(a.data_dir), Path(a.out_dir)
    if any(m % 2 == 0 for m in a.medians):
        raise SystemExit(f"median windows must be odd: {a.medians}")
    out.mkdir(parents=True, exist_ok=True)
    probs = {}
    for line in (inf / "diarize.scp").open():
        rid, path = line.split(maxsplit=1)
        p = np.load(path.strip())
        probs[rid] = p[:, 0] if p.ndim == 2 else p
    num_samples = {
        x.split()[0]: int(x.split()[1]) for x in (data / "utt2num_samples").open()
    }
    refs = set()
    with (data / "rttm").open() as fi, (out / "ref.md-eval.rttm").open("w") as fo:
        for line in fi:
            p = line.split()
            if p and p[0] == "SPEAKER":
                refs.add(p[1])
                p[1] = md_id(p[1])
                fo.write(" ".join(p) + "\n")
    missing = sorted(refs - set(probs))
    if missing:
        raise SystemExit(
            f"{len(missing)} reference recordings have no output: {missing[:3]}"
        )
    write_uem(out / "uem.md-eval", probs, num_samples, a.uem)
    for thr in a.thresholds:
        for med in a.medians:
            hyp = out / f"hyp_{thr}_{med}.rttm"
            with hyp.open("w") as f:
                for rid, prob in sorted(probs.items()):
                    for start, dur in segments(prob, float(thr), med, a.median_mode):
                        f.write(
                            f"SPEAKER {md_id(rid)} 1 {start:.3f} {dur:.3f} "
                            "<NA> <NA> speech <NA> <NA>\n"
                        )
            with (out / f"result_th{thr}_med{med}_collar0.0").open("w") as f:
                subprocess.run(
                    [a.md_eval, "-c", "0.0", "-r", str(out / "ref.md-eval.rttm")]
                    + ["-s", str(hyp), "-u", str(out / "uem.md-eval")],
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            hyp.unlink()


def read_md_eval(path):
    text = Path(path).read_text()
    vals = []
    for name in ("SCORED SPEAKER TIME", "MISSED SPEAKER TIME", "FALARM SPEAKER TIME"):
        m = re.search(rf"{name}\s*=\s*([0-9.]+)\s*secs", text)
        if not m:
            return None
        vals.append(float(m.group(1)))
    return vals


def err(r):
    scored, miss, fa = r
    return 100 * (miss + fa) / scored


def name_of(exp: Path) -> str:
    """The model's name: the exp directory, or its parent for <name>/diar_<tag>."""
    return exp.parent.name if exp.name.startswith("diar_") else exp.name


def cmd_summarize(a):
    rows = []
    for exp in map(Path, a.exp_dirs):
        sets = {}
        for sdir in sorted(exp.glob(f"diarized_*/{a.scoring}")):
            res = {}
            for path in sdir.glob("result_th*_med*_collar0.0"):
                m = RESULT_RE.match(path.name)
                r = read_md_eval(path)
                if m and r:
                    res[(m.group(1), int(m.group(2)))] = r
            if res:
                sets[sdir.parent.name[len("diarized_") :]] = res
        if a.tune_set not in sets:
            print(f"skip {exp}: no {a.tune_set}")
            continue
        tune = sets[a.tune_set]
        setting = min(tune, key=lambda k: err(tune[k]))
        devs = []
        for name, res in sorted(sets.items()):
            own = min(res, key=lambda k: err(res[k]))
            scored, miss, fa = res[setting]
            devs.append((err(res[setting]), scored))
            rows.append(
                dict(
                    model=name_of(exp),
                    set=name,
                    speech_h=round(scored / 3600, 2),
                    deter=round(err(res[setting]), 2),
                    fa=round(100 * fa / scored, 2),
                    miss=round(100 * miss / scored, 2),
                    threshold=setting[0],
                    median=setting[1],
                    own_best=round(err(res[own]), 2),
                    own_threshold=own[0],
                    own_median=own[1],
                )
            )
        macro = sum(d for d, _ in devs) / len(devs)
        weighted = sum(d * h for d, h in devs) / sum(h for _, h in devs)
        print(
            f"{name_of(exp)}: {len(devs)} sets, threshold {setting[0]}, median "
            f"{setting[1]} ({a.scoring}): macro DetER {macro:.2f}, "
            f"hour-weighted {weighted:.2f}"
        )
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("inference_dir")
    sc.add_argument("data_dir")
    sc.add_argument("out_dir")
    sc.add_argument("--md_eval", required=True)
    sc.add_argument("--thresholds", nargs="+", required=True)
    sc.add_argument("--medians", nargs="+", type=int, required=True)
    sc.add_argument("--median_mode", choices=["centred", "causal"], default="centred")
    sc.add_argument(
        "--uem", default=None, help="score these regions only (DIHARD all.uem)"
    )
    su = sub.add_parser("summarize")
    su.add_argument("out_csv")
    su.add_argument("exp_dirs", nargs="+")
    su.add_argument("--tune_set", default="ami_mh_dev")
    su.add_argument("--scoring", default="scoring", help="scoring or scoring_causal")
    a = ap.parse_args()
    cmd_score(a) if a.cmd == "score" else cmd_summarize(a)


if __name__ == "__main__":
    main()
