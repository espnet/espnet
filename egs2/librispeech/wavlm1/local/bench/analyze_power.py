#!/usr/bin/env python3
"""Summarise and plot nvidia-smi power samples from a benchmark run.

Usage: analyze_power.py <power_csv> [--target-low 550] [--target-high 650]
                                    [--out-prefix local/bench/power/curve]
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime


def load(path):
    per_gpu = defaultdict(list)          # idx -> [(t, watts, util)]
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("timestamp"):
                continue
            try:
                t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                idx = int(row[1])
                w = float(row[2])
                util = float(row[3])
            except (ValueError, IndexError):
                continue
            per_gpu[idx].append((t, w, util))
    return per_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target-low", type=float, default=550.0)
    ap.add_argument("--target-high", type=float, default=650.0)
    ap.add_argument("--tdp", type=float, default=700.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="ignore this leading fraction when reporting steady state")
    ap.add_argument("--warmup-sec", type=float, default=None,
                    help="ignore this many leading SECONDS instead of a fraction. "
                         "Use this to compare runs of different total length: a "
                         "fixed fraction discards wildly different amounts of "
                         "start-up on an 8-minute run vs a 27-minute one.")
    ap.add_argument("--auto-steady", action="store_true",
                    help="report only the longest contiguous training plateau")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    per_gpu = load(args.csv)
    if not per_gpu:
        raise SystemExit(f"no usable samples in {args.csv}")

    t0 = min(v[0][0] for v in per_gpu.values())
    # node total per timestamp bucket (align on whole seconds)
    buckets = defaultdict(dict)
    for idx, rows in per_gpu.items():
        for t, w, u in rows:
            buckets[int((t - t0).total_seconds())][idx] = (w, u)

    n_gpu = len(per_gpu)
    series = []
    for sec in sorted(buckets):
        vals = buckets[sec]
        if len(vals) != n_gpu:          # drop partial samples
            continue
        series.append((sec,
                       sum(w for w, _ in vals.values()),
                       sum(w for w, _ in vals.values()) / n_gpu,
                       sum(u for _, u in vals.values()) / n_gpu))
    if not series:
        raise SystemExit("no complete samples across all GPUs")

    if args.auto_steady:
        # Isolate the training plateau: keep the longest contiguous run of
        # samples above half the run's p95. Short benchmark bursts spend a large
        # fraction of wall time in start-up and teardown, which would otherwise
        # dominate the mean and make points of different length incomparable.
        srt_all = sorted(s[2] for s in series)
        thr = 0.5 * srt_all[min(len(srt_all) - 1, int(0.95 * len(srt_all)))]
        best, cur = [], []
        for smp in series:
            if smp[2] >= thr:
                cur.append(smp)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = []
        if len(cur) > len(best):
            best = cur
        steady = best or series
    elif args.warmup_sec is not None:
        steady = [s for s in series if s[0] >= args.warmup_sec] or series
    else:
        cut = int(len(series) * args.warmup_frac)
        steady = series[cut:] or series
    per = [s[2] for s in steady]
    util = [s[3] for s in steady]
    dur = steady[-1][0] - steady[0][0]

    def pct(lo, hi):
        return 100.0 * sum(1 for p in per if lo <= p <= hi) / len(per)

    srt = sorted(per)
    q = lambda p: srt[min(len(srt) - 1, int(p * len(srt)))]
    energy_kwh = sum(s[1] for s in steady) * (dur / max(len(steady) - 1, 1)) / 3.6e6

    print(f"  file                : {args.csv}")
    print(f"  GPUs                : {n_gpu}   samples: {len(series)}"
          f"   steady window: {dur/60:.1f} min")
    print(f"  per-GPU power  mean : {sum(per)/len(per):7.1f} W"
          f"   ({100*sum(per)/len(per)/args.tdp:.0f}% of {args.tdp:.0f} W TDP)")
    print(f"                 p50  : {q(0.50):7.1f} W")
    print(f"                 p05  : {q(0.05):7.1f} W    p95: {q(0.95):7.1f} W")
    print(f"                 min  : {min(per):7.1f} W    max: {max(per):7.1f} W")
    print(f"  node total     mean : {sum(s[1] for s in steady)/len(steady):7.1f} W")
    print(f"  GPU utilisation mean: {sum(util)/len(util):7.1f} %")
    print(f"  TIME IN {args.target_low:.0f}-{args.target_high:.0f} W"
          f"    : {pct(args.target_low, args.target_high):7.1f} %  <-- target band")
    print(f"  time below {args.target_low:.0f} W   : {pct(0, args.target_low-0.001):7.1f} %")
    print(f"  time above {args.target_high:.0f} W   : {pct(args.target_high+0.001, 1e9):7.1f} %")
    print(f"  energy (steady win) : {energy_kwh:7.3f} kWh over {n_gpu} GPUs")

    if args.out_prefix:
        # ASCII curve so it is readable without a plotting backend
        txt = args.out_prefix + ".txt"
        H, W = 22, 100
        lo, hi = 0.0, max(args.tdp, max(per)) * 1.02
        step = max(1, len(series) // W)
        cols = [series[i][2] for i in range(0, len(series), step)][:W]
        with open(txt, "w") as f:
            f.write(f"per-GPU power draw over time  ({args.csv})\n")
            f.write(f"target band {args.target_low:.0f}-{args.target_high:.0f} W"
                    f"   TDP {args.tdp:.0f} W\n\n")
            for r in range(H, 0, -1):
                y = lo + (hi - lo) * r / H
                ylo = lo + (hi - lo) * (r - 1) / H
                mark = "|"
                if ylo <= args.target_high <= y or ylo <= args.target_low <= y:
                    mark = "+"
                line = "".join("#" if v >= ylo else " " for v in cols)
                f.write(f"{y:6.0f} W {mark}{line}\n")
            f.write(" " * 9 + "+" + "-" * len(cols) + "\n")
            f.write(" " * 10 + f"0{' ' * (len(cols) - 12)}{dur/60:.0f} min\n")
        print(f"  ascii curve written : {txt}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot([s[0] / 60 for s in series], [s[2] for s in series],
                    lw=0.8, color="#1f77b4", label="per-GPU power")
            ax.axhspan(args.target_low, args.target_high, color="green", alpha=0.12,
                       label=f"target {args.target_low:.0f}-{args.target_high:.0f} W")
            ax.axhline(args.tdp, color="red", ls="--", lw=1,
                       label=f"TDP {args.tdp:.0f} W")
            ax.set_xlabel("minutes"); ax.set_ylabel("watts per GPU")
            ax.set_title("WavLM large - GPU power draw")
            ax.set_ylim(0, args.tdp * 1.05); ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(args.out_prefix + ".png", dpi=130)
            print(f"  png written         : {args.out_prefix}.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
