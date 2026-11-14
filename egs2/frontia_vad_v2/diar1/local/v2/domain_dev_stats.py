#!/usr/bin/env python3
"""Loss and accuracy per domain at each validation of a training run.

    python3 local/v2/domain_dev_stats.py exp/diar_<tag>
    python3 local/v2/domain_dev_stats.py --windows dump/raw/dev [--chunk 160000]

The model reports, for every domain c, valid_frac_<c>, valid_loss_num_<c> and
valid_acc_num_<c> (espnet2 ESPnetDiarizationModel, valid_category_file); the
loss of c is valid_loss_num_<c> / valid_frac_<c>, its accuracy
valid_acc_num_<c> / valid_frac_<c>, per look-ahead (suffix _la<frames>).
The values are read at full precision from the reporter state in
<exp>/checkpoint.pth (every finished epoch). train.log is not used: it prints
values between 0.001 and 1000 with three decimals, so the fraction of a small
domain (e.g. 0.006) keeps one significant digit and the ratios become
fractions such as 5/6. --windows instead counts the validation windows per
domain (chunks of --chunk samples of each recording, as the chunk iterator
cuts).
"""

import argparse
import collections
from pathlib import Path


def windows(dev: Path, chunk: int):
    cat = dict(line.split() for line in (dev / "utt2category").read_text().splitlines())
    count = collections.Counter()
    for line in (dev / "utt2num_samples").read_text().splitlines():
        rid, n = line.split()
        count[cat[rid]] += int(n) // chunk
    for c, n in sorted(count.items()):
        print(f"{c:18s} {n:8d} windows")
    print(f"{'total':18s} {sum(count.values()):8d} windows")


def epoch_stats(exp: Path):
    """{epoch: {key: value}} of the validations of a run, from checkpoint.pth."""
    import torch

    states = torch.load(exp / "checkpoint.pth", map_location="cpu", weights_only=False)
    return {
        int(e): dict(d.get("valid", {}))
        for e, d in sorted(states["reporter"]["stats"].items())
    }


def domain_table(stats):
    """[(domain, frac, {suffix: (loss, acc)})] of one validation's stats."""
    suffixes = sorted(
        {
            k.rsplit("_la", 1)[1]
            for k in stats
            if k.startswith("valid_frac_") and "_la" in k
        },
        key=int,
    )
    domains = sorted(
        {
            k[len("valid_frac_") :].rsplit("_la", 1)[0]
            for k in stats
            if k.startswith("valid_frac_")
        }
    )
    rows = []
    for c in domains:
        per = {}
        for s in suffixes:
            f = float(stats[f"valid_frac_{c}_la{s}"])
            per[s] = (
                float(stats[f"valid_loss_num_{c}_la{s}"]) / f if f else float("nan"),
                float(stats[f"valid_acc_num_{c}_la{s}"]) / f if f else float("nan"),
            )
        frac = (
            float(stats[f"valid_frac_{c}_la{suffixes[0]}"])
            if suffixes
            else float("nan")
        )
        rows.append((c, frac, per))
    return suffixes, rows


def report(exp: Path):
    for epoch, stats in epoch_stats(exp).items():
        suffixes, rows = domain_table(stats)
        if not rows:
            continue
        print(f"== epoch {epoch}")
        print(
            "domain             frac    "
            + "".join(f"  loss_la{s}  acc_la{s}" for s in suffixes)
        )
        for c, frac, per in rows:
            print(
                f"{c:18s} {frac:.5f}"
                + "".join(f"  {per[s][0]:9.4f}  {per[s][1]:7.4f}" for s in suffixes)
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exp", nargs="?", type=Path, help="exp/diar_<tag>")
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--chunk", type=int, default=160000)
    a = ap.parse_args()
    if a.windows:
        windows(a.windows, a.chunk)
    else:
        report(a.exp)


if __name__ == "__main__":
    main()
