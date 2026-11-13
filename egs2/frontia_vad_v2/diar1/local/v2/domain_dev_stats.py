#!/usr/bin/env python3
"""Loss and accuracy per domain at each validation, from a training log.

    python3 local/v2/domain_dev_stats.py exp/diar_<tag>/train.log
    python3 local/v2/domain_dev_stats.py --windows dump/raw/dev [--chunk 160000]

The model reports, for every domain c, valid_frac_<c>, valid_loss_num_<c> and
valid_acc_num_<c> (espnet2 ESPnetDiarizationModel, valid_category_file); the
loss of c is valid_loss_num_<c> / valid_frac_<c>, its accuracy
valid_acc_num_<c> / valid_frac_<c>, per look-ahead (suffix _la<frames>) and
as their mean. --windows instead counts the validation windows per domain
(chunks of --chunk samples of each recording, as the chunk iterator cuts).
"""

import argparse
import collections
import re
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


def from_log(log: Path):
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"(\d+)epoch results: .*\[valid\] (.*?)(, \[|$)", line)
        if not m:
            continue
        stats = dict(kv.split("=", 1) for kv in m.group(2).split(", ") if "=" in kv)
        suffixes = sorted(
            {
                k.rsplit("_la", 1)[1]
                for k in stats
                if "_la" in k and k.startswith("valid_frac_")
            }
        )
        domains = sorted(
            {
                k[len("valid_frac_") :].rsplit("_la", 1)[0]
                for k in stats
                if k.startswith("valid_frac_")
            }
        )
        print(f"== epoch {m.group(1)}")
        head = "domain             frac " + "".join(
            f"  loss_la{s}  acc_la{s}" for s in suffixes
        )
        print(head)
        for c in domains:
            row = f"{c:18s} {float(stats[f'valid_frac_{c}']):.3f}"
            for s in suffixes:
                f = float(stats[f"valid_frac_{c}_la{s}"])
                loss = (
                    float(stats[f"valid_loss_num_{c}_la{s}"]) / f if f else float("nan")
                )
                acc = (
                    float(stats[f"valid_acc_num_{c}_la{s}"]) / f if f else float("nan")
                )
                row += f"  {loss:9.4f}  {acc:7.4f}"
            print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?", type=Path)
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--chunk", type=int, default=160000)
    a = ap.parse_args()
    if a.windows:
        windows(a.windows, a.chunk)
    else:
        from_log(a.log)


if __name__ == "__main__":
    main()
