#!/usr/bin/env python3
"""Post-hoc convert an ESPnet train.log into TensorBoard tfevents.

Walks a train log for `Nepoch results:` lines, splits the per-epoch
summary into [train] / [valid] halves, and emits one scalar per metric
under <out_dir>/{train,valid}/. Useful when training ran under the
DeepSpeed trainer (no TB hook yet) so logs are the only signal.
"""

import argparse
import os
import re
import sys

from torch.utils.tensorboard import SummaryWriter

EPOCH_LINE = re.compile(r"(\d+)epoch results:")
PAIR = re.compile(r"(\w+)=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")


def split_train_valid(rest):
    if "[valid]" in rest:
        train_part, valid_part = rest.split("[valid]", 1)
    else:
        train_part, valid_part = rest, ""
    train_part = train_part.split("[train]", 1)[-1]
    return train_part, valid_part


def parse_kvs(s):
    out = {}
    for m in PAIR.finditer(s):
        k, v = m.group(1), m.group(2)
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_log")
    ap.add_argument("out_dir")
    args = ap.parse_args()

    train_w = SummaryWriter(os.path.join(args.out_dir, "train"))
    valid_w = SummaryWriter(os.path.join(args.out_dir, "valid"))

    seen_epochs = set()
    with open(args.train_log) as f:
        for line in f:
            m = EPOCH_LINE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            if ep in seen_epochs:
                continue  # rotation / resume: keep first
            seen_epochs.add(ep)
            rest = line[m.end() :]
            tr, va = split_train_valid(rest)
            for k, v in parse_kvs(tr).items():
                train_w.add_scalar(k, v, ep)
            for k, v in parse_kvs(va).items():
                valid_w.add_scalar(k, v, ep)
            print(f"epoch {ep}: train+valid scalars written", flush=True)

    train_w.flush()
    train_w.close()
    valid_w.flush()
    valid_w.close()
    print(f"Done. {len(seen_epochs)} epochs → {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
