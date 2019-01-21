#!/usr/bin/env python
import argparse
import os
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--snapshots", required=True, type=str, nargs="+")
parser.add_argument("--out", required=True, type=str)
parser.add_argument("--num", default=10, type=int)
args = parser.parse_args()

last = sorted(args.snapshots, key=os.path.getmtime)
last = last[-args.num:]
print("average over", last)
avg = None

# sum
for path in last:
    states = torch.load(path, map_location=torch.device("cpu"))["model"]
    if avg is None:
        avg = states
    else:
        for k in avg.keys():
            avg[k] += states[k]

# average
for k in avg.keys():
    # print(k)
    if avg[k] is not None:
        avg[k] /= args.num

torch.save(avg, args.out)
