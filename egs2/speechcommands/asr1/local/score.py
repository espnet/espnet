#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Yifan Peng)

import argparse
import os
import os.path

parser = argparse.ArgumentParser(description="Calculate classification accuracy.")
parser.add_argument("--wer_dir", type=str, help="folder containing hyp.trn and ref.trn")
args = parser.parse_args()


with open(os.path.join(args.wer_dir, "hyp.trn"), "r") as f:
    hyp_dict = {ln.split()[1]: ln.split()[0] for ln in f.readlines()}
with open(os.path.join(args.wer_dir, "ref.trn"), "r") as f:
    ref_dict = {ln.split()[1]: ln.split()[0] for ln in f.readlines()}

n_correct = 0
n_samples = 0
for sample_id in ref_dict:
    n_samples += 1
    if ref_dict[sample_id] == hyp_dict[sample_id]:
        n_correct += 1

with open(os.path.join(args.wer_dir, "..", "accuracy.csv"), "w") as f:
    f.write("total,correct,accuracy\n")
    f.write(f"{n_samples},{n_correct},{n_correct/n_samples}\n")
