#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Adapted from fairseq/examples/wav2vec/unsupervised/scripts/apply_pca.py
#         to fit the scp data format
# This source code is licensed under the MIT license in
# https://github.com/facebookresearch/fairseq

import argparse
import math
import os
import os.path as osp
from shutil import copyfile

import numpy as np
import torch
import tqdm
from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    parser.add_argument("feats_scp", help="features scp file")
    parser.add_argument("--split", help="which split to read", required=True)
    parser.add_argument("--save_dir", help="where to save the output", required=True)
    parser.add_argument(
        "--pca_path",
        type=str,
        help="pca location. will append _A.npy and _b.npy",
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=2048000, help="batch size")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"data path: {args.feats_scp}")
    os.makedirs(args.save_dir, exist_ok=True)

    x = []
    length_file = f"{args.save_dir}/{args.split}.lengths"

    with open(args.feats_scp, "r") as f_scp, open(length_file, "w") as lf:
        for line in f_scp.readlines():
            utt_id, feats_path = line.split()
            feats = np.load(feats_path)
            x.append(feats)
            lf.write(f"{utt_id} {feats.shape[0]}\n")

    features = np.vstack(x)

    pca_A = torch.from_numpy(np.load(args.pca_path + "_A.npy")).cuda()
    pca_b = torch.from_numpy(np.load(args.pca_path + "_b.npy")).cuda()

    save_path = osp.join(args.save_dir, args.split)

    npaa = NpyAppendArray(save_path + ".npy")

    batches = math.ceil(features.shape[0] / args.batch_size)

    with torch.no_grad():
        for b in tqdm.trange(batches):
            start = b * args.batch_size
            end = start + args.batch_size
            x = torch.from_numpy(features[start:end]).cuda()
            x = torch.matmul(x, pca_A) + pca_b
            npaa.append(x.cpu().numpy())


if __name__ == "__main__":
    main()
