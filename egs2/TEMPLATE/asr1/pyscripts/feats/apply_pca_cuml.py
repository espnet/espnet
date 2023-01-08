#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#               2022 Johns Hopkins Univertiy (Author: Dongji Gao)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import math
import numpy as np
import tqdm
from shutil import copyfile

import pickle
from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    # fmt: off
    parser.add_argument('feats_scp', help='features scp file')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--pca_path', type=str, help='pca location. will append _A.npy and _b.npy', required=True)
    parser.add_argument('--batch_size', type=int, default=2048000, help='batch size')
    parser.add_argument('--dim', type=int, default=512, help="pca dimension")
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    model_file = f"{args.pca_path}/pca_{args.dim}.pkl"
    pca = pickle.load(open(model_file, "rb"))

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

    save_path = osp.join(args.save_dir, args.split)

    npaa = NpyAppendArray(save_path + ".npy")

    batches = math.ceil(features.shape[0] / args.batch_size)

    for b in tqdm.trange(batches):
        start = b * args.batch_size
        end = start + args.batch_size
        x = features[start:end]
        x = pca.transform(x)
        npaa.append(x)


if __name__ == "__main__":
    main()
