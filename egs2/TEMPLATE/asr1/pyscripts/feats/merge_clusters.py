#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#               Dongji Gao (2022)
#
# Adapted from fairseq/examples/wav2vec/unsupervised/\
#                  scripts/merge_clusters.py
#         to fit the scp data format
# This source code is licensed under the MIT license in
# https://github.com/facebookresearch/fairseq

import argparse
import os
import os.path as osp
import random
from shutil import copyfile

import numpy as np
import torch
import tqdm
from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    parser.add_argument("source", help="directory with features")
    parser.add_argument("--split", help="which split to read", required=True)
    parser.add_argument("--save_dir", help="where to save the output", required=True)
    parser.add_argument("--cluster_dir", help="where the clusters are")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "sample"],
        help="how to pool",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split)
    cluster_path = osp.join(args.cluster_dir, args.split + ".cluster")

    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r")
    sizes = []
    offsets = []
    offset = 0
    with open(source_path + ".lengths", "r") as len_f:
        for line in len_f.readlines():
            utt_id, length = line.split()
            length = int(length)
            sizes.append(length)
            offsets.append(offset)
            offset += length

    clusters = []
    with open(cluster_path, "r") as cf:
        for line in cf.readlines():
            line_list = line.split()
            items = line_list[1:]
            items = list(map(int, items))
            clusters.append(items)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy")

    def merge(feats, clust):
        feats = torch.from_numpy(feats.copy())
        clust = torch.LongTensor(clust)
        _, counts = clust.unique_consecutive(return_counts=True)
        curr = 0

        merged = []
        for c in counts:
            c = c.item()
            start = curr
            end = curr + c
            curr += c
            if args.pooling == "mean":
                new_x = feats[start:end].mean(dim=0)
            elif args.pooling == "sample":
                new_x = feats[start + int(random.random() * c)]
            else:
                raise NotImplementedError()
            merged.append(new_x)

        return torch.stack(merged, dim=0).numpy()

    with open(save_path + ".lengths_pure", "w") as l_f:
        for size, offset, clust in tqdm.tqdm(
            zip(sizes, offsets, clusters), total=len(sizes)
        ):
            end = size + offset
            feats = features[offset:end]
            feats = merge(feats, clust)
            print(len(feats), file=l_f)
            npaa.append(feats)


if __name__ == "__main__":
    main()
