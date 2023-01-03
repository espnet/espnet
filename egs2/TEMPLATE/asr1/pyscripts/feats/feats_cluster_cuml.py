#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#               2022 Dongji Gao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import os.path as osp
import random
import tqdm


from collections import namedtuple

import pickle
import numpy as np
from cuml.cluster import KMeans


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    parser.add_argument("feats_scp", help="location of feature scp files")
    parser.add_argument("--save_dir", help="where to save the output", required=True)
    parser.add_argument(
        "--num_clusters", type=int, default=128, help="number of clusters"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    n_clusters = args.num_clusters
    save_path = osp.join(args.save_dir, f"CLUS{n_clusters}")
    os.makedirs(save_path, exist_ok=True)

    feats_list = []
    with open(args.feats_scp, "r") as f_scp:
        for line in f_scp.readlines():
            _, feat_file = line.split()
            feat = np.load(feat_file)
            feats_list.append(feat)

        feats = np.concatenate(feats_list)

        gc.collect()

    print("Computing kmeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=1).fit(feats)

    np.save(osp.join(save_path, "centroids"), kmeans.cluster_centers_)
    pickle.dump(kmeans, open(f"{save_path}/kmeans_{n_clusters}.pkl", "wb"))
    del kmeans
    gc.collect()


if __name__ == "__main__":
    main()
