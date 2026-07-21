#!/usr/bin/env python3
#
# Copyright 2022 Johns Hopkins University (Author: Dongji Gao)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import os
import os.path as osp
import pickle
import random
from collections import namedtuple

import numpy as np
import tqdm
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(feats)

    np.save(osp.join(save_path, "centroids"), kmeans.cluster_centers_)
    pickle.dump(kmeans, open(f"{save_path}/kmeans_{n_clusters}.pkl", "wb"))
    del kmeans
    gc.collect()


if __name__ == "__main__":
    main()
