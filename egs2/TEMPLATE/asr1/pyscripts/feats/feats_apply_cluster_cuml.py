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
import logging
import os
import os.path as osp
import pickle
import sys

import numpy as np
import tqdm
from cuml.cluster import KMeans


def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    parser.add_argument("data", help="location of feature scp files")
    parser.add_argument("--split", help="split to process", required=True)
    parser.add_argument("--output_path", help="output_path", required=True)
    parser.add_argument("--model_path", help="model_path", required=True)
    parser.add_argument("--num_clusters", type=int, default=128)

    return parser


def get_iterator(args):
    with open(args.data, "r") as fp:
        lines = fp.read().split("\n")
        files = [line.rstrip() for line in lines if len(line) > 0]

        num = len(files)

        def iterate():
            for scp_line in files:
                fname, feats_path = scp_line.split()
                feats = np.load(feats_path)
                yield feats, fname

        return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    model_file = f"{args.model_path}/kmeans_{args.num_clusters}.pkl"
    kmeans = pickle.load(open(model_file, "rb"))

    generator, num = get_iterator(args)
    iterator = generator()

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    output_cluster = f"{args.output_path}/{args.split}.cluster"
    with open(output_cluster, "w") as oc:
        for (
            feats,
            fname,
        ) in tqdm.tqdm(iterator, total=num):
            clusters = kmeans.predict(feats)

            clusters_str = " ".join(str(c) for c in clusters)
            oc.write(f"{fname} {clusters_str}\n")


if __name__ == "__main__":
    main()
