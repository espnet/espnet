#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#               2022 Johns Hopkins University (Author: Dongji Gao)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np

import pickle
from cuml import PCA



def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given an array of numpy features"
    )
    # fmt: off
    parser.add_argument('data', help='features scp files')
    parser.add_argument('--output', help='where to save the pca matrix', required=True)
    parser.add_argument('--dim', type=int, help='dim for pca reduction', default=512)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print("Reading features")
    x = []
    with open(args.data, "r") as fp:
        for line in fp.readlines():
            _, feats_path = line.split()
            feats = np.load(feats_path)
            x.append(feats)

    x = np.vstack(x)

    print("Computing PCA")
    pca = PCA(n_components=args.dim)
    pca.fit(x)

    os.makedirs(args.output, exist_ok=True)

    pickle.dump(pca, open(f"{args.output}/pca_{args.dim}.pkl", "wb"))


if __name__ == "__main__":
    main()
