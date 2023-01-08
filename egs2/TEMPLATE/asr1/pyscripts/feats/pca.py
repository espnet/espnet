#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np

import faiss



def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given an array of numpy features"
    )
    # fmt: off
    parser.add_argument('data', help='features scp files')
    parser.add_argument('--output', help='where to save the pca matrix', required=True)
    parser.add_argument('--dim', type=int, help='dim for pca reduction', required=True)
    parser.add_argument('--eigen_power', type=float, default=0, help='eigen power, -0.5 for whitening')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print("Reading features")
    x = []
    with open(args.data, "r") as fp:
        for line in fp.readlines():
            utt_id, feats_path = line.split()
            feats = np.load(feats_path)
            x.append(feats)

    x = np.vstack(x)

    print("Computing PCA")
    pca = faiss.PCAMatrix(x.shape[-1], args.dim, args.eigen_power)
    pca.train(x)
    b = faiss.vector_to_array(pca.b)
    A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

    os.makedirs(args.output, exist_ok=True)

    prefix = str(args.dim)
    if args.eigen_power != 0:
        prefix += f"_{args.eigen_power}"

    np.save(osp.join(args.output, f"{prefix}_pca_A"), A.T)
    np.save(osp.join(args.output, f"{prefix}_pca_b"), b)


if __name__ == "__main__":
    main()
