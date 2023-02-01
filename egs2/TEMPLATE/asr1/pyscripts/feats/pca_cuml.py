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
import os
import os.path as osp
import pickle

import numpy as np
from cuml import PCA


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given an array of numpy features"
    )
    parser.add_argument("data", help="features scp files")
    parser.add_argument("--output", help="where to save the pca matrix", required=True)
    parser.add_argument("--dim", type=int, help="dim for pca reduction", default=512)

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
