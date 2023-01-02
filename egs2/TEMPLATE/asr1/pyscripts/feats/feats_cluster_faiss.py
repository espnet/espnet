#!/usr/bin/env python3 -u
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
import numpy as np
import tqdm
import torch

from collections import namedtuple

import faiss

import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('feats_scp', help='location of feature scp files')
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--sample-pct', '-r', type=float, help='percentage of timesteps to sample', default=0)
    parser.add_argument('--faiss-specs', '-f', type=str,
                        help='faiss index specs; separated by space '
                             'format is: PCAx_NORM_CLUSx_SPHERICAL -> '
                                'PCAx if exists first apply PCA '
                                'NORM if exists, normalize the vector by L2 norm '
                                'CLUSx must exist, cluster to x clusters '
                                'SPEHRICAL if exists, apply spherical kmeans',
                        default='l2')
    # fmt: on

    return parser


faiss_spec = namedtuple("faiss_spec", ["pca", "norm", "n_clus", "sphere", "spec_str"])


def parse_faiss_specs(specs_str):
    specs = []
    for ss in specs_str.split():
        comps = ss.split("_")
        pca = 0
        norm = False
        n_clus = 0
        sphere = False
        for c in comps:
            if c.startswith("PCA"):
                pca = int(c[3:])
            elif c == "NORM":
                norm = True
            elif c.startswith("CLUS"):
                n_clus = int(c[4:])
            elif c == "SPHERICAL":
                sphere = True
        assert n_clus > 0
        specs.append(
            faiss_spec(pca=pca, norm=norm, n_clus=n_clus, sphere=sphere, spec_str=ss)
        )
    return specs


def main():
    parser = get_parser()
    args = parser.parse_args()

    faiss_specs = parse_faiss_specs(args.faiss_specs)
    print("Faiss Specs:", faiss_specs)

    feats_list = []
    with open(args.feats_scp, "r") as f_scp:
        for line in f_scp.readlines():
            _, feat_file = line.split()
            feat = np.load(feat_file)
            feats_list.append(feat)

        feats = np.concatenate(feats_list)

        os.makedirs(args.save_dir, exist_ok=True)
        # np.save(feat_path, feats)

        gc.collect()

    reload = False
    for spec in faiss_specs:
        print("Processing spec", spec)

        if reload:
            print("Reloading...")
            del feats
            gc.collect()
            feats = np.load(feat_path + ".npy")

        save_path = osp.join(args.save_dir, spec.spec_str)
        os.makedirs(save_path, exist_ok=True)
        d = feats.shape[-1]
        x = feats
        if spec.pca > 0:
            print("Computing PCA")
            pca = faiss.PCAMatrix(d, spec.pca)
            pca.train(x)
            d = spec.pca
            b = faiss.vector_to_array(pca.b)
            A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
            np.save(osp.join(save_path, "pca_A"), A.T)
            np.save(osp.join(save_path, "pca_b"), b)
            print("Applying PCA")
            x = pca.apply_py(x)

        if spec.norm:
            reload = spec.pca <= 0
            print("Normalizing")
            faiss.normalize_L2(x)

        print("Computing kmeans")
        kmeans = faiss.Kmeans(
            d,
            spec.n_clus,
            niter=50,
            verbose=True,
            spherical=spec.sphere,
            max_points_per_centroid=feats.shape[0],
            gpu=True,
            nredo=3,
        )
        kmeans.train(x)
        np.save(osp.join(save_path, "centroids"), kmeans.centroids)
        del kmeans
        del x
        gc.collect()


if __name__ == "__main__":
    main()
