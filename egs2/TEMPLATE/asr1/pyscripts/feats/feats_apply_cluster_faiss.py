#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#               2022 Johns Hopkins University (Author: Dongji Gao)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as osp
import sys

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from feats_cluster_faiss import parse_faiss_specs


def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    # fmt: off
    parser.add_argument('data', help='location of feature scp files')
    parser.add_argument('--split', help='split to process', required=True)
    parser.add_argument('--labels', help='split to process', default="phn")
    parser.add_argument('--output_path', help='output_path', required=True)
    parser.add_argument('--model_path', help="model_path", required=True)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=14)
    parser.add_argument('--max_tsz', type=int, help='batch kmeans up to this much', default=14)
    parser.add_argument('--delimiter', type=str, help="delimiter for output cluster list", default=" ")
    parser.add_argument('--faiss_specs', '-f', type=str,
                        help='faiss index specs; separated by space '
                             'format is: PCAx_NORM_CLUSx_SPHERICAL -> '
                                'PCAx if exists first apply PCA '
                                'NORM if exists, normalize the vector by L2 norm '
                                'CLUSx must exist, cluster to x clusters '
                                'SPEHRICAL if exists, apply spherical kmeans',
                        default='l2')
    # fmt: on

    return parser


def get_iterator(args):
    label_path = osp.join(args.data, f"{args.split}.{args.labels}")
    if osp.exists(label_path):
        lp = open(label_path, "r")
    else:
        lp = None

    with open(args.data, "r") as fp:
        lines = fp.read().split("\n")
        files = [line.rstrip() for line in lines if len(line) > 0]

        if lp is not None:
            lbls = [line.rstrip() for line in lp]
        else:
            lbls = [None] * len(files)

        num = len(files)

        def iterate():
            for scp_line, lbl in zip(files, lbls):
                fname, feats_path = scp_line.split()
                feats = np.load(feats_path)
                yield feats, fname, lbl

        return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    try:
        faiss_spec = parse_faiss_specs(args.faiss_specs.rstrip("/"))[0]
    except:
        print(spec)
        raise

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca:
        A = torch.from_numpy(np.load(osp.join(args.model_path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.model_path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.model_path, "centroids.npy"))
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources()
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    )
    faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    faiss_index.add(centroids)

    generator, num = get_iterator(args)
    iterator = generator()

    if not osp.exists(args.output_path):
        os.makedirs(args.output_path)

    had_labels = False
    label_path = osp.join(args.output_path, f"{args.split}.{args.labels}")

    output_cluster = f"{args.output_path}/{args.split}.cluster"
    with torch.no_grad():
        with open(output_cluster, "w") as oc, open(label_path, "w") as lp:
            for f, fname, lbl in tqdm.tqdm(iterator, total=num):
                if faiss_spec.pca:
                    f = torch.mm(f, A) + b
                if faiss_spec.norm:
                    f = F.normalize(f, p=2, dim=-1)

                _, z = faiss_index.search(f, 1)

                cluster = f"{args.delimiter}".join(str(x.item()) for x in z)
                oc.write(f"{fname} {cluster}\n")

                if lbl is not None:
                    print(lbl, file=lp)
                    had_labels = True
    if not had_labels:
        os.remove(label_path)


if __name__ == "__main__":
    main()
