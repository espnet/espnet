#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import os.path as osp
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="mean pools representations by compressing uniform splits of the data"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--root', help='root of espnet', required=True)
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--subsample_rate', type=float, default=0.5, help='size to subsample data to')
    parser.add_argument("--utt_id", type=str)
    parser.add_argument('--remove_extra', action='store_true', help='if true, removes extra states that cant be pooled, otherwise pads with 0s')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split)

    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = osp.join(args.save_dir, args.split)

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")

    utt_ids = []
    lengths = []
    with open(source_path + ".lengths", "r") as lf:
        for line in lf.readlines():
            utt_id, length = line.split()
            utt_ids.append(utt_id)
            lengths.append(length)

    fsz = features.shape[-1]
    start = 0
    output_dir = f"{save_path}/"
    os.makedirs(output_dir, exist_ok=True)
    scp_file = f"{save_path}/feats.scp"
    prefix = args.root
    with torch.no_grad():
        with open(save_path + ".lengths", "w") as lengths_out, open(
            scp_file, "w"
        ) as sf:

            for length, utt_id in tqdm.tqdm(zip(lengths, utt_ids)):
                utt_id = utt_id.rstrip()
                length = int(length)
                end = start + length
                feats = features[start:end]
                start += length
                x = torch.from_numpy(feats).cuda()
                target_num = math.ceil(length * args.subsample_rate)
                rem = length % target_num

                if rem > 0:
                    if args.remove_extra:
                        to_rem = target_num - rem
                        target_num -= 1
                        x = x[:-to_rem]
                    else:
                        to_add = target_num - rem
                        x = F.pad(x, [0, 0, 0, to_add])
                        x[-to_add:] = x[-to_add - 1]

                x = x.view(target_num, -1, fsz)
                x = x.mean(dim=-2)
                print(target_num, file=lengths_out)
                feat_file = f"{output_dir}/{utt_id}.npy"
                np.save(feat_file, x.cpu().numpy())
                sf.write(f"{utt_id} {feat_file}\n")


if __name__ == "__main__":
    main()
