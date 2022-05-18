import argparse
import logging
import os
import pdb
import sys

import joblib
import numpy as np
import torch
import tqdm
from sklearn_km import (HubertFeatureReader, MfccFeatureReader,
                        get_path_iterator)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("dump_km_label")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km-path", type=str)
    parser.add_argument("--label-path", type=str)
    parser.add_argument(
        "--recog-set", default=None, nargs="+", help="folders contain wav.scp for recog"
    )
    parser.add_argument("--feature", default="mfcc", type=str)
    parser.add_argument("--nj", default=1, type=int)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hurl", type=str, default="./")
    parser.add_argument("--hdir", type=str, default="./")

    return parser


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.nc = self.km_model.cluster_centers_.transpose()
        self.nc_norm = (self.nc**2).sum(0, keepdims=True)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        probs = (
            (x**2).sum(1, keepdims=True) - 2 * np.matmul(x, self.nc) + self.nc_norm
        )
        return np.argmin(probs, axis=1)


def dump_pseudo_label_mfcc(km_path, task, sample_rate, nj):
    apply_kmeans = ApplyKmeans(km_path)
    reader = MfccFeatureReader(sample_rate)
    generator, num = get_path_iterator(f"{task}/wav.scp", 1.0)
    iterator = generator()

    if nj > 1:
        feats = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(reader.get_feats)(path)
            for utt_id, path in tqdm.tqdm(iterator, total=num)
        )

        p_labs = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(apply_kmeans)(feat) for feat in tqdm.tqdm(feats, total=num)
        )
        iterator = generator()
        utt_ids = [utt_id for utt_id, _ in iterator]
    else:
        utt_ids, p_labs = [], []
        for utt_id, path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            p_lab = apply_kmeans(feat).tolist()
            p_labs.append(p_lab)
            utt_ids.append(utt_id)
    return utt_ids, p_labs


def dump_pseudo_label_hubert(km_path, task, sample_rate, url, dir, layer):
    apply_kmeans = ApplyKmeans(km_path)
    reader = HubertFeatureReader(sample_rate, url, dir, layer)
    generator, num = get_path_iterator(f"{task}/wav.scp", 1.0)
    iterator = generator()

    utt_ids, p_labs = [], []
    for utt_id, path in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        p_lab = apply_kmeans(feat).tolist()
        p_labs.append(p_lab)
        utt_ids.append(utt_id)
    return utt_ids, p_labs


def dump_label(km_path, label_path, recog_set, feature, nj, sample_rate, hurl, hdir):
    feature = feature.lower()
    if recog_set:
        for task in recog_set:
            logger.info("Dumping pseudo labeling for: %s", task)
            if feature == "mfcc":
                utt_ids, p_labs = dump_pseudo_label_mfcc(
                    f"{km_path}",
                    task,
                    sample_rate,
                    nj,
                )
            elif "hubert" in feature:
                hlayer = int(feature.replace("hubert", ""))
                utt_ids, p_labs = dump_pseudo_label_hubert(
                    f"{km_path}",
                    task,
                    sample_rate,
                    hurl,
                    hdir,
                    hlayer,
                )

            with open(label_path, "w") as f:
                for utt_id, p_lab in zip(utt_ids, p_labs):
                    f.write(utt_id + " " + " ".join(map(str, p_lab)) + "\n")

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))
