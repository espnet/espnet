# The sklearn_km.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging
import os
import sys
from random import sample
import warnings

import joblib
import numpy as np
import math

import soundfile as sf
import torch
import torchaudio
import tqdm

from sklearn.cluster import MiniBatchKMeans
import fairseq

from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder

from feature_loader import MfccFeatureReader
from feature_loader import HubertFeatureReader

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("sklearn_kmeans")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feats-dir", type=str, help="folder contains wav.scp for training"
    )
    parser.add_argument(
        "--n-clusters", default=100, type=int, help="number of clusters for K-Means"
    )
    parser.add_argument("--nj", default=1, type=int, help="only support mfcc")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--feature-type", type=str, default="mfcc")
    parser.add_argument("--hubert-model-url", type=str, default=None)
    parser.add_argument("--hubert-model-path", type=str, default=None)
    parser.add_argument(
        "--portion", type=float, default=1.0, help="Using a subset of the data."
    )

    group = parser.add_argument_group(description="K-means model.")
    group.add_argument("--km-path", type=str, help="path for k-means model.")
    group.add_argument("--init", default="k-means++")
    group.add_argument("--max-iter", default=100, type=int)
    group.add_argument("--batch-size", default=10000, type=int)
    group.add_argument("--tol", default=0.0, type=float)
    group.add_argument("--max-no-improvement", default=100, type=int)
    group.add_argument("--n-init", default=20, type=int)
    group.add_argument("--reassignment-ratio", default=0.0, type=float)

    return parser


def get_path_iterator(wav, portion=0.1):
    with open(wav, "r") as f:
        lines = [line.rstrip() for line in f]
        lines = sample(lines, int(portion * len(lines)))

        def iterate():
            for line in lines:
                utt_id, path = line.split(" ")
                yield utt_id, f"{path}"

        return iterate, len(lines)


def get_mfcc_feature(feats_dir, fs, nj, portion):
    reader = MfccFeatureReader(fs)
    print(f"{feats_dir}/wav.scp")
    generator, num = get_path_iterator(f"{feats_dir}/wav.scp", portion)
    iterator = generator()

    if nj > 1:
        feats = joblib.Parallel(n_jobs=nj)(
            joblib.delayed(reader.get_feats)(path)
            for utt_id, path in tqdm.tqdm(iterator, total=num)
        )
    else:
        feats = []
        for utt_id, path in tqdm.tqdm(iterator, total=num):
            feat = reader.get_feats(path)
            feats.append(feat.cpu().numpy())
        np.random.shuffle(feat)
    logger.info("Getting MFCC feature successfully")
    return np.vstack(feats)


def get_hubert_feature(feats_dir, fs, portion, url, dir, layer):

    reader = HubertFeatureReader(fs, url, dir, layer)
    generator, num = get_path_iterator(f"{feats_dir}/wav.scp", portion)
    iterator = generator()
    feats = []
    for utt_id, path in tqdm.tqdm(iterator, total=num):
        feat = reader.get_feats(path)
        feats.append(feat.cpu().numpy())
    np.random.shuffle(feat)
    logger.info("Getting HuBERT feature successfully")
    return np.vstack(feats)


def load_feature(
    feats_dir,
    fs,
    nj,
    portion,
    feature_type,
    hubert_model_url,
    hubert_model_path,
):
    # generate mfcc feature
    if feature_type == "mfcc":
        feat = get_mfcc_feature(feats_dir, fs, nj, portion)
    elif "hubert" in feature_type:
        hlayer = int(feature_type.replace("hubert", ""))
        feat = get_hubert_feature(
            feats_dir, fs, portion, hubert_model_url, hubert_model_path, hlayer
        )
    else:
        raise ValueError(f"feature_type: {feature_type}")
    return feat


def train_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def learn_kmeans(
    feats,
    km_path,
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    km_model = train_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feats)
    joblib.dump(km_model, f"{km_path}")

    inertia = -km_model.score(feats) / len(feats)
    logger.info("total intertia: %.5f", inertia)
    logger.info("K-means training successfully")


def main(args):
    np.random.seed(args.seed)
    print("Loading Features")
    feats = load_feature(
        feats_dir=args.feats_dir,
        fs=args.fs,
        nj=args.nj,
        portion=args.portion,
        feature_type=args.feature_type.lower(),
        hubert_model_path=args.hubert_model_path,
        hubert_model_url=args.hubert_model_url,
    )
    print("Learning kmeans")
    learn_kmeans(
        feats,
        km_path=args.km_path,
        n_clusters=args.n_clusters,
        init=args.init,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        tol=args.tol,
        max_no_improvement=args.max_no_improvement,
        n_init=args.n_init,
        reassignment_ratio=args.reassignment_ratio,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.info(str(args))
    main(args)
