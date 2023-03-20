# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging
import os
import random
import sys

import joblib
import numpy as np

from espnet.utils.cli_readers import file_reader_helper

try:
    from cuml.cluster import KMeans
except ModuleNotFoundError:
    raise RuntimeError(
        "cuml not found. please follow guide at tools/installers to install"
    )

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans_cuml")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="scalable-k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--n_init", default=20, type=int)

    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "rspecifier",
        type=str,
        nargs="+",
        help="Read specifier for feats. e.g. ark:some.ark",
    )
    return parser


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    seed,
):
    return KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        tol=tol,
        n_init=n_init,
        random_state=seed,
    )


def load_feature_shard(rspecifier, in_filetype, percent):
    feats = []
    for utt, feat in file_reader_helper(rspecifier, in_filetype):
        feats.append(feat)
    if percent < 0:
        return np.concatenate(feats, axis=0)
    else:
        nsample = int(np.ceil(len(feats) * percent))
        sampled_feat = random.sample(feats, nsample)
        sampled_feat = np.concatenate(
            sampled_feat,
            axis=0,
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from rspecifier {rspecifier}"
            )
        )
        return sampled_feat


def load_feature(rspecifiers, in_filetype, percent):
    assert percent <= 1.0
    if not isinstance(rspecifiers, list):
        rspecifiers = [rspecifiers]
    feat = np.concatenate(
        [
            load_feature_shard(rspecifier, in_filetype, percent)
            for rspecifier in rspecifiers
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    rspecifier,
    in_filetype,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
):
    np.random.seed(seed)
    feat = load_feature(rspecifier, in_filetype, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        n_init,
        seed,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
