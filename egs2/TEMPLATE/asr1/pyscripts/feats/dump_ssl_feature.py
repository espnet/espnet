import argparse
import logging
import os

import numpy as np
from ssl_feature_utils import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
    S3PRLFeatureReader,
    dump_feature,
    format_feature_conf_str,
)

from espnet2.utils.types import str2bool

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("dump_hubert_or_wavlm_feature")


feature_reader_choice = dict(
    mfcc=MfccFeatureReader,
    fairseq_hubert=HubertFeatureReader,
    espnet_hubert=ESPnetHubertFeatureReader,
    s3prl=S3PRLFeatureReader,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_conf", type=str, default=None)
    parser.add_argument("--use_gpu", default=True, type=str2bool)
    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound", "kaldi_ark"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--out_filetype",
        type=str,
        default="npy",
        choices=["npy", "mat", "hdf5"],
        help="Specify the file format for the wspecifier. "
        '"npy" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--utt2num_samples",
        type=str,
        default=None,
        help="Specify the utt2num_samples file.",
    )
    parser.add_argument(
        "--write_num_frames", type=str, help="Specify wspecifer for utt2num_frames"
    )
    parser.add_argument(
        "--batch_bins", type=int, default=1, help="Number of sample points in a batch."
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier. e.g. ark:some.ark"
    )

    return parser


def main(args):
    logging.info("Loading Features")

    feature_conf = format_feature_conf_str(args.feature_conf)
    logging.info(f"Feature configuration: {feature_conf}")
    reader_class = feature_reader_choice[feature_conf["type"]]
    reader_conf = feature_conf.get("conf", dict())

    if reader_conf.get("multilayer_feature", None):
        reader_conf["multilayer_feature"] = str2bool(reader_conf["multilayer_feature"])
    if reader_conf.get("layer", None):
        reader_conf["layer"] = int(reader_conf["layer"])
    reader = reader_class(use_gpu=args.use_gpu, **reader_conf)

    dump_feature(
        reader,
        in_filetype=args.in_filetype,
        rspecifier=args.rspecifier,
        out_filetype=args.out_filetype,
        wspecifier=args.wspecifier,
        utt2num_samples=args.utt2num_samples,
        write_num_frames=args.write_num_frames,
        batch_bins=args.batch_bins,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.info(str(args))

    main(args)
