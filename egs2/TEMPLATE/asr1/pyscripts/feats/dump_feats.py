# The sklearn_km.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/learn_kmeans.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import logging

import numpy as np
from pyscripts.feats.feats_loader import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
    S3PRLFeatureReader,
)

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import is_scipy_wav_style
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("sklearn_kmeans")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_type", type=str, default="mfcc", choices=["mfcc", "hubert", "s3prl"]
    )
    parser.add_argument("--hubert-model-url", type=str, default=None)
    parser.add_argument("--hubert-model-path", type=str, default=None)
    parser.add_argument("--s3prl-upstream-name", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--hubert_type",
        type=str,
        default="espnet",
        choices=["espnet", "fairseq"],
        help="Whether the HuBERT encoder implementation is based on espnet or fairseq.",
    )
    parser.add_argument(
        "--in_filetype",
        type=str,
        default="sound",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
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
        "--write_num_frames", type=str, help="Specify wspecifer for utt2num_frames"
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier. e.g. ark:some.ark"
    )

    return parser


def dump_feature(
    reader, in_filetype, rspecifier, out_filetype, wspecifier, write_num_frames=None
):
    count = 0
    with file_writer_helper(
        wspecifier,
        filetype=out_filetype,
        write_num_frames=write_num_frames,
    ) as writer:
        for utt, mat in file_reader_helper(rspecifier, in_filetype):
            if is_scipy_wav_style(mat):
                # If data is sound file, then got as Tuple[int, ndarray]
                rate, mat = mat
                mat = mat.astype(np.float64, order="C") / 32768.0
            nsample = len(mat)
            feat = reader.get_feats(mat, nsample).numpy()
            writer[utt] = feat
            count += 1
            if count % 1000 == 0:
                logging.info("process {}".format(count))
    logger.info("finished successfully")


def main(args):
    np.random.seed(args.seed)
    logging.info("Loading Features")
    if args.feature_type == "mfcc":
        reader = MfccFeatureReader(sample_rate=args.sample_rate)
    elif args.feature_type == "hubert":
        assert 0 < args.layer < 24
        if args.hubert_type == "fairseq":
            logging.warning(
                "Fairseq based HuBERT is deprecated. Please use the torchaudio one."
            )
            reader = HubertFeatureReader(
                hubert_url=args.hubert_model_url,
                hubert_dir_path=args.hubert_model_path,
                layer=args.layer,
                sample_rate=args.sample_rate,
                max_chunk=args.max_chunk,
            )
        elif args.hubert_type == "espnet":
            reader = ESPnetHubertFeatureReader(
                hubert_model_path=args.hubert_model_path,
                layer=args.layer,
                sample_rate=args.sample_rate,
                max_chunk=args.max_chunk,
            )
        else:
            raise ValueError(f"Unknown hubert type {args.hubert_type}")
    elif args.feature_type == "s3prl":
        reader = S3PRLFeatureReader(
            s3prl_upstream_name=args.s3prl_upstream_name,
            layer=args.layer,
            sample_rate=16000,
            max_chunk=1600000,
        )
    else:
        raise ValueError(f"Unknown feature type {args.feature_type}.")

    dump_feature(
        reader,
        in_filetype=args.in_filetype,
        rspecifier=args.rspecifier,
        out_filetype=args.out_filetype,
        wspecifier=args.wspecifier,
        write_num_frames=args.write_num_frames,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.info(str(args))

    main(args)
