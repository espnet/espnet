import argparse
import logging
import os

import numpy as np
from hubert_feature_loader import HubertFeatureReader
from wavlm_feature_loader import WavLMFeatureReader

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import is_scipy_wav_style
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("dump_hubert_or_wavlm_feature")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ssl_type", type=str, default="wavlm", choices=["wavlm", "hubert"]
    )

    parser.add_argument("--ckpt_path", type=str, default="", required=True, help="")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--seed", default=0, type=int)
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
            feat = reader.get_feats(mat, nsample).detach().cpu().numpy()
            writer[utt] = feat
    logger.info("finished successfully")


def main(args):
    np.random.seed(args.seed)
    logging.info("Loading Features")
    if args.ssl_type == "wavlm":
        reader = WavLMFeatureReader(args.ckpt_path, args.layer, args.max_chunk)
    elif args.ssl_type == "hubert":
        reader = HubertFeatureReader(
            hubert_url=args.ckpt_path,
            hubert_dir_path=os.path.dirname(args.ckpt_path),
            layer=args.layer,
            sample_rate=args.sample_rate,
            max_chunk=args.max_chunk,
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
