# The learn_kmeans.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/simple_kmeans/dump_km_label.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


import argparse
import logging
import os
import sys

import joblib
import numpy as np
import torch
from ssl_feature_utils import (
    ESPnetHubertFeatureReader,
    HubertFeatureReader,
    MfccFeatureReader,
    S3PRLFeatureReader,
    build_data_iterator,
    format_feature_conf_str,
)

from espnet2.utils.types import str2bool
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_writers import file_writer_helper

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


feature_reader_choice = dict(
    mfcc=MfccFeatureReader,
    fairseq_hubert=HubertFeatureReader,
    espnet_hubert=ESPnetHubertFeatureReader,
    s3prl=S3PRLFeatureReader,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    parser.add_argument("--online_feature_extract", type=str2bool, default=False)
    parser.add_argument("--feature_conf", type=str, default=None)
    parser.add_argument("--batch_bins", type=int, default=1)
    parser.add_argument(
        "--utt2num_samples",
        type=str,
        default=None,
        help="Specify the utt2num_samples file.",
    )

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
        default="mat",
        choices=["mat", "hdf5", "sound.hdf5", "sound"],
        help="Specify the file format for the rspecifier. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=16000,
        help="input audio sampling rate (could be different from fs used in SSL)",
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier for labels. e.g. ark,t:some.txt"
    )

    return parser


class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.C.device)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def dump_label(
    rspecifier,
    in_filetype,
    audio_sample_rate,
    wspecifier,
    out_filetype,
    km_path,
    use_gpu,
    online_feature_extract,
    **kwargs
):
    if online_feature_extract:
        assert "feature_conf" in kwargs
        # need to wrap arguments with double-quotes for json string
        feature_conf = format_feature_conf_str(kwargs["feature_conf"])
    else:
        feature_conf = None

    apply_kmeans = ApplyKmeans(km_path, use_gpu=use_gpu)

    if not online_feature_extract:
        # dumped ssl feature in kaldi ark format
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt, feat in file_reader_helper(rspecifier, in_filetype):
                lab = apply_kmeans(feat)
                writer[utt] = lab
    else:
        assert feature_conf["type"] in feature_reader_choice
        reader_class = feature_reader_choice[feature_conf["type"]]
        reader_conf = feature_conf.get("conf", dict())

        if reader_conf.get("multilayer_feature", None):
            reader_conf["multilayer_feature"] = str2bool(
                reader_conf["multilayer_feature"]
            )
        if reader_conf.get("layer", None):
            reader_conf["layer"] = int(reader_conf["layer"])
        reader_conf["audio_sample_rate"] = audio_sample_rate

        reader = reader_class(**reader_conf)
        iterator = build_data_iterator(
            rspecifier,
            in_filetype,
            utt2num_samples=args.utt2num_samples,
            batch_bins=kwargs.get("batch_bins", 1),
        )
        with file_writer_helper(
            wspecifier,
            filetype=out_filetype,
        ) as writer:
            for utt_ids, data in iterator:
                feats, feats_lens = reader.get_feats(
                    data["speech"], data["speech_lengths"]
                )

                for idx, utt in enumerate(utt_ids):
                    lab = apply_kmeans(feats[idx][: feats_lens[idx]].numpy())
                    writer[utt] = lab

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    dump_label(**vars(args))
