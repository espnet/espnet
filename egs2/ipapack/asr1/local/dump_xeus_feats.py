import argparse
import logging
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from espnet2.tasks.ssl import SSLTask
from pyscripts.feats.ssl_feature_utils import BaseFeatureReader, dump_feature


# adapted from https://github.com/simpleoier/ESPnet_SSL_ASR_tutorial_misc/blob/main/dump_feats.py


class XEUSFeatureReader(BaseFeatureReader):
    def __init__(self, checkpoint_path, layer=-1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        assert -1 <= layer <= 18
        self.layer = layer
    
        xeus_model, xeus_train_args = SSLTask.build_model_from_file(
            config_file=None,
            model_file=checkpoint_path,
            device=self.device
        )
        self.model = xeus_model

    def get_feats(
        self, data: torch.Tensor, data_lens: torch.Tensor, ref_len: Optional[int] = None
    ):
        # we recommend use_mask=True during fine-tuning
        # take the output of the last layer -> batch_size x seq_len x hdim
        with torch.no_grad():
            x, x_lens = self.preprocess_data(data, data_lens)
            wavs = x.to(self.device)
            # TODO: allow linear combo of layers??
            # source: https://www.wavlab.org/activities/2024/xeus/
            feats = self.model.encode(wavs, data_lens, use_mask=False, use_final_output=False)[0][self.layer]
            # ex: [1, 1097, 1024] for 1 file that's 20 s

            # based on https://github.com/pytorch/audio/blob/ba696ea3dfec4cbe693bf06a84c75dc196077f5b/src/torchaudio/models/wav2vec2/model.py#L85
                # just return the length of the original data
                # the # frames of each item pre-padding
        return feats.cpu(), x_lens


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_type", type=str, default="mfcc", choices=["mfcc", "hubert", "s3prl"]
    )
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--seed", default=15213, type=int)
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
        "--utt2num_samples", type=str, help="Path to utt2num_samples file"
    )
    parser.add_argument(
        "rspecifier", type=str, help="Read specifier for feats. e.g. ark:some.ark"
    )
    parser.add_argument(
        "wspecifier", type=str, help="Write specifier. e.g. ark:some.ark"
    )

    return parser


if __name__ == "__main__":
    checkpoint_path='/ocean/projects/cis210027p/kchang1/XEUS/model/xeus_checkpoint.pth'
    reader = XEUSFeatureReader(
                checkpoint_path=checkpoint_path,
                layer=-1
            )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("sklearn_kmeans")

    parser = get_parser()
    args = parser.parse_args()
    logging.info(str(args))

    np.random.seed(args.seed)
    logging.info("Loading Features")

    dump_feature(
        reader,
        in_filetype=args.in_filetype,
        rspecifier=args.rspecifier,
        out_filetype=args.out_filetype,
        wspecifier=args.wspecifier,
        utt2num_samples=args.utt2num_samples,
        write_num_frames=args.write_num_frames,
        batch_bins=22500000
    )
