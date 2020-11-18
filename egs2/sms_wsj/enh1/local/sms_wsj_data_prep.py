#!/usr/bin/env python

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
import json
import os

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def create_data_dir(args, exist_ok=False):
    """Create the Kaldi-style data directory for SMS-WSJ.

    The following subset directories will be created:
        [args.dist_dir]
          |-- train_si284/
          |-- cv_dev93/
          |-- test_eval92/

    In each subset directory, the following files will be created:
        [subset]
          |-- wav.scp
          |-- utt2spk
          |-- spk1.scp
          |-- spk2.scp
          |-- noise1.scp
          |-- dereverb1.scp
          |-- dereverb2.scp
          |-- rir1.scp (not used yet)
          |-- rir2.scp (not used yet)

    Args:
        args.min_or_max: min or max version of speech mixture
        args.sample_rate: sample rate (either 8k or 16k)
        args.use_reverb_reference: False to use source signal as the reference
        args.dist_dir: path to the target data directory
        args.sms_wsj_json: a json file with the following structure:
          {
            "dataset": {
              "train_si284": {
                "uttid1": {
                  "room_dimensions": [[x], [y], [z]],
                  "sound_decay_time": rt60,
                  "source_position": [[x1, x2], [y1, y2], [z1, z2]],
                  "sensor_position": [[x1,...,x6], [y1,...,y6], [z1,...,z6]],
                  "example_id": "idx",
                  "num_speakers": 2,
                  "speaker_id": ["spk1id", "spk2id"],
                  "source_id": ["src1id", "src2id"],
                  "gender": ["male", "female"],
                  "kaldi_transcription": [
                    "THERE AREN'T MANY SELLERS AROUND ONE BROKER SAID",
                    "AD AGENCIES ARE MAD"
                  ],
                  "log_weights": [sir, -sir],
                  "num_samples": {
                    "speech_source": [len_src1, len_src2],
                    "observation": max(len_src1, len_src2)
                  },
                  "offset": [start_idx, 0],
                  "audio_path": {
                    "speech_source": ["src1_path", "src2_path"],
                    "rir": ["rir1_path", "rir2_path"],
                    "speech_reverberation_early": ["early1_path", "early2_path"],
                    "speech_reverberation_tail": ["tail1_path", "tail2_path"],
                    "noise_image": "noise_path",
                    "observation": "mixture_path"
                  },
                  "snr": snr
                },
                ...
              },
              "cv_dev93": {
                ...
              },
              "test_eval92": {
                ...
              }
            }
          }
        exist_ok (bool): used for os.makedirs (Default=False)
    """
    with open(args.sms_wsj_json, "r") as f:
        datasets = json.load(f)["datasets"]

    for subset in datasets:
        subset_dir = os.path.join(args.dist_dir, subset)
        os.makedirs(subset_dir, exist_ok=exist_ok)
        with DatadirWriter(subset_dir) as writer:
            for uid, info in datasets[subset].items():
                assert info["num_speakers"] == 2, (uid, info["num_speakers"])
                # uid: index_src1id_src2id
                spkid = "_".join(info["speaker_id"])
                uttid = spkid + "_" + uid
                writer["wav.scp"][uttid] = info["audio_path"]["observation"]
                writer["utt2spk"][uttid] = spkid
                writer["noise1.scp"][uttid] = info["audio_path"]["noise_image"]
                for spk in range(info["num_speakers"]):
                    if args.use_reverb_reference:
                        writer[f"spk{spk+1}.scp"][uttid] = info["audio_path"]["reverb_source"][spk]
                    else:
                        writer[f"spk{spk+1}.scp"][uttid] = info["audio_path"]["source_signal"][spk]
                    writer[f"rir{spk+1}.scp"][uttid] = info["audio_path"]["rir"]
                    writer[f"dereverb{spk+1}.scp"][uttid] = info["audio_path"][
                        "speech_reverberation_early"
                    ][spk]
                writer["wav.scp"][uttid] = info["audio_path"]["observation"]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sms_wsj_json", type=str, help="Path to the generated sms_wsj.json"
    )
    parser.add_argument("--min-or-max", type=str, default="max", choices=["max"])
    parser.add_argument("--sample-rate", type=str, default="8k", choices=["8k", "16k"])
    parser.add_argument("--use-reverb-reference", type=str2bool, default=True)
    parser.add_argument(
        "--dist-dir",
        type=str,
        default="data",
        help="Directory to store the Kaldi-style data",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.exists(args.sms_wsj_json)
    create_data_dir(args, exist_ok=False)
