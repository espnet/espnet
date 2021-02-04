#!/usr/bin/env python

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
import json
import os

import humanfriendly

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def create_data_dir(args, exist_ok=False):
    """Create the Kaldi-style data directory for SMS-WSJ.

    The following subset directories will be created:
        [args.dist_dir]
          |-- train_si284/
          |-- cv_dev93/
          `-- test_eval92/

    (assume args.num_spk=2)
    In each subset directory, the following files will be created:
        [subset]
          |-- dereverb1.scp
          |-- dereverb2.scp
          |-- noise1.scp
          |-- rir1.scp (not used yet)
          |-- rir2.scp (not used yet)
          |-- spk1.scp
          |-- spk2.scp
          |-- text_spk1
          |-- text_spk2
          |-- utt2dur
          |-- utt2spk
          `-- wav.scp

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
                    "original_source": ["org_src1_path", "org_src2_path"],
                    "rir": ["rir1_path", "rir2_path"],
                    "speech_reverberation_early": ["early1_path", "early2_path"],
                    "speech_reverberation_tail": ["tail1_path", "tail2_path"],
                    "noise_image": "noise_path",
                    "observation": "mixture_path",
                    "speech_source": ["src1_path", "src2_path"],
                    "speech_image": ["src1_image_path", "src2_image_path"]
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

    sample_rate = humanfriendly.parse_size(args.sample_rate)
    for subset in datasets:
        subset_dir = os.path.join(args.dist_dir, subset)
        os.makedirs(subset_dir, exist_ok=exist_ok)
        sorted_keys = sorted(datasets[subset].keys())
        with DatadirWriter(subset_dir) as writer:
            for uid in sorted_keys:
                info = datasets[subset][uid]
                paths = info["audio_path"]
                assert info["num_speakers"] == args.num_spk, (uid, info["num_speakers"])
                # uid: index_src1id_src2id
                spkid = "_".join(info["speaker_id"])
                # uttid: spk1id_spk2id_index_src1id_src2id
                uttid = spkid + "_" + uid
                writer["wav.scp"][uttid] = paths["observation"]
                writer["utt2spk"][uttid] = spkid
                writer["noise1.scp"][uttid] = paths["noise_image"]
                if isinstance(info["num_samples"], dict):
                    num_samples = info["num_samples"]["observation"]
                else:
                    num_samples = info["num_samples"]
                writer["utt2dur"][uttid] = f"{num_samples / sample_rate:.2f}"
                for spk in range(info["num_speakers"]):
                    if args.use_reverb_reference:
                        writer[f"spk{spk+1}.scp"][uttid] = paths["speech_image"][spk]
                    else:
                        writer[f"spk{spk+1}.scp"][uttid] = paths["speech_source"][spk]
                    writer[f"rir{spk+1}.scp"][uttid] = paths["rir"][spk]
                    writer[f"dereverb{spk+1}.scp"][uttid] = paths[
                        "speech_reverberation_early"
                    ][spk]
                    writer[f"text_spk{spk+1}"][uttid] = info["kaldi_transcription"][spk]


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sms_wsj_json", type=str, help="Path to the generated sms_wsj.json"
    )
    parser.add_argument("--num-spk", type=int, default=2, choices=[2, 3, 4])
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
