#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
# Copyright 2023 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate Speaker Embedding Cosine Similarity
between generated and groundtruth audios using X-vector
of the speechbrain pretrained models"""

import argparse
import fnmatch
import logging
import os
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import spatial
from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.pretrained import EncoderClassifier


class XVExtractor:
    """Extract X-vector from speechbrain pretrained models"""

    def __init__(self, args, device):
        self.device = device
        self.audio_norm = AudioNormalizer()
        self.model = EncoderClassifier.from_hparams(
            source=args.pretrained_model, run_opts={"device": device}
        )

    def __call__(self, wav, in_sr):
        wav = self.audio_norm(torch.from_numpy(wav), in_sr).to(self.device)
        embeds = self.model.encode_batch(wav).detach().cpu().numpy()[0]
        return embeds


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> List[str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        List[str]: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Speaker Embedding Cosine Similarity."
    )
    parser.add_argument(
        "gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "gt_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )

    # analysis related
    parser.add_argument(
        "--pretrained_model",
        default="speechbrain/spkrec-ecapa-voxceleb",
        type=str,
        help="Speechbrain pretrained model.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run SECS calculation."""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # find files
    if os.path.isdir(args.gen_wavdir_or_wavscp):
        gen_files = sorted(find_files(args.gen_wavdir_or_wavscp))
    else:
        with open(args.gen_wavdir_or_wavscp) as f:
            gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gen_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")
    if os.path.isdir(args.gt_wavdir_or_wavscp):
        gt_files = sorted(find_files(args.gt_wavdir_or_wavscp))
    else:
        with open(args.gt_wavdir_or_wavscp) as f:
            gt_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gt_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")

    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = args.device
    else:
        device = "cpu"

    xv_extractor = XVExtractor(args, device)

    # calculate SECS
    secs_dict = dict()
    for i, gen_path in enumerate(gen_files):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) in gen_path, gt_files)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="int16")
        gt_x, gt_fs = sf.read(gt_path, dtype="int16")

        fs = gen_fs
        if gen_fs != gt_fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, gen_fs)

        # Amp Normalization -1 ~ 1
        gen_amax = np.amax(np.absolute(gen_x))
        gen_x = gen_x.astype(np.float32) / gen_amax
        gt_amax = np.amax(np.absolute(gt_x))
        gt_x = gt_x.astype(np.float32) / gt_amax
        # X-vector embedding
        gen_embeds = xv_extractor(gen_x, fs)
        gt_embeds = xv_extractor(gt_x, fs)
        # Cosine Similarity
        secs = 1 - spatial.distance.cosine(gen_embeds[0], gt_embeds[0])
        logging.info(f"{gt_basename} {secs:.4f}")
        secs_dict[gt_basename] = secs

    # calculate statistics
    mean_secs = np.mean(np.array([v for v in secs_dict.values()]))
    std_secs = np.std(np.array([v for v in secs_dict.values()]))
    logging.info(f"Average: {mean_secs:.4f} ± {std_secs:.4f}")

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2secs", "w") as f:
        for utt_id in sorted(secs_dict.keys()):
            secs = secs_dict[utt_id]
            f.write(f"{utt_id} {secs:.4f}\n")
    with open(f"{args.outdir}/secs_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_secs:.4f} ± {std_secs:.4f}")

    logging.info("Successfully finished SECS evaluation.")


if __name__ == "__main__":
    main()
