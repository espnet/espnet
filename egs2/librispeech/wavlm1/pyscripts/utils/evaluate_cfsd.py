#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
# Copyright 2023 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate Conditional Frechet Speech Distance
between generated and groundtruth audios
using the s3prl pretrained models."""

import argparse
import fnmatch
import logging
import os
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy import linalg

from espnet2.asr.frontend.s3prl import S3prlFrontend


# from https://github.com/bioinf-jku/TTUR
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product;"
            "adding %s to diagonal of cov estimates" % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


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
        description="Evaluate Conditional Frechet Speech Distance."
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
        default="wav2vec2",
        type=str,
        help="S3prl pretrained upstream model.",
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
    """Run CFSD calculation."""
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

    s3prl_frontend = S3prlFrontend(
        download_dir="./hub",
        frontend_conf={"upstream": args.pretrained_model},
    )
    s3prl_frontend.to(device)

    # calculate CFSD
    cfsd_dict = dict()
    for i, gen_path in enumerate(gen_files):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) in gen_path, gt_files)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as float64
        gen_x, gen_fs = sf.read(gen_path, dtype="float64")
        gt_x, gt_fs = sf.read(gt_path, dtype="float64")

        # NOTE: resample because s3prl models support only 16kHz audio currently.
        gen_x = librosa.resample(gen_x, orig_sr=gen_fs, target_sr=16000)
        gt_x = librosa.resample(gt_x, orig_sr=gt_fs, target_sr=16000)

        # prepare input
        gen_x = torch.FloatTensor(gen_x).unsqueeze(0).to(device)
        gen_x_length = torch.LongTensor([gen_x.shape[1]]).to(device)
        gt_x = torch.FloatTensor(gt_x).unsqueeze(0).to(device)
        gt_x_length = torch.LongTensor([gt_x.shape[1]]).to(device)

        # speech embedding
        gen_embeds, gen_embeds_len = s3prl_frontend(gen_x, gen_x_length)  # (B,H)
        gt_embeds, gt_embeds_len = s3prl_frontend(gt_x, gt_x_length)  # (B,H)
        gen_embeds = gen_embeds.detach().cpu().numpy()[0]
        gt_embeds = gt_embeds.detach().cpu().numpy()[0]

        # speech distance
        gen_mu = np.mean(gen_embeds, axis=0)
        gt_mu = np.mean(gt_embeds, axis=0)
        gen_sigma = np.cov(gen_embeds, rowvar=False)
        gt_sigma = np.cov(gt_embeds, rowvar=False)
        cfsd = calculate_frechet_distance(gen_mu, gen_sigma, gt_mu, gt_sigma)
        logging.info(f"{gt_basename} {cfsd:.4f}")
        cfsd_dict[gt_basename] = cfsd

    # calculate statistics
    mean_cfsd = np.mean(np.array([v for v in cfsd_dict.values()]))
    std_cfsd = np.std(np.array([v for v in cfsd_dict.values()]))
    logging.info(f"Average: {mean_cfsd:.4f} ± {std_cfsd:.4f}")

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2cfsd", "w") as f:
        for utt_id in sorted(cfsd_dict.keys()):
            cfsd = cfsd_dict[utt_id]
            f.write(f"{utt_id} {cfsd:.4f}\n")
    with open(f"{args.outdir}/cfsd_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_cfsd:.4f} ± {std_cfsd:.4f}")

    logging.info("Successfully finished CFSD evaluation.")


if __name__ == "__main__":
    main()
