#!/usr/bin/env python3

import argparse
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import kaldiio
import librosa
import numpy as np
import soundfile as sf

from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.types import str_or_none
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description="Generate noisy speech by clean speech and noise",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input / Output
    parser.add_argument(
        "--input_scp",
        type=Path,
        help="Input scp file for clean speech",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output directory",
    )

    # Noise setup
    parser.add_argument(
        "--noise_scp",
        type=str_or_none,
        help="Input scp file for noise",
    )
    parser.add_argument(
        "--noise_apply_prob", type=float, default=1.0, help="probability to apply noise"
    )
    parser.add_argument("--noise_db_range", type=str, help="range of noise in dB")

    # RIR setup
    parser.add_argument(
        "--rir_scp",
        type=str_or_none,
        help="Input scp file for RIR",
    )
    parser.add_argument(
        "--rir_apply_prob", type=float, default=0.5, help="probability to apply RIR"
    )

    # other setups
    parser.add_argument("--fs", type=int, default=16000, help="sampling rate")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="common",
        choices=["common"],
        help="choice of ESPnet preprocessor",
    )

    return parser


def synthesis(
    args,
    preprocessor,
    line,
):
    uttid, path = line.strip().split()
    logging.info(f"start processing {uttid}: {path}")

    # kaldi-style ark file
    if ":" in path and path.split(":")[1].isdigit():
        logging.info(f"proceed with kaldiio")
        fs, speech = kaldiio.load_mat(path)
        assert fs == args.fs
    else:
        speech, _ = librosa.load(path, sr=args.fs)

    processed_speech = preprocessor(
        uid=uttid,
        data={"speech": speech},
    )["speech"]
    processed_speech = np.expand_dims(processed_speech, 1)

    audio_path = args.output_dir / f"{uttid}.wav"
    sf.write(str(audio_path), processed_speech, args.fs)

    logging.info(f"end processing {uttid}")
    return (uttid, audio_path)


def main():
    parser = get_parser()
    args = parser.parse_args()

    # load preprocessor
    if args.preprocessor == "common":
        preprocessor_args = {
            "fs": args.fs,
            "noise_scp": args.noise_scp,
            "noise_apply_prob": args.noise_apply_prob,
            "noise_db_range": args.noise_db_range,
            "rir_scp": args.rir_scp,
            "rir_apply_prob": args.rir_apply_prob,
            "force_single_channel": True,
            "speech_volume_normalize": 1.0,
            "train": True,
        }
        preprocessor = CommonPreprocessor(**preprocessor_args)
    else:
        raise NotImplementedError(f"unsupported preprocessor {args.preprocessor}")
    logging.info(f"preprocessor is built")

    # before the loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    func = partial(synthesis, args, preprocessor)
    writer = open(args.output_dir / "wav.scp", "w")

    results = map(func, open(args.input_scp))

    logging.info("done all jobs. Dump wav.scp")

    for uttid, path in results:
        writer.write(f"{uttid} {path}\n")


if __name__ == "__main__":
    main()
