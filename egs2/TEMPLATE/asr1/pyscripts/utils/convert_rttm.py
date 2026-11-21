#!/usr/bin/env python3

import argparse
import collections.abc
import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

import humanfriendly
import numpy as np
import soundfile
from typeguard import typechecked

from espnet2.utils.types import str_or_int


@typechecked
def convert_rttm_text(
    path: Union[Path, str],
    wavscp_path: Union[Path, str],
    sampling_rate: int,
    output_path: Union[Path, str],
    utt2num_samples_path: Optional[Union[Path, str]] = None,
) -> None:
    """Convert a RTTM file

    Note: only support speaker information now

    The recording lengths come from ``utt2num_samples`` when it is given (it is
    written by format_wav_scp.sh at the dump sampling rate); otherwise every
    wav.scp entry is opened with soundfile, which only works for plain audio
    files, not for kaldi_ark or omniio entries.
    """
    num_samples = None
    if utt2num_samples_path is not None:
        with Path(utt2num_samples_path).open("r", encoding="utf-8") as f:
            num_samples = dict(line.split(maxsplit=1) for line in f if line.strip())

    output_handler = Path(os.path.join(output_path, "espnet_rttm")).open(
        "w", encoding="utf-8"
    )

    utt_ids = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = re.split(" +", line.rstrip())

            # RTTM format must have exactly 9 fields
            assert len(sps) == 9, "{} does not have exactly 9 fields".format(path)
            label_type, utt_id, channel, start, duration, _, _, spk_id, _ = sps

            # Only support speaker label now
            assert label_type == "SPEAKER"

            utt_ids.add(utt_id)
            start = int(np.rint(float(start) * sampling_rate))
            end = start + int(np.rint(float(duration) * sampling_rate))

            output_handler.write(
                "{} {} {} {} {} <NA> <NA> {} <NA>\n".format(
                    label_type, utt_id, channel, start, end, spk_id
                )
            )

    with Path(wavscp_path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            utt_id, wav_path = re.split("[ \t]+", line.rstrip(), maxsplit=1)
            assert utt_id in utt_ids, "{} is not in corresponding rttm {}".format(
                utt_id, path
            )

            if num_samples is not None:
                frames = int(num_samples[utt_id])
            else:
                sf = soundfile.SoundFile(wav_path)
                assert sf.samplerate == sampling_rate
                frames = sf.frames
            output_handler.write(
                (
                    "{} {} <NA> <NA> {} <NA> <NA> <NA> <NA>\n".format(
                        "END", utt_id, frames
                    )
                )
            )

    output_handler.close()


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert standard rttm file to ESPnet format"
    )
    parser.add_argument("--rttm", required=True, type=str, help="Path of rttm file")
    parser.add_argument(
        "--wavscp",
        required=True,
        type=str,
        help="Path of corresponding scp file",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Output directory to storry espnet_rttm",
    )
    parser.add_argument(
        "--utt2num_samples",
        default=None,
        type=str,
        help="Take the recording lengths from this file instead of opening "
        "the audio (needed for kaldi_ark and omniio dumps)",
    )
    parser.add_argument(
        "--sampling_rate",
        type=str_or_int,
        default=16000,
        help="Sampling rate of the audio",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Convert standard rttm to sample-based result"""
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

    sampling_rate = humanfriendly.parse_size(args.sampling_rate)
    convert_rttm_text(
        args.rttm,
        args.wavscp,
        sampling_rate,
        args.output_path,
        utt2num_samples_path=args.utt2num_samples,
    )

    logging.info("Successfully finished RTTM converting.")


if __name__ == "__main__":
    main()
