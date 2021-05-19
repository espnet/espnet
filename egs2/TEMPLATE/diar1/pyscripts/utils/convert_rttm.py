#!/usr/bin/env python3

import collections.abc
import humanfriendly
from pathlib import Path
from typing import Union

import argparse
import logging
import numpy as np
import re
import os
import soundfile
from espnet2.utils.types import str_or_int
from typeguard import check_argument_types


def convert_rttm_text(
    path: Union[Path, str],
    wavscp_path: Union[Path, str],
    sampling_rate: int,
    output_path: Union[Path, str],
) -> None:
    """Convert a RTTM file

    Note: only support speaker information now
    """

    output_handler = Path(os.path.join(output_path, "espnet_rttm")).open(
        "w", encoding="utf-8"
    )

    assert check_argument_types()
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
            sps = re.split("[ \t]+", line.rstrip())
            utt_id, wav_path = sps
            assert utt_id in utt_ids, "{} is not in corresponding rttm {}".foramt(
                utt_id, path
            )

            array, rate = soundfile.read(wav_path, always_2d=True)
            assert rate == sampling_rate
            shape = array.shape[0]
            output_handler.write(
                (
                    "{} {} <NA> <NA> {} <NA> <NA> <NA> <NA>\n".format(
                        "END", utt_id, shape
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
    convert_rttm_text(args.rttm, args.wavscp, sampling_rate, args.output_path)

    logging.info("Successfully finished RTTM converting.")


if __name__ == "__main__":
    main()
