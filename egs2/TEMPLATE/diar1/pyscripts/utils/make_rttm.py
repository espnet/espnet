#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (Yusuke Fujita)
#           2021 Johns Hopkins University (Jiatong Shi)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging

import humanfriendly
import numpy as np
from scipy.signal import medfilt

from espnet2.fileio.npy_scp import NpyScpReader


def get_parser() -> argparse.Namespace:
    """Get argument parser."""

    parser = argparse.ArgumentParser(description="make rttm from decoded result")
    parser.add_argument("diarize_scp")
    parser.add_argument("out_rttm_file")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--frame_shift", default=128, type=int)
    parser.add_argument("--subsampling", default=1, type=int)
    parser.add_argument("--median", default=1, type=int)
    parser.add_argument("--sampling_rate", default="8000", type=str)
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Make rttm based on diarization inference results"""
    args = get_parser().parse_args()
    sampling_rate = humanfriendly.parse_size(args.sampling_rate)
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

    scp_reader = NpyScpReader(args.diarize_scp)

    with open(args.out_rttm_file, "w") as wf:
        for key in scp_reader.keys():
            data = scp_reader[key]
            a = np.where(data[:] > args.threshold, 1, 0)
            if args.median > 1:
                a = medfilt(a, (args.median, 1))
            factor = args.frame_shift * args.subsampling / sampling_rate
            for spkid, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), "constant")
                (changes,) = np.where(np.diff(frames, axis=0) != 0)
                fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    print(
                        fmt.format(
                            key,
                            s * factor,
                            (e - s) * factor,
                            key + "_" + str(spkid),
                        ),
                        file=wf,
                    )

    logging.info("Constructed RTTM for {}".format(args.diarize_scp))


if __name__ == "__main__":
    main()
