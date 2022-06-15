#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import glob
import os

from dateutil import parser


def get_parser():
    parser = argparse.ArgumentParser(description="calculate real time factor (RTF)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="path to logging directory",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="decode",
        choices=["decode", "asr_inference"],
        help="name of logfile, e.g., 'decode' (espnet1) and "
        "'asr_inference' (espnet2)",
    )
    parser.add_argument(
        "--input-shift",
        type=float,
        default=10.0,
        help="shift of inputs in milliseconds",
    )
    parser.add_argument(
        "--start-times-marker",
        type=str,
        default="input length",
        choices=["input length", "speech length"],
        help="String marking start of decoding in logfile, e.g., "
        "'input length' (espnet1) and 'speech length' (espnet2)",
    )
    parser.add_argument(
        "--end-times-marker",
        type=str,
        default="prediction",
        choices=["prediction", "best hypo"],
        help="String marking end of decoding in logfile, e.g., "
        "'prediction' (espnet1) and 'best hypo' (espnet2)",
    )
    return parser


def main():

    args = get_parser().parse_args()

    audio_sec = 0
    decode_sec = 0
    n_utt = 0

    log_files = args.log_name + ".*.log"
    start_times_marker = "INFO: " + args.start_times_marker
    end_times_marker = "INFO: " + args.end_times_marker
    for x in glob.glob(os.path.join(args.log_dir, log_files)):
        audio_durations = []
        start_times = []
        end_times = []
        with codecs.open(x, "r", "utf-8") as f:
            for line in f:
                x = line.strip()
                if start_times_marker in x:
                    audio_durations += [int(x.split(args.start_times_marker + ": ")[1])]
                    start_times += [parser.parse(x.split("(")[0])]
                elif end_times_marker in x:
                    end_times += [parser.parse(x.split("(")[0])]
        assert len(audio_durations) == len(end_times), (
            len(audio_durations),
            len(end_times),
        )
        assert len(start_times) == len(end_times), (len(start_times), len(end_times))
        audio_sec += sum(audio_durations) * args.input_shift / 1000  # [sec]
        decode_sec += sum(
            [
                (end - start).total_seconds()
                for start, end in zip(start_times, end_times)
            ]
        )
        n_utt += len(audio_durations)

    print("Total audio duration: %.3f [sec]" % audio_sec)
    print("Total decoding time: %.3f [sec]" % decode_sec)
    rtf = decode_sec / audio_sec if audio_sec > 0 else 0
    print("RTF: %.3f" % rtf)
    latency = decode_sec * 1000 / n_utt if n_utt > 0 else 0
    print("Latency: %.3f [ms/sentence]" % latency)


if __name__ == "__main__":
    main()
