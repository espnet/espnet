#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model CTC alignment script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.ctc_align_func import ctc_align

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`",
    )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")

    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=0,
        help="Batch size for beam search (0: means no batch processing)",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )

    # task related
    parser.add_argument(
        "--data-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Filename of result label data (json)",
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--utt-text", type=str, help="Text separated into utts"
    )
    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args, extra = parser.parse_known_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    ctc_align(args)

if __name__ == "__main__":
    main(sys.argv[1:])
