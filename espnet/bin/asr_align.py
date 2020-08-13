#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Johns Hopkins University (Xuankai Chang)
#           2020, Technische Universität München, Authors: Dominik Winkelbauer, Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model CTC alignment script."""

import configargparse
import logging
import os
import sys

from espnet.utils.ctc_segmentation import ctc_align


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Align text to audio using CTC segmentation."
                    "using a pre-trained speech recognition model.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Decoding config file path.")
    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs (max. 1 is supported)")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    # task related
    parser.add_argument(
        "--data-json", type=str, help="Json of recognition data for audio and text"
    )
    parser.add_argument(
        "--utt-text", type=str, help="Text separated into utterances"
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # ctc-segmentation related
    parser.add_argument(
        "--subsampling-factor", type=int, default=None, help="Subsampling factor."
        "If the encoder sub-samples its input, the number of frames at the CTC layer is reduced by this factor."
        "For example, a BLSTMP with subsampling 1_2_2_1_1 has a subsampling factor of 4."
    )
    parser.add_argument(
        "--frame-duration", type=int, default=None, help="Non-overlapping duration of a single frame in milliseconds."
    )
    parser.add_argument(
        "--min-window-size", type=int, default=None, help="Minimum window size considered for utterance."
    )
    parser.add_argument(
        "--max-window-size", type=int, default=None, help="Maximum window size considered for utterance."
    )
    parser.add_argument(
        "--output",
        type=configargparse.FileType('w'),
        required=True,
        help="Output segments file",
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

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # check CUDA_VISIBLE_DEVICES
    device = "cpu"
    if args.ngpu == 1:
        device = "cuda"
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
    elif args.ngpu > 1:
        logging.error("Decoding only supports ngpu=1.")
        sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # recog
    logging.info("backend = " + args.backend)
    if args.backend == "pytorch":
        ctc_align(args, device)
    else:
        raise ValueError("Only pytorch is supported.")
    sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
