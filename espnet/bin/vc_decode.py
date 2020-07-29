#!/usr/bin/env python3

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VC decoding script."""

import configargparse
import logging
import os
import platform
import subprocess
import sys

from espnet.utils.cli_utils import strtobool


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of decoding arguments."""
    parser = configargparse.ArgumentParser(
        description="Converting speech using a VC model on one CPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="second config file path that overwrites the settings in `--config`.",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="third config file path that overwrites the settings "
        "in `--config` and `--config2`.",
    )

    parser.add_argument("--ngpu", default=0, type=int, help="Number of GPUs")
    parser.add_argument(
        "--backend",
        default="pytorch",
        type=str,
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--out", type=str, required=True, help="Output filename")
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    # task related
    parser.add_argument(
        "--json", type=str, required=True, help="Filename of train label data (json)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    # decoding related
    parser.add_argument(
        "--maxlenratio", type=float, default=5, help="Maximum length ratio in decoding"
    )
    parser.add_argument(
        "--minlenratio", type=float, default=0, help="Minimum length ratio in decoding"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold value in decoding"
    )
    parser.add_argument(
        "--use-att-constraint",
        type=strtobool,
        default=False,
        help="Whether to use the attention constraint",
    )
    parser.add_argument(
        "--backward-window",
        type=int,
        default=1,
        help="Backward window size in the attention constraint",
    )
    parser.add_argument(
        "--forward-window",
        type=int,
        default=3,
        help="Forward window size in the attention constraint",
    )
    # save related
    parser.add_argument(
        "--save-durations",
        default=False,
        type=strtobool,
        help="Whether to save durations converted from attentions",
    )
    parser.add_argument(
        "--save-focus-rates",
        default=False,
        type=strtobool,
        help="Whether to save focus rates of attentions",
    )
    return parser


def main(args):
    """Run deocding."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose > 0:
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

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == "2":
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(
                    ["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]
                ).strip()
                logging.info("CLSP: use gpu" + cvd)
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = (
                    subprocess.check_output(
                        ["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]
                    )
                    .decode()
                    .strip()
                )
                logging.info("CLSP: use gpu" + cvd)
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # extract
    logging.info("backend = " + args.backend)
    if args.backend == "pytorch":
        from espnet.vc.pytorch_backend.vc import decode

        decode(args)
    else:
        raise NotImplementedError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
