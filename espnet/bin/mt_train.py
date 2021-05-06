#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Neural machine translation model training script."""

import logging
import os
import random
import subprocess
import sys

from distutils.version import LooseVersion

import configargparse
import numpy as np
import torch

from espnet import __version__
from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2")


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train a neural machine translation (NMT) model on one CPU, "
            "one or multiple GPUs",
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

    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--train-dtype",
        default="float32",
        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
        help="Data type for training (only pytorch backend). "
        "O0,O1,.. flags require apex. "
        "See https://nvidia.github.io/apex/amp.html#opt-levels",
    )
    parser.add_argument(
        "--backend",
        default="chainer",
        type=str,
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument(
        "--outdir", type=str, required=required, help="Output directory"
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument(
        "--dict", required=required, help="Dictionary for source/target languages"
    )
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--debugdir", type=str, help="Output directory for debugging")
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="-1",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log dir path",
    )
    parser.add_argument(
        "--report-interval-iters",
        default=100,
        type=int,
        help="Report interval iterations",
    )
    parser.add_argument(
        "--save-interval-iters",
        default=0,
        type=int,
        help="Save snapshot interval iterations",
    )
    # task related
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default=None,
        help="Filename of validation label data (json)",
    )
    # network architecture
    parser.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="model defined module (default: espnet.nets.xxx_backend.e2e_mt:E2E)",
    )
    # loss related
    parser.add_argument(
        "--lsm-weight", default=0.0, type=float, help="Label smoothing weight"
    )
    # translations options to compute BLEU
    parser.add_argument(
        "--report-bleu",
        default=True,
        action="store_true",
        help="Compute BLEU on development set",
    )
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size")
    parser.add_argument("--penalty", default=0.0, type=float, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        default=0.0,
        type=float,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        default=0.0,
        type=float,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument("--lm-weight", default=0.0, type=float, help="RNNLM weight.")
    parser.add_argument("--sym-space", default="<space>", type=str, help="Space symbol")
    parser.add_argument("--sym-blank", default="<blank>", type=str, help="Blank symbol")
    # minibatch related
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batch-count",
        default="auto",
        choices=BATCH_COUNT_CHOICES,
        help="How to count batch_size. "
        "The default (auto) will find how to count by args.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-seqs",
        "-b",
        default=0,
        type=int,
        help="Maximum seqs in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-bins",
        default=0,
        type=int,
        help="Maximum bins in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-in",
        default=0,
        type=int,
        help="Maximum input frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-out",
        default=0,
        type=int,
        help="Maximum output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-inout",
        default=0,
        type=int,
        help="Maximum input+output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--maxlen-in",
        "--batch-seq-maxlen-in",
        default=100,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the input sequence length > ML.",
    )
    parser.add_argument(
        "--maxlen-out",
        "--batch-seq-maxlen-out",
        default=100,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the output sequence length > ML",
    )
    parser.add_argument(
        "--n-iter-processes",
        default=0,
        type=int,
        help="Number of processes of iterator",
    )
    # optimization related
    parser.add_argument(
        "--opt",
        default="adadelta",
        type=str,
        choices=["adadelta", "adam", "noam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--accum-grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--lr-decay", default=1.0, type=float, help="Decaying ratio of learning rate"
    )
    parser.add_argument(
        "--weight-decay", default=0.0, type=float, help="Weight decay ratio"
    )
    parser.add_argument(
        "--criterion",
        default="acc",
        type=str,
        choices=["loss", "acc"],
        help="Criterion to perform epsilon decay",
    )
    parser.add_argument(
        "--threshold", default=1e-4, type=float, help="Threshold to stop iteration"
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/acc",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait "
        "without improvement before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=5, type=float, help="Gradient norm threshold to clip"
    )
    parser.add_argument(
        "--num-save-attention",
        default=3,
        type=int,
        help="Number of samples of attention to be saved",
    )
    # decoder related
    parser.add_argument(
        "--context-residual",
        default=False,
        type=strtobool,
        nargs="?",
        help="The flag to switch to use context vector residual in the decoder network",
    )
    parser.add_argument(
        "--tie-src-tgt-embedding",
        default=False,
        type=strtobool,
        nargs="?",
        help="Tie parameters of source embedding and target embedding.",
    )
    parser.add_argument(
        "--tie-classifier",
        default=False,
        type=strtobool,
        nargs="?",
        help="Tie parameters of target embedding and output projection layer.",
    )
    # finetuning related
    parser.add_argument(
        "--enc-init",
        default=None,
        type=str,
        nargs="?",
        help="Pre-trained ASR model to initialize encoder.",
    )
    parser.add_argument(
        "--enc-init-mods",
        default="enc.enc.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--dec-init",
        default=None,
        type=str,
        nargs="?",
        help="Pre-trained ASR, MT or LM model to initialize decoder.",
    )
    parser.add_argument(
        "--dec-init-mods",
        default="att., dec.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of decoder modules to initialize, separated by a comma.",
    )
    # multilingual related
    parser.add_argument(
        "--multilingual",
        default=False,
        type=strtobool,
        help="Prepend target language ID to the source sentence. "
        "Both source/target language IDs must be prepend in the pre-processing stage.",
    )
    parser.add_argument(
        "--replace-sos",
        default=False,
        type=strtobool,
        help="Replace <sos> in the decoder with a target language ID "
        "(the first token in the target sequence)",
    )

    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32."
        )
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(
            f"--train-dtype {args.train_dtype} does not support the CPU backend."
        )

    from espnet.utils.dynamic_import import dynamic_import

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_mt:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if "chainer_backend" in args.model_module:
        args.backend = "chainer"
    if "pytorch_backend" in args.model_module:
        args.backend = "pytorch"

    # add version info in args
    args.version = __version__

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

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(","))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(
                    ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split("\n")) - 1
        args.ngpu = ngpu
    else:
        if is_torch_1_2_plus and args.ngpu != 1:
            logging.debug(
                "There are some bugs with multi-GPU processing in PyTorch 1.2+"
                + " (see https://github.com/pytorch/pytorch/issues/21108)"
            )
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info("backend = " + args.backend)

    if args.backend == "pytorch":
        from espnet.mt.pytorch_backend.mt import train

        train(args)
    else:
        raise ValueError("Only pytorch are supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
