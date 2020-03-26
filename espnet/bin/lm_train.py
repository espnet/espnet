#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

"""Language model training script."""

import logging
import os
import random
import subprocess
import sys

import configargparse
import numpy as np

from espnet.nets.lm_interface import dynamic_import_lm
from espnet.optimizer.factory import dynamic_import_optimizer
from espnet.scheduler.scheduler import dynamic_import_scheduler


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get parser."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description='Train a new language model on one CPU or one GPU',
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True, help='config file path')
    parser.add('--config2', is_config_file=True,
               help='second config file path that overwrites the settings in `--config`.')
    parser.add('--config3', is_config_file=True,
               help='third config file path that overwrites the settings in `--config` and `--config2`.')

    parser.add_argument('--ngpu', default=None, type=int,
                        help='Number of GPUs. If not given, use all visible devices')
    parser.add_argument('--train-dtype', default="float32",
                        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
                        help='Data type for training (only pytorch backend). '
                        'O0,O1,.. flags require apex. See https://nvidia.github.io/apex/amp.html#opt-levels')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=required,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--dict', type=str, required=required,
                        help='Dictionary')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    parser.add_argument('--report-interval-iters', default=100, type=int,
                        help="Report interval iterations")
    # task related
    parser.add_argument('--train-label', type=str, required=required,
                        help='Filename of train label data')
    parser.add_argument('--valid-label', type=str, required=required,
                        help='Filename of validation label data')
    parser.add_argument('--test-label', type=str,
                        help='Filename of test label data')
    parser.add_argument('--dump-hdf5-path', type=str, default=None,
                        help='Path to dump a preprocessed dataset as hdf5')
    # training configuration
    parser.add_argument('--opt', default='sgd', type=str,
                        help='Optimizer')
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batchsize', '-b', type=int, default=300,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--accum-grad', type=int, default=1,
                        help='Number of gradient accumueration')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--schedulers', default=None, action="append", type=lambda kv: kv.split("="),
                        help='optimizer schedulers, you can configure params like:'
                        ' <optimizer-param>-<scheduler-name>-<schduler-param>'
                        ' e.g., "--schedulers lr=noam --lr-noam-warmup 1000".')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--maxlen', type=int, default=40,
                        help='Batch size is reduced if the input sequence > ML')
    parser.add_argument('--model-module', type=str, default='default',
                        help='model defined module (default: espnet.nets.xxx_backend.lm.default:DefaultRNNLM)')
    return parser


def main(cmd_args):
    """Train LM."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32.")
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(f"--train-dtype {args.train_dtype} does not support the CPU backend.")

    # parse arguments dynamically
    model_class = dynamic_import_lm(args.model_module, args.backend)
    model_class.add_arguments(parser)
    if args.schedulers is not None:
        for k, v in args.schedulers:
            scheduler_class = dynamic_import_scheduler(v)
            scheduler_class.add_arguments(k, parser)

    opt_class = dynamic_import_optimizer(args.opt, args.backend)
    opt_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(','))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(['nvidia-smi', '-L'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split('\n')) - 1
        args.ngpu = ngpu
    else:
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)

    # load dictionary
    with open(args.dict, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')
    args.char_list_dict = {x: i for i, x in enumerate(char_list)}
    args.n_vocab = len(char_list)

    # train
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        from espnet.lm.chainer_backend.lm import train
        train(args)
    elif args.backend == "pytorch":
        from espnet.lm.pytorch_backend.lm import train
        train(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
