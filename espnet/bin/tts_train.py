#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import configargparse
import logging
import os
import platform
import random
import subprocess
import sys

from distutils.util import strtobool

import numpy as np

from espnet.nets.tts_interface import TTSInterface
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES


def main(cmd_args):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # general configuration
    parser.add('--config', is_config_file=True, help='config file path')
    parser.add('--config2', is_config_file=True,
               help='second config file path that overwrites the settings in `--config`.')
    parser.add('--config3', is_config_file=True,
               help='third config file path that overwrites the settings in `--config` and `--config2`.')

    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related
    parser.add_argument('--train-json', type=str, required=True,
                        help='Filename of training json')
    parser.add_argument('--valid-json', type=str, required=True,
                        help='Filename of validation json')
    # network architecture
    parser.add_argument('--model-module', type=str, default="espnet.nets.pytorch_backend.e2e_tts:Tacotron2",
                        help='model defined module')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch_sort_key', default='shuffle', type=str,
                        choices=['shuffle', 'output', 'input'], nargs='?',
                        help='Batch sorting key. "shuffle" only work with --batch-count "seq".')
    parser.add_argument('--batch-count', default='auto', choices=BATCH_COUNT_CHOICES,
                        help='How to count batch_size. The default (auto) will find how to count by args.')
    parser.add_argument('--batch-size', '-b', default=0, type=int,
                        help='Maximum seqs in a minibatch (0 to disable)')
    parser.add_argument('--batch-bins', default=0, type=int,
                        help='Maximum bins in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-in', default=0, type=int,
                        help='Maximum input frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-out', default=0, type=int,
                        help='Maximum output frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-inout', default=0, type=int,
                        help='Maximum input+output frames in a minibatch (0 to disable)')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the input sequence length > ML.')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--use_speaker_embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use_second_target', default=False, type=strtobool,
                        help='Whether to use second target')
    # optimization related
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=1, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=5, type=int,
                        help='Number of samples of attention to be saved')

    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    model_class = dynamic_import(args.model_module)
    assert issubclass(model_class, TTSInterface)
    model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        # python 2 case
        if platform.python_version_tuple()[0] == '2':
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]):
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.backend == "pytorch":
        from espnet.tts.pytorch_backend.tts import train
        train(args)
    else:
        raise NotImplementedError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
