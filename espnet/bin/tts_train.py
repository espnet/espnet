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

import numpy as np

from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES


def main(cmd_args):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

<<<<<<< HEAD
=======
def main(args):
    parser = argparse.ArgumentParser()
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    # general configuration
    parser.add('--config', is_config_file=True,
               help='config file path')
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
<<<<<<< HEAD
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?',
                        help="Tensorboard log directory path")
    parser.add_argument('--save-interval-epochs', default=1, type=int,
                        help="Save interval epochs")
    parser.add_argument('--report-interval-iters', default=100, type=int,
                        help="Report interval iterations")
=======
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    # task related
    parser.add_argument('--train-json', type=str, required=True,
                        help='Filename of training json')
    parser.add_argument('--valid-json', type=str, required=True,
                        help='Filename of validation json')
    # network architecture
<<<<<<< HEAD
    parser.add_argument('--model-module', type=str, default="espnet.nets.pytorch_backend.e2e_tts_tacotron2:Tacotron2",
                        help='model defined module')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-sort-key', default='shuffle', type=str,
=======
    # encoder
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--elayers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=512, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--econv_layers', default=3, type=int,
                        help='Number of encoder convolution layers')
    parser.add_argument('--econv_chans', default=512, type=int,
                        help='Number of encoder convolution channels')
    parser.add_argument('--econv_filts', default=5, type=int,
                        help='Filter size of encoder convolution')
    # attention
    parser.add_argument('--atype', default="location", type=str,
                        choices=["forward_ta", "forward", "location"],
                        help='Type of attention mechanism')
    parser.add_argument('--adim', default=512, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--aconv-chans', default=32, type=int,
                        help='Number of attention convolution channels')
    parser.add_argument('--aconv-filts', default=15, type=int,
                        help='Filter size of attention convolution')
    parser.add_argument('--cumulate_att_w', default=True, type=strtobool,
                        help="Whether or not to cumulate attention weights")
    # decoder
    parser.add_argument('--dlayers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--prenet_layers', default=2, type=int,
                        help='Number of prenet layers')
    parser.add_argument('--prenet_units', default=256, type=int,
                        help='Number of prenet hidden units')
    parser.add_argument('--postnet_layers', default=5, type=int,
                        help='Number of postnet layers')
    parser.add_argument('--postnet_chans', default=512, type=int,
                        help='Number of postnet channels')
    parser.add_argument('--postnet_filts', default=5, type=int,
                        help='Filter size of postnet')
    parser.add_argument('--output_activation', default=None, type=str, nargs='?',
                        help='Output activation function')
    # cbhg
    parser.add_argument('--use_cbhg', default=False, type=strtobool,
                        help='Whether to use CBHG module')
    parser.add_argument('--cbhg_conv_bank_layers', default=8, type=int,
                        help='Number of convoluional bank layers in CBHG')
    parser.add_argument('--cbhg_conv_bank_chans', default=128, type=int,
                        help='Number of convoluional bank channles in CBHG')
    parser.add_argument('--cbhg_conv_proj_filts', default=3, type=int,
                        help='Filter size of convoluional projection layer in CBHG')
    parser.add_argument('--cbhg_conv_proj_chans', default=256, type=int,
                        help='Number of convoluional projection channels in CBHG')
    parser.add_argument('--cbhg_highway_layers', default=4, type=int,
                        help='Number of highway layers in CBHG')
    parser.add_argument('--cbhg_highway_units', default=128, type=int,
                        help='Number of highway units in CBHG')
    parser.add_argument('--cbhg_gru_units', default=256, type=int,
                        help='Number of GRU units in CBHG')
    # model (parameter) related
    parser.add_argument('--use_speaker_embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use_batch_norm', default=True, type=strtobool,
                        help='Whether to use batch normalization')
    parser.add_argument('--use_concate', default=True, type=strtobool,
                        help='Whether to concatenate encoder embedding with decoder outputs')
    parser.add_argument('--use_residual', default=True, type=strtobool,
                        help='Whether to use residual connection in conv layer')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate')
    parser.add_argument('--zoneout', default=0.1, type=float,
                        help='Zoneout rate')
    parser.add_argument('--reduction_factor', default=1, type=int,
                        help='Reduction factor')
    # loss related
    parser.add_argument('--use_masking', default=False, type=strtobool,
                        help='Whether to use masking in calculation of loss')
    parser.add_argument('--bce_pos_weight', default=20.0, type=float,
                        help='Positive sample weight in BCE calculation (only for use_masking=True)')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch_sort_key', default='shuffle', type=str,
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
                        choices=['shuffle', 'output', 'input'], nargs='?',
                        help='Batch sorting key. "shuffle" only work with --batch-count "seq".')
    parser.add_argument('--batch-count', default='auto', choices=BATCH_COUNT_CHOICES,
                        help='How to count batch_size. The default (auto) will find how to count by args.')
    parser.add_argument('--batch-size', '--batch-seqs', '-b', default=0, type=int,
                        help='Maximum seqs in a minibatch (0 to disable)')
    parser.add_argument('--batch-bins', default=0, type=int,
                        help='Maximum bins in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-in', default=0, type=int,
                        help='Maximum input frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-out', default=0, type=int,
                        help='Maximum output frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-inout', default=0, type=int,
                        help='Maximum input+output frames in a minibatch (0 to disable)')
    parser.add_argument('--maxlen-in', '--batch-seq-maxlen-in', default=100, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the input sequence length > ML.')
    parser.add_argument('--maxlen-out', '--batch-seq-maxlen-out', default=200, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the output sequence length > ML')
    parser.add_argument('--num-iter-processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
<<<<<<< HEAD
    parser.add_argument('--use-speaker-embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use-second-target', default=False, type=strtobool,
                        help='Whether to use second target')
=======
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    # optimization related
    parser.add_argument('--opt', default='adam', type=str,
                        choices=['adam'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumuration')
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
<<<<<<< HEAD

    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    model_class = dynamic_import(args.model_module)
    assert issubclass(model_class, TTSInterface)
    model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)
=======
    args = parser.parse_args(args)
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986

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
            elif "fit.vutbr.cz" in subprocess.check_output(["hostname", "-f"]):
                command = 'nvidia-smi --query-gpu=memory.free,memory.total \
                    --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1 / $3 > 0.98) print NR-1}\''
                try:
                    cvd = str(subprocess.check_output(command, shell=True).rsplit('\n')[0:args.ngpu])
                    cvd = cvd.replace("]", "")
                    cvd = cvd.replace("[", "")
                    cvd = cvd.replace("'", "")
                    logging.info("Selected GPU is: " + str(cvd))
                    os.environ['CUDA_VISIBLE_DEVICES'] = cvd
                except subprocess.CalledProcessError:
                    logging.info("No GPU seems to be available")
        # python 3 case
        else:
            if "clsp.jhu.edu" in subprocess.check_output(["hostname", "-f"]).decode():
                cvd = subprocess.check_output(["/usr/local/bin/free-gpu", "-n", str(args.ngpu)]).decode().strip()
                logging.info('CLSP: use gpu' + cvd)
                os.environ['CUDA_VISIBLE_DEVICES'] = cvd
            elif "fit.vutbr.cz" in subprocess.check_output(["hostname", "-f"]).decode():
                command = 'nvidia-smi --query-gpu=memory.free,memory.total \
                    --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1 / $3 > 0.98) print NR-1}\''
                try:
                    cvd = str(subprocess.check_output(command, shell=True).decode().rsplit('\n')[0:args.ngpu])
                    cvd = cvd.replace("]", "")
                    cvd = cvd.replace("[", "")
                    cvd = cvd.replace("'", "")
                    logging.info("Selected GPU is: " + str(cvd))
                    os.environ['CUDA_VISIBLE_DEVICES'] = cvd
                except subprocess.CalledProcessError:
                    logging.info("No GPU seems to be available")
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
