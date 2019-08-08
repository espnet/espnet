#!/usr/bin/env python
# encoding: utf-8

import configargparse
import logging
import os
import random
import subprocess
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Train an automatic speech recognition (ASR) model on one CPU, one or multiple GPUs",
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
                        choices=['pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--dict', required=True,
                        help='Dictionary')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--debugdir', type=str,
                        help='Output directory for debugging')
    parser.add_argument('--resume', type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related
    parser.add_argument('--train-json', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    # network architecture
    parser.add_argument('--model-module', type=str, default=None,
                        help='model defined module (default: espnet.nets.xxx_backend.e2e_asr)')
    # encoder
    parser.add_argument('--etype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
    parser.add_argument('--eunits', '-u', default=300, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=320, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default=1, type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')
    # decoder related
    parser.add_argument('--dtype', default='lstm', type=str,
                        choices=['lstm', 'gru'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=320, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--dec-embed-dim', default=320, type=int,
                        help='Number of decoder embeddings dimensions')
    # attention related
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                 'coverage_location', 'location2d', 'location_recurrent',
                                 'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                 'multi_head_multi_res_loc'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=320, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--awin', default=5, type=int,
                        help='Window size for location2d attention')
    parser.add_argument('--aheads', default=4, type=int,
                        help='Number of heads for multi head attention')
    parser.add_argument('--aconv-chans', default=-1, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--aconv-filts', default=100, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--spa', action='store_true',
                        help='Enable speaker parallel attention.')
    # loss
    parser.add_argument('--rnnt_type', default='warp-transducer', type=str,
                        choices=['warp-transducer'],
                        help='Type of RNN Transducer implementation to calculate loss.')
    # recognition options to compute CER/WER
    parser.add_argument('--report-cer', default=False, action='store_true',
                        help='Compute CER on development set')
    parser.add_argument('--report-wer', default=False, action='store_true',
                        help='Compute WER on development set')
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=4,
                        help='Beam size')
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--lm-weight', default=0.1, type=float,
                        help='RNNLM weight.')
    parser.add_argument('--sym-space', default='<space>', type=str,
                        help='Space symbol')
    parser.add_argument('--sym-blank', default='<blank>', type=str,
                        help='Blank symbol')
    # model related
    parser.add_argument('--rnnt-mode', default=0, type=int, choices=[0, 1],
                        help='RNN-Transducing mode (0:rnnt, 1:rnnt-att)')
    parser.add_argument('--joint-dim', default=320, type=int,
                        help='Number of dimensions in joint space')
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                        help='Dropout rate for the decoder')
    parser.add_argument('--dropout-rate-embed-decoder', default=0.0, type=float,
                        help='Dropout rate for the decoder embeddings')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch-count', default='auto', choices=BATCH_COUNT_CHOICES,
                        help='How to count batch_size. The default (auto) will find how to count by args.')
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--batch-bins', default=0, type=int,
                        help='Maximum bins in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-in', default=0, type=int,
                        help='Maximum input frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-out', default=0, type=int,
                        help='Maximum output frames in a minibatch (0 to disable)')
    parser.add_argument('--batch-frames-inout', default=0, type=int,
                        help='Maximum input+output frames in a minibatch (0 to disable)')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumulation')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay ratio')
    parser.add_argument('--criterion', default='loss', type=str,
                        choices=['loss'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=3, type=int,
                        help='Number of samples of attention to be saved')
    parser.add_argument('--grad-noise', type=strtobool, default=False,
                        help='The flag to switch to use noise injection to gradients during training')
    # finetuning related
    parser.add_argument('--enc-init', default=None, type=str,
                        help='Initialize encoder model part from pre-trained ESPNET ASR model.')
    parser.add_argument('--enc-init-mods', default='enc.enc.',
                        type=lambda s: [str(mod) for mod in s.split(',') if s != ''],
                        help='List of encoder modules to initialize, separated by a comma.')
    parser.add_argument('--dec-init', default=None, type=str,
                        help='Initialize decoder model part from pre-trained ESPNET ASR or LM model.')
    parser.add_argument('--dec-init-mods', default='att.,dec.decoder.,dec.att.,dec.embed.',
                        type=lambda s: [str(mod) for mod in s.split(',') if s != ''],
                        help='List of decoder modules to initialize, separated by a comma.')
    parser.add_argument('--freeze-modules', default='',
                        type=lambda s: [str(mod) for mod in s.split(',') if s != ''],
                        help='List of modules to freeze, separated by a comma.')
    # speech translation related
    parser.add_argument('--use-frontend', type=strtobool, default=False,
                        help='The flag to switch to use frontend system.')
    # WPE related
    parser.add_argument('--use-wpe', type=strtobool, default=False,
                        help='Apply Weighted Prediction Error')
    parser.add_argument('--wtype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm',
                                 'vggblstm', 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup',
                                 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture of the mask estimator for WPE.')
    parser.add_argument('--wlayers', type=int, default=2,
                        help='')
    parser.add_argument('--wunits', type=int, default=300,
                        help='')
    parser.add_argument('--wprojs', type=int, default=300,
                        help='')
    parser.add_argument('--wdropout-rate', type=float, default=0.0,
                        help='')
    parser.add_argument('--wpe-taps', type=int, default=5,
                        help='')
    parser.add_argument('--wpe-delay', type=int, default=3,
                        help='')
    parser.add_argument('--use-dnn-mask-for-wpe', type=strtobool,
                        default=False,
                        help='Use DNN to estimate the power spectrogram. '
                        'This option is experimental.')
    # Beamformer related
    parser.add_argument('--use-beamformer', type=strtobool,
                        default=True, help='')
    parser.add_argument('--btype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm',
                                 'vggblstm', 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup',
                                 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture '
                        'of the mask estimator for Beamformer.')
    parser.add_argument('--blayers', type=int, default=2,
                        help='')
    parser.add_argument('--bunits', type=int, default=300,
                        help='')
    parser.add_argument('--bprojs', type=int, default=300,
                        help='')
    parser.add_argument('--badim', type=int, default=320,
                        help='')
    parser.add_argument('--ref-channel', type=int, default=-1,
                        help='The reference channel used for beamformer. '
                        'By default, the channel is estimated by DNN.')
    parser.add_argument('--bdropout-rate', type=float, default=0.0,
                        help='')
    # Feature transform: Normalization
    parser.add_argument('--stats-file', type=str, default=None,
                        help='The stats file for the feature normalization')
    parser.add_argument('--apply-uttmvn', type=strtobool, default=True,
                        help='Apply utterance level mean '
                        'variance normalization.')
    parser.add_argument('--uttmvn-norm-means', type=strtobool,
                        default=True, help='')
    parser.add_argument('--uttmvn-norm-vars', type=strtobool, default=False,
                        help='')
    # Feature transform: Fbank
    parser.add_argument('--fbank-fs', type=int, default=16000,
                        help='The sample frequency used for '
                        'the mel-fbank creation.')
    parser.add_argument('--n-mels', type=int, default=80,
                        help='The number of mel-frequency bins.')
    parser.add_argument('--fbank-fmin', type=float, default=0.,
                        help='')
    parser.add_argument('--fbank-fmax', type=float, default=None,
                        help='')

    return parser


def main(cmd_args):
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    if args.model_module is not None:
        model_class = dynamic_import(args.model_module)
        model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)
    if args.model_module is None:
        args.model_module = "espnet.nets." + args.backend + "_backend.e2e_asr_rnnt:E2E"
        if 'pytorch_backend' in args.model_module:
            args.backend = 'pytorch'

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
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
    else:
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info('backend = ' + args.backend)

    if args.backend == "pytorch":
        from espnet.asr.pytorch_backend.asr_rnnt import train
        train(args)
    else:
        raise ValueError("Only pytorch is supported for RNN-Transducer.")


if __name__ == '__main__':
    main(sys.argv[1:])
