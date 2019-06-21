#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import configargparse
import logging
import os
import platform
import random
import subprocess
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool
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
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
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
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related
    parser.add_argument('--train-json', type=str, default=None, nargs='+',
                        help='Filenames of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    # network architecture
    # encoder
    parser.add_argument('--etype', default='blstmp', type=str,
                        choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                 'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                        help='Type of encoder network architecture')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=300, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=320, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default="1", type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')
    # loss
    parser.add_argument('--ctc_type', default='warpctc', type=str,
                        choices=['builtin', 'warpctc'],
                        help='Type of CTC implementation to calculate loss.')
    # attention
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
    # decoder
    parser.add_argument('--dtype', default='lstm', type=str,
                        choices=['lstm', 'gru'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=320, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--mtlalpha', default=0.0, type=float,
                        help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='Multitask learning coefficient, alpha: alpha*sa_loss + (1-alpha)*ta_loss ')
    parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                        help='Apply label smoothing with a specified distribution type')
    parser.add_argument('--lsm-weight', default=0.0, type=float,
                        help='Label smoothing weight')
    parser.add_argument('--sampling-probability', default=0.0, type=float,
                        help='Ratio of predicted labels fed back to decoder')

    # recognition options to compute CER/WER
    parser.add_argument('--report-cer', default=False, action='store_true',
                        help='Compute CER on development set')
    parser.add_argument('--report-wer', default=False, action='store_true',
                        help='Compute WER on development set')
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=4,
                        help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float,
                        help='Insertion penalty')
    parser.add_argument('--maxlenratio', default=0.0, type=float,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', default=0.3, type=float,
                        help='CTC weight in joint decoding')
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
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.0, type=float,
                        help='Dropout rate for the encoder')
    parser.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                        help='Dropout rate for the decoder')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
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
    parser.add_argument('--maxlen-in', '--batch-seq-maxlen-in', default=800, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the input sequence length > ML.')
    parser.add_argument('--maxlen-out', '--batch-seq-maxlen-out', default=150, type=int, metavar='ML',
                        help='When --batch-count=seq, batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n-iter-processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam', 'noam'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumulation')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay ratio')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc'],
                        help='Criterion to perform epsilon decay')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='Threshold to stop iteration')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/acc', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=5, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=3, type=int,
                        help='Number of samples of attention to be saved')

    # transfer learning related
    parser.add_argument('--mt-model', default=False, nargs='?',
                        help='Pre-trained MT model')
    parser.add_argument(
        '--use-frontend', type=strtobool, default=False,
        help='The flag to switch to use frontend system.')

    # cycle-consistency related
    parser.add_argument('--asr-model', default='', type=str,
                        help='ASR initial model')
    parser.add_argument('--asr-model-conf', default='', type=str,
                        help='ASR initial model conf')
    parser.add_argument('--tts-model', default='', type=str,
                        help='TTS model for cycle-consistency loss')
    parser.add_argument('--tts-model-conf', default='', type=str,
                        help='TTS model conf for cycle-consistency loss')
    parser.add_argument('--expected-loss', default='tts', type=str,
                        choices=['tts', 'none', 'wer'],
                        help='Type of expected loss (tts, wer, ...)')
    parser.add_argument('--generator', default='tts', type=str,
                        choices=['tts', 'tte'],
                        help='Type of generator (tts, tte, ...)')
    parser.add_argument('--rnnloss', default='ce', type=str,
                        choices=['ce', 'kl', 'kld', 'mmd'],
                        help='RNNLM loss function')
    parser.add_argument('--n-samples-per-input', default=5, type=int,
                        help='Number of samples per input generated from model')
    parser.add_argument('--sample-maxlenratio', default=0.8, type=float,
                        help='Maximum length ratio of each sample to input length')
    parser.add_argument('--sample-minlenratio', default=0.2, type=float,
                        help='Minimum length ratio of each sample to input length')
    parser.add_argument('--sample-topk', default=0, type=int,
                        help='Sample from top-K labels')
    parser.add_argument('--sample-oracle', default=False, action='store_true',
                        help='Oracle sampling of utterance')
    parser.add_argument('--use-speaker-embedding', default=False, action='store_true',
                        help='Intake speaker embedding')
    parser.add_argument('--modify-output', default=False, action='store_true',
                        help='Replace output layer')
    parser.add_argument('--sample-scaling', default=0.1, type=float,
                        help='Scaling factor for sample log-likelihood')
    parser.add_argument('--policy-gradient', action='store_true',
                        help='Policy gradient mode')
    parser.add_argument('--teacher-weight', default=1.0, type=float,
                        help='Weight for teacher forcing loss')
    parser.add_argument('--update-asr-only', action='store_true',
                        help='Update ASR model only')
    parser.add_argument('--plot-corr', action='store_true',
                        help='Get tts loss and cer, wer for plot')
    parser.add_argument('--freeze', default='none', type=str,
                        choices=['attdec', 'dec', 'att', 'encattdec', 'encatt', 'enc', 'none'],
                        help='parameters to be frozen in asr')
    parser.add_argument('--freeze-asr', action='store_true',
                        help='Freeze ASR parameters')
    parser.add_argument('--freeze-tts', action='store_true',
                        help='Freeze TTS parameters')
    parser.add_argument('--check', action='store_true',
                        help='check a new implementation')
    parser.add_argument('--zero-att', action='store_true',
                        help='Zero Att for TTS->ASR')
    parser.add_argument('--softargmax', action='store_true',
                        help='Soft assignment of token embeddings to TTS input')
    parser.add_argument('--lm-loss-weight', default=1.0, type=float,
                        help='LM loss weight')
    args, _ = parser.parse_known_args(cmd_args)

    from espnet.utils.dynamic_import import dynamic_import
    if args.model_module is not None:
        model_class = dynamic_import(args.model_module)
        model_class.add_arguments(parser)
    args = parser.parse_args(cmd_args)
    if args.model_module is None:
        args.model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    if 'chainer_backend' in args.model_module:
        args.backend = 'chainer'
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
        from espnet.asr.pytorch_backend.asrtts import train
        train(args)
    else:
        raise ValueError("pytorch is only supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
