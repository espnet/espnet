#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import platform
import random
import subprocess
import sys

import numpy as np


def main(args):
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='chainer', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--dict-tgt', required=True,
                        help='Dictionary for target language')
    parser.add_argument('--dict-src', default='', nargs='?',
                        help='Dictionary for source language. \
                        Dictionanies are shared between soruce and target languages in default setting.')
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
    parser.add_argument('--train-json', type=str, default=None,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-json', type=str, default=None,
                        help='Filename of validation label data (json)')
    # network architecture
    # encoder
    parser.add_argument('--etype', default='blstmp', type=str,
                        choices=['blstm', 'blstmp', 'vggblstmp', 'vggblstm'],
                        help='Type of encoder network architecture')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=1024, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--eprojs', default=1024, type=int,
                        help='Number of encoder projection units')
    parser.add_argument('--subsample', default=1, type=str,
                        help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                             'every y frame at 2nd layer etc.')
    # attention
    parser.add_argument('--atype', default='dot', type=str,
                        choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                 'coverage_location', 'location2d', 'location_recurrent',
                                 'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                 'multi_head_multi_res_loc',
                                 'luong_dot', 'luong_general'],
                        help='Type of attention architecture')
    parser.add_argument('--adim', default=1024, type=int,
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
                        choices=['lstm'],
                        help='Type of decoder network architecture')
    parser.add_argument('--dlayers', default=1, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                        help='Apply label smoothing with a specified distribution type')
    parser.add_argument('--lsm-weight', default=0.0, type=float,
                        help='Label smoothing weight')
    parser.add_argument('--sampling-probability', default=0.0, type=float,
                        help='Ratio of predicted labels fed back to decoder')
    # recognition options to compute CER/WER
    parser.add_argument('--report-bleu', default=True, action='store_true',
                        help='Compute BLEU on development set')
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=4,
                        help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', default=0.0, type=float,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--lm-weight', default=0.0, type=float,
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
    parser.add_argument('--batch-size', '-b', default=50, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=100, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=100, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--opt', default='adadelta', type=str,
                        choices=['adadelta', 'adam'],
                        help='Optimizer')
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon constant for optimizer')
    parser.add_argument('--eps-decay', default=0.01, type=float,
                        help='Decaying ratio of epsilon')
    parser.add_argument('--weight-decay', default=0.0, type=float,
                        help='Weight decay ratio')
    parser.add_argument('--criterion', default='acc', type=str,
                        choices=['loss', 'acc', 'bleu'],
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
    # decoder related
    parser.add_argument('--input-feeding', default='', nargs='?',
                        help='')
    parser.add_argument('--context-residual', default='', nargs='?',
                        help='')
    # multilingual NMT related
    parser.add_argument('--multilingual', default=False, nargs='?',
                        help='')
    parser.add_argument('--replace-sos', default=False, nargs='?',
                        help='Replace <sos> in the decoder with a target language ID \
                              (the first token in the target sequence)')
    parser.add_argument('--target-forcing', default=False, nargs='?',
                        help='Feed a target language ID (the first token in the target sentence) \
                              to the beggining of the source sentence \
                              (also to the end to the source sentence if the encoder is bidirectional)')
    parser.add_argument('--language-coding', default=False, nargs='?',
                        help='Feed a target language ID (the first token in the target sentence) \
                              to every token in the source sentence')

    args = parser.parse_args(args)

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

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict_tgt is not None:
        with open(args.dict_tgt, 'rb') as f:
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
    if args.backend == "chainer":
        raise NotImplementedError()
    # TODO(hirfumi): support for chainer backend
        # from espnet.mt.chainer_backend.mt import train
        # train(args)
    elif args.backend == "pytorch":
        from espnet.mt.pytorch_backend.mt import train
        train(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
