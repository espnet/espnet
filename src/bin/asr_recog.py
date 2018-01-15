#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import json
import logging
import os
import pickle
import random

# chainer related
import chainer
import numpy as np

# spnet related
from e2e_asr_attctc import E2E
from e2e_asr_attctc import Loss

# for kaldi io
import kaldi_io_py

# rnnlm
import lm_train


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--gpu', '-g', default='-1', type=str,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--recog-feat', type=str, required=True,
                        help='Filename of recognition feature data (Kaldi scp)')
    parser.add_argument('--recog-label', type=str, required=True,
                        help='Filename of recognition label data (json)')
    parser.add_argument('--result-label', type=str, required=True,
                        help='Filename of result label data (json)')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, required=True,
                        help='Model config file')
    # search related
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain max output length.'
                        + 'If maxlenratio=0.0 (default), it uses a end-detect function'
                        + 'to automatically find maximum hypothesis lengths')
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', default=0.0, type=float,
                        help='CTC weight in joint decoding')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--lm-weight', default=0.1, type=float,
                        help='RNNLM weight.')
    args = parser.parse_args()

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # display PYTHONPATH
    logging.info('python path = ' + os.environ['PYTHONPATH'])

    # display chainer version
    logging.info('chainer version = ' + chainer.__version__)

    # seed setting (chainer seed may not need it)
    nseed = args.seed
    random.seed(nseed)
    np.random.seed(nseed)
    os.environ["CHAINER_SEED"] = str(nseed)
    logging.info('chainer seed = ' + os.environ['CHAINER_SEED'])

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    chainer.serializers.load_npz(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm = lm_train.ClassifierWithState(lm_train.RNNLM(len(train_args.char_list), 650))
        chainer.serializers.load_npz(args.rnnlm, rnnlm)

    # prepare Kaldi reader
    reader = kaldi_io_py.read_mat_ark(args.recog_feat)

    # read json data
    with open(args.recog_label, 'rb') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name, feat in reader:
        logging.info('decoding ' + name)
        y_hat = e2e.recognize(feat, args, train_args.char_list, rnnlm)
        y_true = map(int, recog_json[name]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
        seq_true_text = "".join(seq_true).replace('<space>', ' ')
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = recog_json[name]

        # added recognition results to json
        new_json[name]['rec_tokenid'] = " ".join(
            [str(idx[0]) for idx in y_hat])
        new_json[name]['rec_token'] = " ".join(seq_hat)
        new_json[name]['rec_text'] = seq_hat_text

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_json}, indent=4).encode('utf_8'))


if __name__ == '__main__':
    main()
