#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
import random
import logging
import numpy as np
import pickle
import json

# chainer related
import chainer

# spnet related
from e2e_asr_attctc import E2E
from e2e_asr_attctc import MTLLoss

# for kaldi io
import kaldi_io


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
    parser.add_argument('--maxlenratio', default=0.5, type=float,
                        help='Input length ratio to obtain max output length')
    parser.add_argument('--minlenratio', default=0.0, type=float,
                        help='Input length ratio to obtain min output length')
    args = parser.parse_args()

    # logging info
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    if args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
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
    with open(args.model_conf, "r") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    logging.info('reading model parameters from' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = MTLLoss(e2e, train_args.mtlalpha)
    chainer.serializers.load_npz(args.model, model)

    # prepare Kaldi reader
    reader = kaldi_io.SequentialBaseFloatMatrixReader(args.recog_feat)

    # read json data
    with open(args.recog_label, 'r') as f:
        recog_json = json.load(f)['utts']

    new_json = {}
    for name, feat in reader:
        y_hat = e2e.recognize(feat, args, train_args.char_list)
        y_true = map(int, recog_json[name]['tokenid'].split())

        # print out decoding result
        seq_hat = [train_args.char_list[int(idx)] for idx in y_hat]
        seq_true = [train_args.char_list[int(idx)] for idx in y_true]
        seq_hat_text = "".join(seq_hat)
        seq_true_text = "".join(seq_true)
        logging.info("groundtruth[%s]: " + seq_true_text, name)
        logging.info("prediction [%s]: " + seq_hat_text, name)

        # copy old json info
        new_json[name] = recog_json[name]

        # added recognition results to json
        new_json[name]['rec_tokenid'] = " ".join([str(idx[0]) for idx in y_hat])
        new_json[name]['rec_token'] = " ".join(seq_hat)
        new_json[name]['rec_text'] = seq_hat_text

    # TODO fix character coding problems when saving it
    with open(args.result_label, 'w') as f:
        f.write(json.dumps({'utts': new_json}, indent=4).encode('utf_8'))


if __name__ == '__main__':
    main()
