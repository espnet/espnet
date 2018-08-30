#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import random

import numpy as np

import kaldi_io_py


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, batch_sort_key='shuffle'):
    """Function to make batch set from json dictionary

    :param dict data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param str batch_sort_key: 'shuffle' or 'input' or 'output'
    :return: list of batches
    """
    minibatch = []
    start = 0
    # sort data with batch_sort_key
    if batch_sort_key == 'shuffle':
        logging.info('use shuffled batch.')
        sorted_data = random.sample(data.items(), len(data.items()))
    elif batch_sort_key == 'input':
        logging.info('use batch sorted by input length and adaptive batch size.')
        # sort it by input lengths (long to short)
        # NOTE: input and output are reversed due to the use of same json as asr
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['output'][0]['shape'][0]), reverse=True)
    elif batch_sort_key == 'output':
        logging.info('use batch sorted by output length and adaptive batch size.')
        # sort it by output lengths (long to short)
        # NOTE: input and output are reversed due to the use of same json as asr
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=True)
    else:
        ValueError('batch_sort_key should be selected from None, input, and output.')

    logging.info('# utts: ' + str(len(sorted_data)))

    if batch_sort_key is None:
        # use fixed size batch
        while True:
            end = min(len(sorted_data), start + batch_size)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end
    else:
        # use adaptive batch size
        while True:
            # NOTE: input and output are reversed due to the use of same json as asr
            ilen = int(sorted_data[start][1]['output'][0]['shape'][0])
            olen = int(sorted_data[start][1]['input'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            b = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + b)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end

    # for debugging
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def load_inputs_and_targets(batch, use_speaker_embedding=False, use_second_target=False):
    """Function to load inputs and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :param bool use_speaker_embedding: whether to load speaker embedding vector
    :param bool use_second_target: whether to load second target vector
    :return: list of input token id sequences [(T_1), (T_2), ..., (T_B)]
    :rtype: list of int ndarray
    :return: list of target feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of speaker embedding vectors (only if use_speaker_embedding = True)
    :rtype: list of float adarray
    :return: list of second target feature sequences [(T_1, V), (T_2, V), ..., (T_B, V)]
    :rtype: list of float ndarray
    """

    # load acoustic features and target sequence of token ids
    xs = [b[1]['output'][0]['tokenid'].split() for b in batch]
    ys = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        logging.warning('Input sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(nonzero_sorted_idx)))

    # remove zero-length samples
    xs = [np.fromiter(map(int, xs[i]), dtype=np.int64) for i in nonzero_sorted_idx]
    ys = [ys[i] for i in nonzero_sorted_idx]

    # load second target for CHBG
    if use_second_target:
        spcs = [kaldi_io_py.read_mat(b[1]['input'][1]['feat']) for b in batch]
        spcs = [spcs[i] for i in nonzero_sorted_idx]
    else:
        spcs = None

    # load speaker embedding
    if use_speaker_embedding:
        spembs = [kaldi_io_py.read_vec_flt(b[1]['input'][1]['feat']) for b in batch]
        spembs = [spembs[i] for i in nonzero_sorted_idx]
    else:
        spembs = None

    return xs, ys, spembs, spcs
