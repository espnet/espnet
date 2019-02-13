#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import random

import numpy as np


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, batch_sort_key='shuffle', min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    :param dict data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param str batch_sort_key: 'shuffle' or 'input' or 'output'
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :return: list of batches
    """
    # sort data with batch_sort_key
    if batch_sort_key == 'shuffle':
        logging.info('use shuffled batch.')
        sorted_data = random.sample(data.items(), len(data.items()))
    elif batch_sort_key == 'input':
        logging.info('use batch sorted by input length and adaptive batch size.')
        # sort it by input lengths (long to short)
        # NOTE: input and output are reversed due to the use of same json as asr
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['output'][0]['shape'][0]), reverse=not shortest_first)
    elif batch_sort_key == 'output':
        logging.info('use batch sorted by output length and adaptive batch size.')
        # sort it by output lengths (long to short)
        # NOTE: input and output are reversed due to the use of same json as asr
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=not shortest_first)
    else:
        raise ValueError('batch_sort_key should be selected from None, input, and output.')
    logging.info('# utts: ' + str(len(sorted_data)))

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        if batch_sort_key == 'shuffle':
            end = min(len(sorted_data), start + batch_size)
        else:
            # NOTE: input and output are reversed due to the use of same json as asr
            ilen = int(sorted_data[start][1]['output'][0]['shape'][0])
            olen = int(sorted_data[start][1]['input'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # change batchsize depending on the input and output length
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            bs = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + bs)

        # check each batch is more than minimum batchsize
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i] for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # for debugging
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatches)))

    return minibatches
