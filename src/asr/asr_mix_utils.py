#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import os
import shutil
import tempfile

import numpy as np
import torch

# chainer related
import chainer

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer
from chainer import training
from chainer.training import extension

# io related
import kaldi_io_py

# matplotlib related
import matplotlib
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *
def make_batchset(data, batch_size, max_length_in, max_length_out, num_batches=0):
    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['input'][0]['shape'][0]), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))
    # change batchsize depending on the input and output length
    minibatch = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
        olen = max(map(lambda x: x['shape'][0], sorted_data[start][1]['output']))
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
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def load_inputs_and_targets(batch):
    """Function to load inputs and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :return: list of input feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target token id sequences [(L_1, L'_1), (L_2, L'_2), ..., (L_B, L'_B)]
    :rtype: list of int ndarray
    """
    # load acoustic features and target sequence of token ids
    xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
    ys_sd = []
    num_spkrs = len(b[1]['output'])
    for ns in range(num_spkrs):
        ys_sd.append([b[1]['output'][ns]['tokenid'].split() for b in batch])

    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys_sd[0][i]) > 0, range(len(xs)))
    for ns in range(1, num_spkrs):
        nonzero_idx = filter(lambda i: len(ys_sd[ns][i]) > 0, nonzero_idx)

    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
            len(xs), len(nonzero_sorted_idx)))

    # remove zero-length samples
    xs = [xs[i] for i in nonzero_sorted_idx]
    ys = []
    for ns in range(num_spkrs):
        ys.append([np.fromiter(map(int, ys_sd[ns][i]), dtype=np.int64) for i in nonzero_sorted_idx])
    ys = zip(*ys)

    return xs, ys


# * -------------------- chainer extension related -------------------- *
class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param function converter: function to convert data
    :param int device: device id
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws_sd = self.att_vis_fn(*batch)
        for i, att_ws in enumerate(att_ws_sd):
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{.updater.epoch}.output%d.png" % (
                    self.outdir, self.data[idx][0], i+1)
                if self.reverse:
                    dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
                    enc_len = int(self.data[idx][1]['output'][i]['shape'][0])
                else:
                    dec_len = int(self.data[idx][1]['output'][i]['shape'][0])
                    enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
                if len(att_w.shape) == 3:
                    att_w = att_w[:, :dec_len, :enc_len]
                else:
                    att_w = att_w[:dec_len, :enc_len]
                self._plot_and_save_attention(att_w, filename.format(trainer))

    def _plot_and_save_attention(self, att_w, filename):
        # dynamically import matplotlib due to not found error
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def add_results_to_json(js, nbest_hyps_sd, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps_sd: list of hypothesis for multi_speakers
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    num_spkrs = len(nbest_hyps_sd)
    new_js['output'] = []

    for ns in range(num_spkrs):
        tmp_out = []
        nbest_hyps = nbest_hyps_sd[ns]

        for n, hyp in enumerate(nbest_hyps, 1):
            # parse hypothesis
            rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

            # copy ground-truth
            out_dic = dict(js['output'][ns].items())

            # update name
            out_dic['name'] += '[%d]' % n

            # add recognition results
            out_dic['rec_text'] = rec_text
            out_dic['rec_token'] = rec_token
            out_dic['rec_tokenid'] = rec_tokenid
            out_dic['score'] = score

            # add to list of N-best result dicts
            tmp_out.append(out_dic)

            # show 1-best result
            if n == 1:
                logging.info('groundtruth: %s' % out_dic['text'])
                logging.info('prediction : %s' % out_dic['rec_text'])

        new_js['output'].append(tmp_out)
    return new_js
