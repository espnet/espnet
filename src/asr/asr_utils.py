#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import logging
import os

# chainer related
import chainer
from chainer import training
from chainer.training import extension
from chainer.backends import cuda

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
        olen = int(sorted_data[start][1]['output'][0]['shape'][0])
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


# TODO(watanabe) perform mean and variance normalization during the python program
# and remove the data dump process in run.sh
def converter_kaldi(batch, device=None):
    # batch only has one minibatch utterance, which is specified by batch[0]
    for data in batch:
        feat = kaldi_io_py.read_mat(data[1]['input'][0]['feat'])
        data[1]['feat'] = feat
    return batch


def delete_feat(batch):
    for data in batch:
        del data[1]['feat']

    return batch


# * -------------------- chainer extension related -------------------- *
class CompareValueTrigger(object):
    '''Trigger invoked when key value getting bigger or lower than before

    Args:
        key (str): Key of value.
        compare_fn: Function to compare the values.
        trigger: Trigger that decide the comparison interval

    '''

    def __init__(self, key, compare_fn, trigger=(1, 'epoch')):
        self._key = key
        self._best_value = None
        self._interval_trigger = training.util.get_trigger(trigger)
        self._init_summary()
        self._compare_fn = compare_fn

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None:
            # initialize best value
            self._best_value = value
            return False
        elif self._compare_fn(self._best_value, value):
            return True
        else:
            self._best_value = value
            return False

    def _init_summary(self):
        self._summary = chainer.reporter.DictSummary()


def restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    '''Extension to restore snapshot'''
    @training.make_extension(trigger=(1, 'epoch'))
    def restore_snapshot(trainer):
        _restore_snapshot(model, snapshot, load_fn)

    return restore_snapshot


def _restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    load_fn(snapshot, model)
    logging.info('restored from ' + str(snapshot))


def adadelta_eps_decay(eps_decay):
    '''Extension to perform adadelta eps decay'''
    @training.make_extension(trigger=(1, 'epoch'))
    def adadelta_eps_decay(trainer):
        _adadelta_eps_decay(trainer, eps_decay)

    return adadelta_eps_decay


def _adadelta_eps_decay(trainer, eps_decay):
    optimizer = trainer.updater.get_optimizer('main')
    # for chainer
    if hasattr(optimizer, 'eps'):
        current_eps = optimizer.eps
        setattr(optimizer, 'eps', current_eps * eps_decay)
        logging.info('adadelta eps decayed to ' + str(optimizer.eps))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p["eps"] *= eps_decay
            logging.info('adadelta eps decayed to ' + str(p["eps"]))


class PlotAttentionReport(extension.Extension):
    def __init__(self, model, data, outdir, converter=None, reverse=False):
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # TODO(kan-bayashi): clean up this process
        if hasattr(model, "module"):
            if hasattr(model.module, "predictor"):
                self.att_vis_fn = model.module.predictor.calculate_all_attentions
            else:
                self.att_vis_fn = model.module.calculate_all_attentions
        else:
            if hasattr(model, "predictor"):
                self.att_vis_fn = model.predictor.calculate_all_attentions
            else:
                self.att_vis_fn = model.calculate_all_attentions

    def __call__(self, trainer):
        if self.converter is not None:
            # TODO(kan-bayashi): need to be fixed due to hard coding
            x = self.converter([self.data], False)
        else:
            x = self.data
        if isinstance(x, tuple):
            att_ws = self.att_vis_fn(*x)
        elif isinstance(x, dict):
            att_ws = self.att_vis_fn(**x)
        else:
            att_ws = self.att_vis_fn(x)
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            if self.reverse:
                dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
                enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
            else:
                dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
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


# * -------------------- language model related -------------------- *
def load_labeldict(dict_file):
    labeldict = {'<blank>': 0}  # <blank>'s Id is 0
    for ln in open(dict_file, 'r').readlines():
        s, i = ln.split()
        labeldict[s] = int(i)
    if '<eos>' not in labeldict:
        labeldict['<eos>'] = len(labeldict)
    return labeldict
