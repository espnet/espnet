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
import torch
from torch.nn.parameter import Parameter
import math


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
    batch = batch[0]
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
    def __init__(self, model, data, outdir):
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if hasattr(model, "module"):
            self.att_vis_fn = model.module.predictor.calculate_all_attentions
        else:
            self.att_vis_fn = model.predictor.calculate_all_attentions

    def __call__(self, trainer):
        att_ws = self.att_vis_fn(self.data)
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            if len(att_w.shape) == 3:
                att_w = att_w[:, :int(self.data[idx][1]['output'][0]['shape'][0]),
                              :int(self.data[idx][1]['input'][0]['shape'][0])]
            else:
                att_w = att_w[:int(self.data[idx][1]['output'][0]['shape'][0]),
                              :int(self.data[idx][1]['input'][0]['shape'][0])]
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_parameters(model, elayers, *freeze_layer):
    size = 0
    count = 0
    for child in model.children():
        logging.info(str(type(child)))
        for name, module in child.named_children():
            logging.info(str(type(module)))
            if name not in freeze_layer and name is 'enc':
                for enc_name, enc_module in module.named_children():
                    for enc_layer_name, enc_layer_module in enc_module.named_children():
                        count += 1
                        if count <= elayers:
                            logging.info(str(enc_layer_name)+ " components is frozen")
                            for mname, param in module.named_parameters():
                                param.requires_grad = False
                                size += param.numel()
                        else:
                            logging.info(str(enc_layer_name)+ " components is not frozen")
            elif name not in freeze_layer and name is not 'enc':
                for mname, param in module.named_parameters():
                    logging.info(str(mname)+ " components is frozen")
                    logging.info(str(mname)+ " >> params after re-init")
                    param.requires_grad = False
                    size += param.numel()
            else:
                logging.info(str(name)+" components is not frozen" )
    return model, size

def sgd_lr_decay(lr_decay):
    '''Extension to perform sgd lr decay'''
    @training.make_extension(trigger=(1, 'epoch'))
    def sgd_lr_decay(trainer):
        _sgd_lr_decay(trainer, lr_decay)
    return sgd_lr_decay

def _sgd_lr_decay(trainer, lr_decay):
    optimizer = trainer.updater.get_optimizer('main')
    # for chainer
    if hasattr(optimizer, 'lr'):
        current_lr = optimizer.lr
        setattr(optimizer, 'lr', current_lr * lr_decay)
        logging.info('sgd lr decayed to ' + str(optimizer.lr))
    # pytorch
    else:
        for p in optimizer.param_groups:
            p['lr'] *= lr_decay
            logging.info('sgd lr decayed to ' + str(p["lr"]))

def init_parameter(weight=None, bias=None):
    if weight is not None:
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
    if bias is not None:
        stdv = 1. / math.sqrt(bias.size(0))
        bias.data.uniform_(-stdv, stdv)
    return weight, bias

def remove_output_layer(pretrained_model, odim, eprojs, dunits):
    w = Parameter(torch.Tensor(odim, eprojs).cuda())
    b = Parameter(torch.Tensor(odim).cuda())
    ew = Parameter(torch.Tensor(odim, dunits).cuda())
    eo = Parameter(torch.Tensor(odim, dunits).cuda())
    w, b = init_parameter(weight=w, bias=b)
    ew, _ = init_parameter(weight=ew)
    eo, _ = init_parameter(weight=eo)

    pretrained_model['predictor.ctc.ctc_lo.weight'] = w
    pretrained_model['predictor.dec.embed.weight'] = ew
    pretrained_model['predictor.ctc.ctc_lo.bias'] = b
    pretrained_model['predictor.dec.output.weight'] = eo
    pretrained_model['predictor.dec.output.bias'] = b
    return pretrained_model
