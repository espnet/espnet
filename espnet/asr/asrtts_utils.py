#!/usr/bin/env python

# Copyright 2017 Brno University (Karthick Baskar)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import math
# matplotlib related

# chainer related
from chainer import training

# io related
import kaldi_io_py
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *
def merge_batchsets(train_subsets, shuffle=True):
    ndata = len(train_subsets)
    train = []
    for num in range(ndata):
        train.extend(train_subsets[num])
    if shuffle:
        import random
        random.shuffle(train)
    return train


def load_inputs_spk_and_targets(batch, use_speaker_embedding=False):
    """Function to load inputs, speaker and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :param bool use_speaker_embedding: whether to load speaker embedding vector
    :return: list of input feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
    :rtype: list of int ndarray
    """
    # load acoustic features and target sequence of token ids
    if 'input' in batch[0][1].keys() and 'output' in batch[0][1].keys():
        xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
        ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        if len(nonzero_sorted_idx) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        ys = [np.fromiter(map(int, ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]
        # load speaker embedding
        if use_speaker_embedding and len(batch[0][1]['input']) > 1:
            spembs = [kaldi_io_py.read_vec_flt(b[1]['input'][1]['feat']) for b in batch]
            spembs = [spembs[i] for i in nonzero_sorted_idx]
            return xs, ys, spembs
        else:
            return xs, ys

    if 'input' not in batch[0][1].keys():
        ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(ys)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(ys[i]))
        if len(nonzero_sorted_idx) != len(ys):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(ys), len(nonzero_sorted_idx)))

        # remove zero-length samples
        ys = [np.fromiter(map(int, ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]
        # load speaker embedding
        if use_speaker_embedding:
            spembs = [kaldi_io_py.read_vec_flt(b[1]['output'][1]['feat']) for b in batch]
            spembs = [spembs[i] for i in nonzero_sorted_idx]
        else:
            spembs = None
        return ys, ys, spembs

    if 'output' not in batch[0][1].keys():
        xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        if len(nonzero_sorted_idx) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        # load speaker embedding
        if use_speaker_embedding:
            spembs = [kaldi_io_py.read_vec_flt(b[1]['input'][1]['feat']) for b in batch]
            spembs = [spembs[i] for i in nonzero_sorted_idx]
        else:
            spembs = None
        return xs, xs, spembs


def load_inputs_and_targets(batch):
    """Function to load inputs and targets from list of dicts

    :param list batch: list of dict which is subset of loaded data.json
    :param bool use_speaker_embedding: whether to load speaker embedding vector
    :return: list of input feature sequences [(T_1, D), (T_2, D), ..., (T_B, D)]
    :rtype: list of float ndarray
    :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
    :rtype: list of int ndarray
    """
    # load acoustic features and target sequence of token ids
    if 'input' in batch[0][1].keys() and 'output' in batch[0][1].keys():
        xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]
        ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        if len(nonzero_sorted_idx) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        ys = [np.fromiter(map(int, ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]
        return xs, ys

    if 'input' not in batch[0][1].keys():
        ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(ys)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(ys[i]))
        if len(nonzero_sorted_idx) != len(ys):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(ys), len(nonzero_sorted_idx)))

        # remove zero-length samples
        ys = [np.fromiter(map(int, ys[i]), dtype=np.int64) for i in nonzero_sorted_idx]
        return ys, ys

    if 'output' not in batch[0][1].keys():
        xs = [kaldi_io_py.read_mat(b[1]['input'][0]['feat']) for b in batch]

        # get index of non-zero length samples
        nonzero_idx = filter(lambda i: len(xs[i]) > 0, range(len(xs)))
        # sort in input lengths
        nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
        if len(nonzero_sorted_idx) != len(xs):
            logging.warning('Target sequences include empty tokenid (batch %d -> %d).' % (
                len(xs), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [xs[i] for i in nonzero_sorted_idx]
        return xs, xs


def remove_output_layer(pretrained_model, odim, eprojs, dunits, model_type):
    stdv_bias = 1. / math.sqrt(odim)
    stdv_weight = 1. / math.sqrt(dunits)
    stdv_cweight = 1. / math.sqrt(eprojs)
    wt = torch.nn.init.uniform_(torch.FloatTensor(odim, eprojs).cuda(),
                                -stdv_cweight, stdv_cweight)
    ewt = torch.nn.init.uniform_(torch.FloatTensor(odim, dunits).cuda(),
                                 -stdv_weight, stdv_weight)
    eot = torch.nn.init.uniform_(torch.FloatTensor(odim, dunits).cuda(),
                                 -stdv_weight, stdv_weight)
    bs = torch.nn.init.uniform_(torch.FloatTensor(odim).cuda(),
                                -stdv_bias, stdv_bias)
    if model_type == 'asr':
        pretrained_model['ctc.ctc_lo.weight'] = wt
        pretrained_model['ctc.ctc_lo.bias'] = bs
        pretrained_model['dec.embed.weight'] = ewt
        pretrained_model['dec.output.weight'] = eot
        pretrained_model['dec.output.bias'] = bs
    elif model_type == 'tts':
        pretrained_model['enc.embed.weight'] = ewt
    return pretrained_model


def freeze_parameters(model, elayers, *freeze_layer):
    size = 0
    count = 0
    for child in model.children():
        logging.info(str(type(child)))
        for name, module in child.named_children():
            logging.info(str(type(module)))
            if name not in freeze_layer and name == 'enc':
                for enc_name, enc_module in module.named_children():
                    for enc_layer_name, enc_layer_module in enc_module.named_children():
                        count += 1
                        if count <= elayers:
                            logging.info(str(enc_layer_name) + " components is frozen")
                            for mname, param in module.named_parameters():
                                param.requires_grad = False
                                size += param.numel()
                        else:
                            logging.info(str(enc_layer_name) + " components is not frozen")
            elif name not in freeze_layer and name != 'enc':
                for mname, param in module.named_parameters():
                    logging.info(str(mname) + " components is frozen")
                    logging.info(str(mname) + " >> params after re-init")
                    param.requires_grad = False
                    size += param.numel()
            else:
                logging.info(str(name) + " components is not frozen")
    return model, size


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations

    Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


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
