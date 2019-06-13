#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import itertools
import json
import logging
import math
# matplotlib related
import os
import shutil
import tempfile

# chainer related
import chainer

from chainer import training
from chainer.training import extension

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer

# io related
import kaldi_io_py
import matplotlib
import numpy as np
import torch
matplotlib.use('Agg')


# * -------------------- training iterator related -------------------- *
def make_batchset_asr(data, batch_size, max_length_in, max_length_out,
                      num_batches=0, min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as "make_batchset_within_category"

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get('category'), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for _, d in category2data.items():
        # batch: List[List[Tuple[str, dict]]]
        batches = make_batchset_within_category(
            data=d,
            batch_size=batch_size,
            max_length_in=max_length_in,
            max_length_out=max_length_out,
            min_batch_size=min_batch_size,
            shortest_first=shortest_first)
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info('# minibatches: ' + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as "make_batchset_within_category"

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get('category'), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for _, d in category2data.items():
        # batch: List[List[Tuple[str, dict]]]
        batches = make_batchset_within_category(
            data=d,
            batch_size=batch_size,
            max_length_in=max_length_in,
            max_length_out=max_length_out,
            min_batch_size=min_batch_size,
            shortest_first=shortest_first)
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info('# minibatches: ' + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches


def make_batchset_within_category(
        data, batch_size, max_length_in, max_length_out,
        min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    # sort it by input lengths (long to short)
    try:
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=not shortest_first)
        logging.info('# utts: ' + str(len(sorted_data)))
    except Exception:
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['output'][0]['shape'][0]), reverse=not shortest_first)
    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        if 'input' in sorted_data[0][1].keys() and 'output' in sorted_data[0][1].keys():
            ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
            olen = int(sorted_data[start][1]['output'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        if 'output' not in sorted_data[0][1].keys():
            ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
            factor = int(ilen / max_length_in)
        if 'input' not in sorted_data[0][1].keys():
            olen = int(sorted_data[start][1]['output'][0]['shape'][0])
            factor = int(olen / max_length_out)

        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i]
                                    for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # batch: List[List[Tuple[str, dict]]]
    return minibatches


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


class CompareValueTrigger(object):
    """Trigger invoked when key value getting bigger or lower than before

    :param str key : Key of value
    :param function compare_fn : Function to compare the values
    :param (int, str) trigger : Trigger that decide the comparison interval
    """

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


class PlotAttentionReport(extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param CustomConverter converter: function to convert data
    :param int | torch.device device: device
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, transform, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.transform = transform
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            att_w = self.get_attention_weight(idx, att_w)
            plot = self.draw_attention_plot(att_w)
            logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()

    def get_attention_weights(self):
        batch = self.converter([self.transform(self.data)], self.device)
        att_ws = self.att_vis_fn(*batch)
        return att_ws

    def get_attention_weight(self, idx, att_w):
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
        return att_w

    def draw_attention_plot(self, att_w):
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
        return plt

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


def restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    """Extension to restore snapshot"""

    @training.make_extension(trigger=(1, 'epoch'))
    def restore_snapshot(trainer):
        _restore_snapshot(model, snapshot, load_fn)

    return restore_snapshot


def _restore_snapshot(model, snapshot, load_fn=chainer.serializers.load_npz):
    load_fn(snapshot, model)
    logging.info('restored from ' + str(snapshot))


def adadelta_eps_decay(eps_decay):
    """Extension to perform adadelta eps decay"""

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


def torch_snapshot(savefun=torch.save,
                   filename='snapshot.ep.{.updater.epoch}'):
    """Returns a trainer extension to take snapshots of the trainer for pytorch."""

    @extension.make_extension(trigger=(1, 'epoch'), priority=-100)
    def torch_snapshot(trainer):
        _torch_snapshot_object(trainer, trainer, filename.format(trainer), savefun)

    return torch_snapshot


def _torch_snapshot_object(trainer, target, filename, savefun):
    # make snapshot_dict dictionary
    s = DictionarySerializer()
    s.save(trainer)
    if hasattr(trainer.updater.model, "model"):
        # (for TTS)
        if hasattr(trainer.updater.model.model, "module"):
            model_state_dict = trainer.updater.model.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.model.state_dict()
    else:
        # (for ASR)
        if hasattr(trainer.updater.model, "module"):
            model_state_dict = trainer.updater.model.module.state_dict()
        else:
            model_state_dict = trainer.updater.model.state_dict()
    snapshot_dict = {
        "trainer": s.target,
        "model": model_state_dict,
        "optimizer": trainer.updater.get_optimizer('main').state_dict()
    }

    # save snapshot dictionary
    fn = filename.format(trainer)
    prefix = 'tmp' + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=trainer.out)
    tmppath = os.path.join(tmpdir, fn)
    try:
        savefun(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(trainer.out, fn))
    finally:
        shutil.rmtree(tmpdir)


# * -------------------- general -------------------- *
class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def __getitem__(self, name):
        return self.obj[name]

    def __len__(self):
        return len(self.obj)

    def fields(self):
        return self.obj

    def items(self):
        return self.obj.items()

    def keys(self):
        return self.obj.keys()


def get_model_conf(model_path, conf_path=None):
    """Get model config information by reading a model config file (model.json)

    :param str model_path: model path
    :param str conf_path: optional model config path
    """

    if conf_path is None:
        model_conf = os.path.dirname(model_path) + '/model.json'
    else:
        model_conf = conf_path
    with open(model_conf, "rb") as f:
        logging.info('reading a config file from ' + model_conf)
        return json.load(f, object_hook=AttributeDict)


def chainer_load(path, model):
    """Function to load chainer model parameters

    :param str path: model file or snapshot file to be loaded
    :param chainer.Chain model: chainer model
    """
    if 'snapshot' in path:
        chainer.serializers.load_npz(path, model, path='updater/model:main/')
    else:
        chainer.serializers.load_npz(path, model)


def torch_save(path, model):
    """Function to save torch model states

    :param str path: file path to be saved
    :param torch.nn.Module model: torch model
    """
    if hasattr(model, 'module'):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def torch_load(path, model):
    """Function to load torch model states

    :param str path: model file or snapshot file to be loaded
    :param torch.nn.Module model: torch model
    """
    if 'snapshot' in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)['model']
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


def torch_resume(snapshot_path, trainer):
    """Function to resume from snapshot for pytorch

    :param str snapshot_path: snapshot file path
    :param instance trainer: chainer trainer instance
    """
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore trainer states
    d = NpzDeserializer(snapshot_dict['trainer'])
    d.load(trainer)

    # restore model states
    if hasattr(trainer.updater.model, "model"):
        # (for TTS model)
        if hasattr(trainer.updater.model.model, "module"):
            trainer.updater.model.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.model.load_state_dict(snapshot_dict['model'])
    else:
        # (for ASR model)
        if hasattr(trainer.updater.model, "module"):
            trainer.updater.model.module.load_state_dict(snapshot_dict['model'])
        else:
            trainer.updater.model.load_state_dict(snapshot_dict['model'])

    # retore optimizer states
    trainer.updater.get_optimizer('main').load_state_dict(snapshot_dict['optimizer'])

    # delete opened snapshot
    del snapshot_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text string
    :return: recognition token string
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            if 'text' in out_dic.keys():
                logging.info('groundtruth: %s' % out_dic['text'])
            logging.info('prediction : %s' % out_dic['rec_text'])

    return new_js


def plot_spectrogram(plt, spec, mode='db', fs=None, frame_shift=None,
                     bottom=True, left=True, right=True, top=False,
                     labelbottom=True, labelleft=True, labelright=True,
                     labeltop=False, cmap='inferno'):
    """Plot spectrogram using matplotlib

    :param matplotlib.pyplot plt:
    :param np.ndarray spec: Input stft (Freq, Time)
    :param str mode: db or linear.
    :param int fs: Sample frequency. To convert y-axis to kHz unit.
    :param int frame_shift: The frame shift of stft. To convert x-axis to second unit.
    :param bool bottom:
    :param bool left:
    :param bool right:
    :param bool top:
    :param bool labelbottom:
    :param bool labelleft:
    :param bool labelright:
    :param bool labeltop:
    :param str cmap: colormap defined in matplotlib

    """
    spec = np.abs(spec)
    if mode == 'db':
        x = 20 * np.log10(spec + np.finfo(spec.dtype).eps)
    elif mode == 'linear':
        x = spec
    else:
        raise ValueError(mode)

    if fs is not None:
        ytop = fs / 2000
        ylabel = 'kHz'
    else:
        ytop = x.shape[0]
        ylabel = 'bin'

    if frame_shift is not None and fs is not None:
        xtop = x.shape[1] * frame_shift / fs
        xlabel = 's'
    else:
        xtop = x.shape[1]
        xlabel = 'frame'

    extent = (0, xtop, 0, ytop)
    plt.imshow(x[::-1], cmap=cmap, extent=extent)

    if labelbottom:
        plt.xlabel('time [{}]'.format(xlabel))
    if labelleft:
        plt.ylabel('freq [{}]'.format(ylabel))
    plt.colorbar().set_label('{}'.format(mode))

    plt.tick_params(bottom=bottom, left=left, right=right, top=top,
                    labelbottom=labelbottom, labelleft=labelleft,
                    labelright=labelright, labeltop=labeltop)
    plt.axis('auto')
