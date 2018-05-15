#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging

# chainer related
import chainer
from chainer import training
import numpy as np


# * -------------------- agumenting data prep -------------------- *
def make_augment_batchset(data, batch_size,
                          max_length_in, max_length_out,
                          num_batches=0, subsample=1):
    # sort it by input lengths (long to short)
    # data has keys ifilename, ofilename, sentences
    # sentences has keys id, and values ilen, olen, ioffset, ooffset
    # ifilename = data['ifilename']
    # ofilename = data['ofilename']
    # idict = data['idict']
    # odict = data['odict']
    meta = {'ifilename': data['ifilename'],
            'ofilename': data['ofilename'],
            'idict': data['idict'],
            'odict': data['odict']}
    assert '<unk>' in data['odict']
    sentences = data['sentences']
    sorted_data = sorted(sentences.items(), key=lambda data: int(
        data[1]['ilen']), reverse=True)
    logging.info('# augmenting data: ' + str(len(sorted_data)))
    if subsample == 1:
        len_fac = 1
    else:
        print("Subsample: ", subsample)
        len_fac = np.prod([int(i) for i in subsample.split('_')])
    # change batchsize depending on the input and output length
    minibatches = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['ilen'])
        olen = int(sorted_data[start][1]['olen'])
        ilen *= len_fac
        factor = max(int(ilen / max_length_in), int(olen / max_length_out))
        b = max(1, int(batch_size / (1 + factor)))
        end = start + b  # (start + b) if (start + b < len(sorted_data)) else len(sorted_data)
        minibatches.append(sorted_data[start:end])
        if end >= len(sorted_data):
            break
        start = end
    if num_batches > 0:
        minibatches = minibatches[:num_batches]
    logging.info('# augmenting data minibatches: ' + str(len(minibatches)))
    return minibatches, meta


def converter_augment(batch, idict, odict, ifile, ofile):
    for b_idx, b_obj in batch:
        ifile.seek(b_obj['ioffset'])
        ofile.seek(b_obj['ooffset'])
        iline = ifile.readline()
        oline = ofile.readline()
        iline = ['<s>'] + iline.strip().split() + ['</s>']  # BOS and EOS for output handled by decoder
        print("ILINE: ", iline)
        print("--------DICT---------")
        print(idict)
        iline = [idict[i] for i in iline]
        assert len(iline) > 2
        iline = np.array(iline, dtype=np.int64)
        oline = oline.strip().split()[1:]  # so that we can use the same aug files from OpenNMT, removed "aug"
        assert len(oline) > 0
        oline = ' '.join([str(odict.get(i, odict['<unk>'])) for i in oline])
        b_obj['feat'] = iline
        b_obj['tokenid'] = oline
    return batch


# * -------------------- training iterator related -------------------- *
def make_batchset(data, batch_size, max_length_in, max_length_out, num_batches=0):
    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['ilen']), reverse=True)
    logging.info('# utts: ' + str(len(sorted_data)))
    # change batchsize depending on the input and output length
    minibatch = []
    start = 0
    while True:
        ilen = int(sorted_data[start][1]['ilen'])
        olen = int(sorted_data[start][1]['olen'])
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
def converter_kaldi(batch, reader):
    for data in batch:
        feat = reader[data[0].encode('ascii', 'ignore')]
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
