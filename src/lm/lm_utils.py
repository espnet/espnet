#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import chainer
import logging
import random
import numpy as np
import os

from chainer.training import extension


# read tokens as a sequence of sentences
def read_tokens(filename, label_dict):

    data = []
    for ln in open(filename, 'rb').readlines():
        data.append(np.array([label_dict[label]
                              if label in label_dict else label_dict['<unk>']
                              for label in ln.decode('utf-8').split()], dtype=np.int32))
    return data


# count tokens and oovs
def count_tokens(data, unk=None):

    n_tokens = 0
    n_oovs = 0
    for sentence in data:
        n_tokens += len(sentence)
        if unk is not None:
            n_oovs += np.count_nonzero(sentence == unk)
    return n_tokens, n_oovs


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


# Dataset iterator to create a batch of sentences.
# This iterator returns a pair of sentences, where one token is shifted
# between the sentences like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
# Sentence batches are made in order of longer sentences, and then
# randomly shuffled.
class ParallelSentenceIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, max_length=0, sos=0, eos=0, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.batch_indices = []
        # make mini-batches
        if batch_size > 1:
            indices = sorted(range(len(dataset)), key=lambda i: -len(dataset[i]))
            bs = 0
            while bs < length:
                be = min(bs + batch_size, length)
                # batch size is automatically reduced if the sentence length
                # is larger than max_length
                if max_length > 0:
                    sent_length = len(dataset[indices[bs]])
                    be = min(be, bs + max(batch_size // (sent_length // max_length + 1), 1))
                self.batch_indices.append(np.array(indices[bs:be]))
                bs = be
            # shuffle batches
            random.shuffle(self.batch_indices)
        else:
            self.batch_indices = [np.array([i]) for i in six.moves.range(length)]

        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        self.sos = sos
        self.eos = eos
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a sentence pair like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
        # represented by token IDs.
        n_batches = len(self.batch_indices)
        if not self.repeat and self.iteration >= n_batches:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration

        batch = []
        for idx in self.batch_indices[self.iteration % n_batches]:
            batch.append((np.append([self.sos], self.dataset[idx]),
                          np.append(self.dataset[idx], [self.eos])))

        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1

        epoch = self.iteration // n_batches
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return batch

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration / len(self.batch_indices)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - 1) / len(self.batch_indices)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


# Extension that makes a symbolic link to the best model
class MakeSymlinkToBestModel(extension.Extension):

    def __init__(self, key, prefix='model', suffix='best'):
        super(MakeSymlinkToBestModel, self).__init__()
        self.best_model = -1
        self.min_loss = 0.0
        self.key = key
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, trainer):
        observation = trainer.observation
        if self.key in observation:
            loss = observation[self.key]
            if self.best_model == -1 or loss < self.min_loss:
                self.min_loss = loss
                self.best_model = trainer.updater.epoch
                src = '%s.%d' % (self.prefix, self.best_model)
                dest = os.path.join(trainer.out, '%s.%s' % (self.prefix, self.suffix))
                if os.path.lexists(dest):
                    os.remove(dest)
                os.symlink(src, dest)
                logging.info('best model is ' + src)

    def serialize(self, serializer):
        if isinstance(serializer, chainer.serializer.Serializer):
            serializer('_best_model', self.best_model)
            serializer('_min_loss', self.min_loss)
            serializer('_key', self.key)
            serializer('_prefix', self.prefix)
            serializer('_suffix', self.suffix)
        else:
            self.best_model = serializer('_best_model', -1)
            self.min_loss = serializer('_min_loss', 0.0)
            self.key = serializer('_key', '')
            self.prefix = serializer('_prefix', 'model')
            self.suffix = serializer('_suffix', 'best')
