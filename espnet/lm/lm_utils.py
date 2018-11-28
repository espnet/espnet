#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import chainer
import logging
import numpy as np
import os
import random
import six

from chainer.training import extension


def read_tokens(filename, label_dict):
    """
    Read tokens as a sequence of sentences

    :param str filename : The name of the input file
    :param dict label_dict : dictionary that maps token label string to its ID number
    :return list of ID sequences
    :rtype list
    """

    data = []
    for ln in open(filename, 'rb').readlines():
        data.append(np.array([label_dict[label]
                              if label in label_dict else label_dict['<unk>']
                              for label in ln.decode('utf-8').split()], dtype=np.int32))
    return data


def count_tokens(data, unk_id=None):
    """
    Count tokens and oovs in token ID sequences

    :param list[np.ndarray] data: list of token ID sequences
    :param int unk_id: ID of unknown token '<unk>'
    :return number of token occurrences, number of oov tokens
    :rtype int, int
    """

    n_tokens = 0
    n_oovs = 0
    for sentence in data:
        n_tokens += len(sentence)
        if unk_id is not None:
            n_oovs += np.count_nonzero(sentence == unk_id)
    return n_tokens, n_oovs


def compute_perplexity(result):
    """
    Computes and add the perplexity to the LogReport
    :param dict result: The current observations
    """
    # Routine to rewrite the result dictionary of LogReport to add perplexity values
    result['perplexity'] = np.exp(result['main/loss'] / result['main/count'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


class ParallelSentenceIterator(chainer.dataset.Iterator):
    """
    Dataset iterator to create a batch of sentences.

       This iterator returns a pair of sentences, where one token is shifted
       between the sentences like '<sos> w1 w2 w3' and 'w1 w2 w3 <eos>'
       Sentence batches are made in order of longer sentences, and then
       randomly shuffled.
    """

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
            self._previous_epoch_detail = self.epoch + (self.current_position - 1) / len(self.batch_indices)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


class MakeSymlinkToBestModel(extension.Extension):
    """
    Extension that makes a symbolic link to the best model

    :param str key: Key of value
    :param str prefix: Prefix of model files and link target
    :param str suffix: Suffix of link target
    """

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


# TODO(Hori): currently it only works with character-word level LM.
#             need to consider any types of subwords-to-word mapping.
def make_lexical_tree(word_dict, subword_dict, word_unk):
    """
    Make a lexical tree to compute word-level probabilities

    """
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root
