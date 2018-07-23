#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging

import numpy as np
import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc

from torch.autograd import Variable

# from e2e_asr_attctc_mod_th.model import linear_tensor
# from e2e_asr_attctc_mod_th.model import to_cuda
from e2e_asr_attctc_th import linear_tensor
from e2e_asr_attctc_th import to_cuda


class _ChainerLikeCTC(warp_ctc._CTC):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens):
        is_cuda = True if acts.is_cuda else False
        acts = acts.contiguous()
        loss_func = warp_ctc.gpu_ctc if is_cuda else warp_ctc.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(acts,
                  grads,
                  labels,
                  label_lens,
                  act_lens,
                  minibatch_size,
                  costs)
        # modified only here from original
        costs = torch.FloatTensor([costs.sum()]) / acts.size(1)
        ctx.grads = Variable(grads)
        ctx.grads /= ctx.grads.size(1)

        return costs


def chainer_like_ctc_loss(acts, labels, act_lens, label_lens):
    """Chainer like CTC Loss

    acts: Tensor of (seqLength x batch x outputDim) containing output from network
    labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    act_lens: Tensor of (batch) containing label length of each example
    """
    assert len(labels.size()) == 1  # labels must be 1 dimensional
    from torch.nn.modules.loss import _assert_no_grad
    _assert_no_grad(labels)
    _assert_no_grad(act_lens)
    _assert_no_grad(label_lens)
    return _ChainerLikeCTC.apply(acts, labels, act_lens, label_lens)


class CTC(torch.nn.Module):
    def __init__(self, odim, eprojs, dropout_rate):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.loss_fn = chainer_like_ctc_loss  # CTCLoss()

    def forward(self, hpad, ilens, ys):
        '''CTC forward

        :param hs:
        :param ys:
        :return:
        '''
        self.loss = None
        ilens = Variable(torch.from_numpy(np.fromiter(ilens, dtype=np.int32)))
        olens = Variable(torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32)))

        # zero padding for hs
        y_hat = linear_tensor(
            self.ctc_lo, F.dropout(hpad, p=self.dropout_rate))

        # zero padding for ys
        y_true = torch.cat(ys).cpu().int()  # batch x olen

        # get length info
        logging.info(self.__class__.__name__ +
                     ' input lengths:  ' + ''.join(str(ilens).split('\n')))
        logging.info(self.__class__.__name__ +
                     ' output lengths: ' + ''.join(str(olens).split('\n')))

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        y_hat = y_hat.transpose(0, 1)
        self.loss = to_cuda(self, self.loss_fn(y_hat, y_true, ilens, olens))
        logging.info('ctc loss:' + str(self.loss.data[0]))

        return self.loss

    def log_softmax(self, hpad):
        '''log_softmax of frame activations

        :param hs:
        :return:
        '''
        return F.log_softmax(linear_tensor(self.ctc_lo, hpad), dim=2)
