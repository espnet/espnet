# encoding: utf-8

import logging
import numpy as np

import chainer

import chainer.functions as F


class LabelSmoothingLoss(chainer.Chain):
    def __init__(self, smoothing, n_target_vocab, normalize_length=False, ignore_id=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.use_label_smoothing = False
        if smoothing > 0.0:
            logging.info("Use label smoothing")
            self.smoothing = smoothing
            self.confidence = 1. - smoothing
            self.use_label_smoothing = True
            self.n_target_vocab = n_target_vocab
        self.normalize_length = normalize_length
        self.ignore_id = ignore_id

    def forward(self, ys_block, ys_pad, eos):
        xp = self.xp
        eos = np.array([eos], 'i')
        ys_out = [np.concatenate([y, eos], axis=0) for y in ys_pad]
        ys_out = F.pad_sequence(ys_out, padding=-1)
        t_block = chainer.Variable(xp.array(ys_out.data))
        # Output (all together at once for efficiency)
        batch, length, dims = ys_block.shape
        concat_logit_block = ys_block.reshape(-1, dims)
        rebatch, _ = concat_logit_block.shape
        # Make target
        concat_t_block = t_block.reshape((rebatch)).data
        ignore_mask = (concat_t_block >= 0)
        n_token = ignore_mask.sum()
        normalizer = n_token if self.normalize_length else batch
        if not self.use_label_smoothing:
            loss = F.softmax_cross_entropy(concat_logit_block, concat_t_block)
            loss = loss * n_token / normalizer
        else:
            log_prob = F.log_softmax(concat_logit_block)
            broad_ignore_mask = self.xp.broadcast_to(
                ignore_mask[:, None],
                concat_logit_block.shape)
            pre_loss = ignore_mask * \
                log_prob[self.xp.arange(rebatch), concat_t_block]
            loss = - F.sum(pre_loss) / normalizer
            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = self.confidence * loss + self.smoothing * label_smoothing

        acc = F.accuracy(concat_logit_block, concat_t_block, ignore_label=self.ignore_id)
        return loss, acc
