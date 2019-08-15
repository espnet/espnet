# encoding: utf-8
"""Class Declaration of Transformer's Label Smootion loss."""

import logging

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
        self.acc = None

    def forward(self, ys_block, ys_pad):
        # Output (all together at once for efficiency)
        batch, length, dims = ys_block.shape
        concat_logit_block = ys_block.reshape(-1, dims)

        # Target reshape
        concat_t_block = ys_pad.reshape((batch * length))
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
                log_prob[self.xp.arange(batch * length), concat_t_block]
            loss = - F.sum(pre_loss) / normalizer
            label_smoothing = broad_ignore_mask * \
                - 1. / self.n_target_vocab * log_prob
            label_smoothing = F.sum(label_smoothing) / normalizer
            loss = self.confidence * loss + self.smoothing * label_smoothing
        return loss
