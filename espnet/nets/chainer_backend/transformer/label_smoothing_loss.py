# encoding: utf-8
"""Class Declaration of Transformer's Label Smootion loss."""

import logging

import chainer

import chainer.functions as F


class LabelSmoothingLoss(chainer.Chain):
    """Label Smoothing Loss.

    Args:
        smoothing (float): smoothing rate (0.0 means the conventional CE).
        n_target_vocab (int): number of classes.
        normalize_length (bool): normalize loss by sequence length if True.

    """

    def __init__(self, smoothing, n_target_vocab, normalize_length=False):
        """Initialize Loss.

        Args:
            smoothing (float): smoothing rate (0.0 means the conventional CE).
            n_target_vocab (int): number of classes.
            normalize_length (bool): normalize loss by sequence length if True.

        """
        super(LabelSmoothingLoss, self).__init__()
        self.use_label_smoothing = False
        if smoothing > 0.0:
            logging.info("Use label smoothing")
            self.smoothing = smoothing
            self.confidence = 1. - smoothing
            self.use_label_smoothing = True
            self.n_target_vocab = n_target_vocab
        self.normalize_length = normalize_length

    def forward(self, concat_logit_block, t_block, batch, length):
        """E2E forward propagation.

        Args:
            concat_logit_block (chainer.Variable): Predicted labels.
            t_block (chainer.Variable): Target (true) labels.
            batch (int): mini-batch size.

        Returns:
            float: Training loss.

        """
        # Output (all together at once for efficiency)
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

        return loss
