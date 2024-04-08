#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
# Copyright 2023 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Singing Tacotron decoder related modules."""

import six
import torch

from espnet.nets.pytorch_backend.rnn.attentions import AttForwardTA
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet, Prenet, ZoneOutCell


def decoder_init(m):
    """Initialize decoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class Decoder(torch.nn.Module):
    """Decoder module of Spectrogram prediction network.

    This is a module of decoder of Spectrogram prediction network in Singing Tacotron,
    which described in `https://arxiv.org/pdf/2202.07907v1.pdf`_.
    The decoder generates the sequence of
    features from the sequence of the hidden states.

    .. _`Singing-Tacotron: Global Duration Control Attention and Dynamic
    Filter for End-to-end Singing Voice Synthesis`:
       https://arxiv.org/pdf/2202.07907v1.pdf

    """

    def __init__(
        self,
        idim,
        odim,
        att,
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        output_activation_fn=None,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=1,
    ):
        """Initialize Singing Tacotron decoder module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            att (torch.nn.Module): Instance of attention class.
            dlayers (int, optional): The number of decoder lstm layers.
            dunits (int, optional): The number of decoder lstm units.
            prenet_layers (int, optional): The number of prenet layers.
            prenet_units (int, optional): The number of prenet units.
            postnet_layers (int, optional): The number of postnet layers.
            postnet_filts (int, optional): The number of postnet filter size.
            postnet_chans (int, optional): The number of postnet filter channels.
            output_activation_fn (torch.nn.Module, optional):
                Activation function for outputs.
            cumulate_att_w (bool, optional):
                Whether to cumulate previous attention weight.
            use_batch_norm (bool, optional): Whether to use batch normalization.
            use_concate (bool, optional): Whether to concatenate encoder embedding
                with decoder lstm outputs.
            dropout_rate (float, optional): Dropout rate.
            zoneout_rate (float, optional): Zoneout rate.
            reduction_factor (int, optional): Reduction factor.

        """
        super(Decoder, self).__init__()

        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_concate = use_concate
        self.reduction_factor = reduction_factor

        # check attention type
        if isinstance(self.att, AttForwardTA):
            self.use_att_extra_inputs = True
        else:
            self.use_att_extra_inputs = False

        # define lstm network
        prenet_units = prenet_units if prenet_layers != 0 else odim
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(dlayers):
            iunits = idim + prenet_units if layer == 0 else dunits
            lstm = torch.nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm += [lstm]

        # define prenet
        if prenet_layers > 0:
            self.prenet = Prenet(
                idim=odim,
                n_layers=prenet_layers,
                n_units=prenet_units,
                dropout_rate=dropout_rate,
            )
        else:
            self.prenet = None

        # define postnet
        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )
        else:
            self.postnet = None

        # define projection layers
        iunits = idim + dunits if use_concate else dunits
        self.feat_out = torch.nn.Linear(iunits, odim * reduction_factor, bias=False)
        self.prob_out = torch.nn.Linear(iunits, reduction_factor)

        # initialize
        self.apply(decoder_init)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, hs, hlens, trans_token, ys):
        """Calculate forward propagation.

        Args:
            hs (Tensor): Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            trans_token (Tensor): Global transition token for duration (B x Tmax x 1)
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: Batch of output tensors before postnet (B, Lmax, odim).
            Tensor: Batch of logits of stop prediction (B, Lmax).
            Tensor: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        outs, logits, att_ws = [], [], []
        for y in ys.transpose(0, 1):
            if trans_token is None:
                if self.use_att_extra_inputs:
                    att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w, prev_out)
                else:
                    att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            else:  # GDCA
                att_c, att_w = self.att(hs, hlens, trans_token, z_list[0], prev_att_w)
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(hs.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            att_ws += [att_w]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        if self.reduction_factor > 1:
            before_outs = before_outs.view(
                before_outs.size(0), self.odim, -1
            )  # (B, odim, Lmax)

        if self.postnet is not None:
            after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        else:
            after_outs = before_outs
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        logits = logits

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits, att_ws

    def inference(
        self,
        h,
        trans_token,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=30.0,
        use_att_constraint=False,
        use_dynamic_filter=True,
        backward_window=1,
        forward_window=3,
    ):
        """Generate the sequence of features given the sequences of characters.

        Args:
            h (Tensor): Input sequence of encoder hidden states (T, C).
            trans_token (Tensor): Global transition token for duration.
            threshold (float, optional): Threshold to stop generation.
            minlenratio (float, optional): Minimum length ratio.
                If set to 1.0 and the length of input is 10,
                the minimum length of outputs will be 10 * 1 = 10.
            minlenratio (float, optional): Minimum length ratio.
                If set to 10 and the length of input is 10,
                the maximum length of outputs will be 10 * 10 = 100.
            use_att_constraint (bool):
                Whether to apply attention constraint introduced in `Deep Voice 3`_.
            use_dynamic_filter (bool):
                Whether to apply dynamic filter introduced in `Singing Tacotron`_.
            backward_window (int): Backward window size in attention constraint.
            forward_window (int): Forward window size in attention constraint.
        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).
        Note:
            This computation is performed in auto-regressive manner.
        .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654
        """
        # setup
        assert len(h.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # setup for attention constraint
        if use_att_constraint or use_dynamic_filter:
            last_attended_idx = 0
        else:
            last_attended_idx = None

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []

        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs,
                    ilens,
                    z_list[0],
                    prev_att_w,
                    prev_out,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window,
                )
            else:
                if trans_token is None:
                    att_c, att_w = self.att(
                        hs,
                        ilens,
                        z_list[0],
                        prev_att_w,
                        last_attended_idx=last_attended_idx,
                        backward_window=backward_window,
                        forward_window=forward_window,
                    )
                else:  # GDCA
                    att_c, att_w = self.att(
                        hs,
                        ilens,
                        trans_token,
                        z_list[0],
                        prev_att_w,
                        last_attended_idx=last_attended_idx,
                        backward_window=backward_window,
                        forward_window=forward_window,
                    )

            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
            probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            if use_att_constraint or use_dynamic_filter:
                last_attended_idx = int(att_w.argmax())

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws
