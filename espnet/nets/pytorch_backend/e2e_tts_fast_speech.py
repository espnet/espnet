#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.tts_interface import TTSInterface


class DurationPredictorLoss(torch.nn.Module):
    """Duration predictor loss module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """
    def __init__(self):
        super(DurationPredictorLoss, self).__init__()

    def forward(self):
        pass


class FeedForwardTransformer(TTSInterface, torch.nn.Module):
    """Feed Forward Transformer for TTS

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)
    """

    def __init__(self):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

    def forward(self):
        pass


class DurationCalculator(torch.nn.Module):
    """Duration calculator using teacher model

    :param e2e_tts_transformer.Transformer teacher_model: teacher auto-regressive Transformer
    """

    def __init__(self, teacher_model):
        super(DurationCalculator, self).__init__()
        if not isinstance(teacher_model, Transformer):
            raise ValueError("teacher model should be the instance of e2e_tts_transformer.Transformer")
        self.teacher_model = teacher_model
        self.diag_head_info = None

    def forward(self, xs, ilens, ys, olens):
        """Calculate duration of each inputs

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor ilens: list of lengths of each output batch (B)
        :return torch.Tensor: batch of durations (B, Tmax)
        """
        att_ws_dict = self.calculate_all_attentions(xs, ilens, ys, olens)
        if self.diag_head_info is None:
            self._decide_diogonal_head(att_ws_dict)
        att_ws = att_ws_dict[self.diag_head_info[0]][:, self.diag_head_info[1]]  # (B, Lmax, Tmax)
        durations = torch.stack(
            [att_ws.argmax(dim=-1).eq(i).sum(dim=-1) for i in range(max(ilens))], dim=-1)  # (B, Tmax)
        durations.masked_fill_(make_pad_mask(ilens).to(durations.device), 0)

        return durations

    def _calculate_att_ws_dict(self, xs, ilens, ys, olens):
        with torch.no_grad():
            x_masks = self.teacher_model._source_mask(ilens)
            hs, _ = self.teacher_model.encoder(xs, x_masks)
            ys_in = self.teacher_model._add_first_frame_and_remove_last_frame(ys)
            if self.teacher_model.reduction_factor > 1:
                ys_in = ys_in[:, self.teacher_model.reduction_factor - 1::self.teacher_model.reduction_factor]
                olens_in = olens.new([olen // self.teacher_model.reduction_factor for olen in olens])
            else:
                olens_in = olens
            y_masks = self.teacher_model._target_mask(olens_in)
            xy_masks = self.teacher_model._source_to_target_mask(ilens, olens_in)
            self.teacher_model.decoder(ys_in, y_masks, hs, xy_masks)
        att_ws_dict = {}
        for name, m in self.teacher_model.named_modules():
            if isinstance(m, MultiHeadedAttention):
                att_ws_dict[name] = m.attn

        return att_ws_dict

    def _decide_diogonal_head(self, att_ws_dict):
        best_diagonal_score = 0.0
        for key in att_ws_dict.keys():
            att_ws = att_ws_dict[key]
            diagonal_scores = att_ws.max(dim=-1).mean(dim=-1).mean(dim=0)  # (H,)
            if best_diagonal_score < diagonal_scores.max():
                best_diagonal_score = diagonal_scores.max()
                self.diag_head_info = (key, int(diagonal_scores.argmax()))


class LengthRegularizer(torch.nn.Module):
    """Length regularizer module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)

    :param float pad_value: value used for padding
    """

    def __init__(self, pad_value=0.0):
        super(LengthRegularizer, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, ilens):
        """Apply length regularizer

        :param torch.Tensor x: input tensor with the shape (B, Tmax, D)
        :param torch.Tensor d: duration of each components of each sequence B, T,)
        :param torch.Tensor d: batch of input lengths (B,)
        :return torch.Tensor: length regularized input tensor (B, T*, D)
        """
        xs = [x[:ilen] for x, ilen in zip(xs, ilens)]
        ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
        xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]

        return pad_list(xs, self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """Repeat each frame according to duration

        >>> x = torch.tensor([[1], [2], [3]])
        tensor([[1],
                [2],
                [3]])
        >>> d = torch.tensor([1, 2, 3])
        tensor([1, 2, 3])
        >>> self._repeat_one_sequence(x, d)
        tensor([[1],
                [2],
                [2],
                [3],
                [3],
                [3]])

        :param torch.Tensor x: input tensor with the shape (T, D)
        :param torch.Tensor d: duration of each frame of input tensor (T,)
        :return torch.Tensor: length regularized input tensor (T*, D)
        """
        return torch.cat([x_.repeat(int(d_), 1) for x_, d_ in zip(x, d)], dim=0)


class DurationPredictor(torch.nn.Module):
    """Duration predictor module

    Reference:
        - FastSpeech: Fast, Robust and Controllable Text to Speech
          (https://arxiv.org/pdf/1905.09263.pdf)

    :param int idim: input dimension
    :param int n_layers: number of convolutional layers
    :param int n_chans: number of channels of convolutional layers
    :param int kernel_size: kernel size of convolutional layers
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1):
        super(DurationPredictor, self).__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        """Calculate duration predictor forward propagation

        :param torch.Tensor xs: input tensor (B, Tmax, idim)
        :param torch.Tensor x_masks: mask of input tensor (non-padded part should be 1) (B, Tmax)
        :return torch.Tensor: predicted duration tensor in log domain (B, Tmax, 1)
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx in len(self.conv):
            xs = self.conv[idx](xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, 1)

        if x_masks is not None:
            x_masks = x_masks.eq(0).unsqueeze(-1)  # (B, Tmax, 1)
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def inference(self, xs, x_masks=None):
        """Inference duration

        :param torch.Tensor xs: input tensor with tha shape (B, Tmax, idim)
        :param torch.Tensor x_masks: mask of input tensor (non-padded part should be 1) with the shape (B, Tmax)
        :return torch.Tensor: predicted duration tensor with the shape (B, Tmax, 1)
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for idx in len(self.conv):
            xs = self.conv[idx](xs)  # (B, C, Tmax)
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax, 1)
        xs = torch.ceil(torch.exp(xs)).long()  # use ceil to avoid length = 0

        if x_masks is not None:
            x_masks = x_masks.eq(0).unsqueeze(-1)  # (B, Tmax, 1)
            xs = xs.masked_fill(x_masks, 0)

        return xs
