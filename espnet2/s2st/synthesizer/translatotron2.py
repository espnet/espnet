# Copyright 2022 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Translatotron2 related modules for ESPnet2."""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.s2st.synthesizer.abs_synthesizer import AbsSynthesizer
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.fastspeech2.loss import FastSpeech2Loss
from espnet2.tts.fastspeech2.variance_predictor import VariancePredictor
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)


class Translatotron2(AbsSynthesizer):
    """Translatotron2 module.

    This is a module of the synthesizer in Translatotron2 described in `Translatotron 2:
    High-quality direct speech-to-speech translation with voice preservation`_.

    .. _`Translatotron 2:
    High-quality direct speech-to-speech translation with voice preservation`:
        https://arxiv.org/pdf/2107.08661v5.pdf

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        synthesizer_type: str = "rnn",
        layers: int = 2,
        units: int = 1024,
        # for prenet
        prenet_layers: int = 2,
        prenet_units: int = 128,
        prenet_dropout_rate: float = 0.5,
        # for postnet
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_dropout_rate: float = 0.5,
        # for transformer
        adim: int = 384,
        aheads: int = 4,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # duration predictor
        duration_predictor_layers: int = 2,
        duration_predictor_type: str = "rnn",
        duration_predictor_units: int = 128,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        return


class Prenet(nn.Module):
    """Non-Attentive Tacotron (NAT) Prenet."""

    def __init__(self, idim, units=128, num_layers=2, dropout=0.5):
        super(Prenet, self).__init__()
        sizes = [units] * num_layers
        in_sizes = [idim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_p)
        self.activation = nn.ReLU()

    def forward(self, x):
        for linear in self.layers:
            x = self.dropout(self.activation(linear(x)))
        return x


class DurationPredictor(nn.Module):
    """Non-Attentive Tacotron (NAT) Duration Predictor module."""

    def __init__(self, cfg):
        super(DurationPredictor, self).__init__()

        self.lstm = nn.LSTM(
            units,
            int(cfg.duration_lstm_dim / 2),
            2,
            batch_first=True,
            bidirectional=True,
        )

        self.proj = LinearNorm(cfg.duration_lstm_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, input_lengths=None):
        """
        :param encoder_outputs: [batch_size, hidden_length, encoder_lstm_dim]
        :param input_lengths: [batch_size, hidden_length]
        :return: [batch_size, hidden_length]
        """

        batch_size = encoder_outputs.size(0)
        hidden_length = encoder_outputs.size(1)

        ## remove pad activations
        if input_lengths is not None:
            encoder_outputs = pack_padded_sequence(
                encoder_outputs, input_lengths, batch_first=True, enforce_sorted=False
            )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(encoder_outputs)

        if input_lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        outputs = self.relu(self.proj(outputs))

        return outputs.view(batch_size, hidden_length)


class GaussianUpsampling(nn.Module):
    """
    Non-attention Tacotron:
        - https://arxiv.org/abs/2010.04301
    this source code is implemenation of the ExpressiveTacotron from BridgetteSong
        - https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_duration.py
    """

    def __init__(self):
        super(GaussianUpsampling, self).__init__()
        self.mask_score = -1e15

    def forward(self, encoder_outputs, durations, vars, input_lengths=None):
        """Gaussian upsampling.

        Args:
            encoder_outputs: encoder outputs  [batch_size, hidden_length, dim]
            durations: phoneme durations  [batch_size, hidden_length]
            vars : phoneme attended ranges [batch_size, hidden_length]
            input_lengths : [batch_size]

        Return:
            encoder_upsampling_outputs: upsampled encoder_output  [batch_size, frame_length, dim]
        """
        batch_size = encoder_outputs.size(0)
        hidden_length = encoder_outputs.size(1)
        frame_length = int(torch.sum(durations, dim=1).max().item())
        c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
        c = c.unsqueeze(2)  # [batch_size, hidden_length, 1]
        t = (
            torch.arange(frame_length, device=encoder_outputs.device)
            .expand(batch_size, hidden_length, frame_length)
            .float()
        )  # [batch_size, hidden_length, frame_length]
        vars = vars.view(batch_size, -1, 1)  # [batch_size, hidden_length, 1]

        w_t = -0.5 * (
            np.log(2.0 * np.pi) + torch.log(vars) + torch.pow(t - c, 2) / vars
        )  # [batch_size, hidden_length, frame_length]

        if input_lengths is not None:
            input_masks = ~self.get_mask_from_lengths(
                input_lengths, hidden_length
            )  # [batch_size, hidden_length]
            input_masks = torch.tensor(input_masks, dtype=torch.bool, device=w_t.device)
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)

        encoder_upsampling_outputs = torch.bmm(
            w_t.transpose(1, 2), encoder_outputs
        )  # [batch_size, frame_length, encoder_hidden_size]

        return encoder_upsampling_outputs

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = max(lengths)
        ids = np.arange(0, max_len)
        mask = ids < lengths.reshape(-1, 1)
        return mask
