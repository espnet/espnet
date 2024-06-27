from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.dptnet import DPTNet
from espnet2.enh.layers.tcn import choose_norm
from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DPTNetSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        post_enc_relu: bool = True,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        unit: int = 256,
        att_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "gLN",
        layer: int = 6,
        segment_size: int = 20,
        nonlinear: str = "relu",
    ):
        """Dual-Path Transformer Network (DPTNet) Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            unit: int, dimension of the hidden state.
            att_heads: number of attention heads.
            dropout: float, dropout ratio. Default is 0.
            activation: activation function applied at the output of RNN.
            norm_type: type of normalization to use after each inter- or
                intra-chunk Transformer block.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            segment_size: dual-path segment size
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.segment_size = segment_size

        self.post_enc_relu = post_enc_relu
        self.enc_LN = choose_norm(norm_type, input_dim)
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.dptnet = DPTNet(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            att_heads=att_heads,
            dropout=dropout,
            activation=activation,
            num_layers=layer,
            bidirectional=bidirectional,
            norm_type=norm_type,
        )
        # gated output layer
        self.output = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Tanh()
        )
        self.output_gate = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Sigmoid()
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        elif self.post_enc_relu:
            feature = torch.nn.functional.relu(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T
        feature = self.enc_LN(feature)
        segmented = self.split_feature(feature)  # B, N, L, K

        processed = self.dptnet(segmented)  # B, N*num_spk, L, K
        processed = processed.reshape(
            B * self.num_outputs, -1, processed.size(-2), processed.size(-1)
        )  # B*num_spk, N, L, K

        processed = self.merge_feature(processed, length=T)  # B*num_spk, N, T

        # gated output layer for filter generation (B*num_spk, N, T)
        processed = self.output(processed) * self.output_gate(processed)

        masks = processed.reshape(B, self.num_outputs, N, T)

        # list[(B, T, N)]
        masks = self.nonlinear(masks.transpose(-1, -2)).unbind(dim=1)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    def split_feature(self, x):
        B, N, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, N, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        B, N, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, N * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, N, length)

    @property
    def num_spk(self):
        return self._num_spk
