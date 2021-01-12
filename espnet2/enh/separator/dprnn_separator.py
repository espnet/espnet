from collections import OrderedDict
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor


from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.dprnn import DPRNN, split_feature, merge_feature


class DPRNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._num_spk = num_spk


        self.segment_size = segment_size

        self.dprnn = DPRNN(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_layers=layer,
            bidirectional=bidirectional,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:

        # if complex spectrum,
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T
        segmented, rest = split_feature(feature, segment_size=self.segment_size) # B, N, L, K

        processed = self.dprnn(segmented) # B, N*num_spk, L, K

        processed = merge_feature(processed, rest) # B, N*num_spk, T

        processed = processed.transpose(1,2) # B, T, N*num_spk
        processed = processed.view(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)

        maksed = [input * m for m in masks]

        others = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return maksed, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

