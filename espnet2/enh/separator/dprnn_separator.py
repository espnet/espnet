from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.dprnn import DPRNN
from espnet2.enh.layers.dprnn import merge_feature
from espnet2.enh.layers.dprnn import split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator


class DPRNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
    ):
        """Dual-Path RNN (DPRNN) Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            segment_size: dual-path segment size
            dropout: float, dropout ratio. Default is 0.
        """
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
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]

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
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input

        B, T, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, T
        segmented, rest = split_feature(
            feature, segment_size=self.segment_size
        )  # B, N, L, K

        processed = self.dprnn(segmented)  # B, N*num_spk, L, K

        processed = merge_feature(processed, rest)  # B, N*num_spk, T

        processed = processed.transpose(1, 2)  # B, T, N*num_spk
        processed = processed.view(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
