from collections import OrderedDict
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor


from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.rnn.encoders import RNN


class RNNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "blstm",
        num_spk: int = 2,
        nonlinear: str = "sigmoid",
        layer: int = 3,
        unit: int = 512,
        dropout: float = 0.0,
        mask_type: str = "IRM",
        loss_type: str = "mask_mse",
    ):
        super().__init__()

        self._num_spk = num_spk
        self.mask_type = mask_type
        self.loss_type = loss_type

        self.rnn = RNN(
            idim=input_dim,
            elayers=layer,
            cdim=unit,
            hdim=unit,
            dropout=dropout,
            typ=rnn_type,
        )

        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(unit, input_dim) for _ in range(self.num_spk)]
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

        x, ilens, _ = self.rnn(feature, ilens)

        masks = []

        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        maksed = [input * m for m in masks]

        others = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return maksed, ilens, others
    
    @property
    def num_spk(self):
        return self._num_spk

