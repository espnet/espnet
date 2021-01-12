from collections import OrderedDict
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor


from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcn import TCN


class TCNSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        layer: int = 8,
        stack: int = 3,
        bottleneck_dim: int = 128,
        hidden_dim: int = 512,
        kernel: int = 3,
        causal: bool = False,
        loss_type: str = "si_snr",
        nonlinear: str = "sigmoid",
    ):
        super().__init__()

        self._num_spk = num_spk
        self.loss_type = loss_type

        self.tcn = TCN(
            input_dim=input_dim,
            output_dim=input_dim * num_spk,
            BN_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            layer=layer,
            stack=stack,
            kernel=kernel,
            causal=causal,
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

        feature_tcn = feature.transpose(1, 2)  # B, N, L

        B, N, L = feature_tcn.shape

        masks = self.nonlinear(self.tcn(feature_tcn))  # B, num_spk * N, L
        masks = masks.transpose(1, 2)  # B, L, num_spk * N
        masks = masks.view(B, L, self.num_spk, N)  # B, L, num_spk, N
        masks = masks.unbind(dim=2)  # List[B, L, N]

        maksed = [feature * m for m in masks]

        others = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return maksed, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

