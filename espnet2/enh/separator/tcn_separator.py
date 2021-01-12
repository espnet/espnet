from collections import OrderedDict
from math import frexp
from espnet2.enh.layers.mask_estimator import MaskEstimator
from typing import Tuple, Union, List
import torch
from torch_complex.tensor import ComplexTensor


from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.tcn import TemporalConvNet


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
        norm_type: str = "gLN",
        nonlinear: str = "relu",
    ):
        super().__init__()

        self._num_spk = num_spk

        self.tcn = TemporalConvNet(
            N=input_dim,
            B=bottleneck_dim,
            H=hidden_dim,
            P=kernel,
            X=layer,
            R=stack,
            C=num_spk,
            norm_type=norm_type,
            causal=causal,
            mask_nonlinear=nonlinear,
        )

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
                'spk1': torch.Tensor(Batch, Frames, Freq),
                'spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # if complex spectrum
        if isinstance(input, ComplexTensor):
            feature = abs(input)
        else:
            feature = input
        B, L, N = feature.shape

        feature = feature.transpose(1, 2)  # B, N, L

        masks = self.tcn(feature)  # B, num_spk, N, L
        masks = masks.transpose(2, 3)  # B, num_spk, L, N
        masks = masks.unbind(dim=1)  # List[B, L, N]

        maksed = [input * m for m in masks]

        others = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return maksed, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

