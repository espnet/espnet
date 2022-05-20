from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.skim import SkiM
from espnet2.enh.separator.abs_separator import AbsSeparator


class SkiMSeparator(AbsSeparator):
    """Skipping Memory (SkiM) Separator

    Args:
        input_dim: input feature dimension
        causal: bool, whether the system is causal.
        num_spk: number of target speakers.
        nonlinear: the nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'
        layer: int, number of SkiM blocks. Default is 3.
        unit: int, dimension of the hidden state.
        segment_size: segmentation size for splitting long features
        dropout: float, dropout ratio. Default is 0.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states
            will be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):

        super().__init__()

        self._num_spk = num_spk

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
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
        else:
            feature = input

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

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
