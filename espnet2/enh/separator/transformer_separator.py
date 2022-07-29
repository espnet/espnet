from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,
)

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class TransformerSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        predict_noise: bool = False,
        adim: int = 384,
        aheads: int = 4,
        layers: int = 6,
        linear_units: int = 1536,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        normalize_before: bool = False,
        concat_after: bool = False,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_scaled_pos_enc: bool = True,
        nonlinear: str = "relu",
    ):
        """Transformer separator.

        Args:
            input_dim: input feature dimension
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            adim (int): Dimension of attention.
            aheads (int): The number of heads of multi head attention.
            linear_units (int): The number of units of position-wise feed forward.
            layers (int): The number of transformer blocks.
            dropout_rate (float): Dropout rate.
            attention_dropout_rate (float): Dropout rate in attention.
            positional_dropout_rate (float): Dropout rate after adding
                                             positional encoding.
            normalize_before (bool): Whether to use layer_norm before the first block.
            concat_after (bool): Whether to concat attention layer's input and output.
                if True, additional linear will be applied.
                i.e. x -> x + linear(concat(x, att(x)))
                if False, no additional linear will be applied. i.e. x -> x + att(x)
            positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
            positionwise_conv_kernel_size (int): Kernel size of
                                                 positionwise conv1d layer.
            use_scaled_pos_enc (bool) : use scaled positional encoding or not
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        pos_enc_class = (
            ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
        )
        self.transformer = TransformerEncoder(
            idim=input_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers,
            input_layer="linear",
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(adim, input_dim) for _ in range(num_outputs)]
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

        # prepare pad_mask for transformer
        pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(feature.device)

        x, ilens = self.transformer(feature, pad_mask)

        masks = []
        for linear in self.linear:
            y = linear(x)
            y = self.nonlinear(y)
            masks.append(y)

        if self.predict_noise:
            *masks, mask_noise = masks

        masked = [input * m for m in masks]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk
