from collections import OrderedDict
from typing import Tuple, Union, List

import torch
from torch_complex.tensor import ComplexTensor


from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)
from espnet2.enh.separator.abs_separator import AbsSeparator


class TransformerSeparator(AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        adim: int = 384,
        aheads: int = 4,
        layers: int = 6,
        linear_units: int = 1536,
        positionwise_layer_type: str = "linear",
        input_layer: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        normalize_before: bool = False,
        concat_after: bool = False,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        use_scaled_pos_enc: bool = True,
        nonlinear: str = "relu",
    ):
        super().__init__()

        self._num_spk = num_spk

        pos_enc_class = (
            ScaledPositionalEncoding if use_scaled_pos_enc else PositionalEncoding
        )
        self.transformer = TransformerEncoder(
            idim=input_dim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=linear_units,
            num_blocks=layers,
            input_layer=input_layer,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(adim, input_dim) for _ in range(self.num_spk)]
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
                'spk1': torch.Tensor(Batch, Frames, Freq),
                'spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if isinstance(input, ComplexTensor):
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

        maksed = [input * m for m in masks]

        others = OrderedDict(
            zip(["spk{}".format(i + 1) for i in range(len(masks))], masks)
        )

        return maksed, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

