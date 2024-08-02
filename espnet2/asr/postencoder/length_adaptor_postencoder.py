#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length adaptor PostEncoder."""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError


class LengthAdaptorPostEncoder(AbsPostEncoder):
    """Length Adaptor PostEncoder."""

    @typechecked
    def __init__(
        self,
        input_size: int,
        length_adaptor_n_layers: int = 0,
        input_layer: Optional[str] = None,
        output_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        return_int_enc: bool = False,
        output_layer: Optional[str] = None,
    ):
        """Initialize the module."""
        super().__init__()

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            self.out_proj = torch.nn.Identity()
            self.in_sz = output_size
            self.out_sz = output_size
        elif output_layer == "linear":
            self.embed = torch.nn.Identity()
            self.out_proj = torch.nn.Linear(input_size, output_size)
            self.in_sz = input_size
            self.out_sz = output_size
        else:
            self.embed = torch.nn.Identity()
            self.out_proj = torch.nn.Identity()
            self.in_sz = input_size
            self.out_sz = input_size

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        if length_adaptor_n_layers > 0:
            length_adaptor_layers = []
            for _ in range(length_adaptor_n_layers):
                length_adaptor_layers.append(
                    torch.nn.Conv1d(self.in_sz, self.in_sz, 2, 2)
                )
                length_adaptor_layers.append(torch.nn.ReLU())
        else:
            length_adaptor_layers = [torch.nn.Identity()]

        self.length_adaptor = torch.nn.Sequential(*length_adaptor_layers)
        self.length_adaptor_ratio = 2**length_adaptor_n_layers
        self.return_int_enc = return_int_enc

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        if input.size(1) < self.length_adaptor_ratio:
            raise TooShortUttError(
                f"has {input.size(1)} frames and is too short for subsampling "
                + f"(it needs at least {self.length_adaptor_ratio} frames), "
                + "return empty results",
                input.size(1),
                self.length_adaptor_ratio,
            )

        input = self.embed(input)

        input = input.permute(0, 2, 1)
        input = self.length_adaptor(input)
        input = input.permute(0, 2, 1)

        input = self.out_proj(input)

        input_lengths = (
            input_lengths.float().div(self.length_adaptor_ratio).floor().long()
        )

        return input, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.out_sz
