#!/usr/bin/env python3
#  2023, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length Adaptor PostEncoder."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError


class LengthAdaptorPostEncoder(AbsPostEncoder):
    """Length Adaptor PostEncoder."""

    def __init__(
        self,
        input_size: int,
        length_adaptor_n_layers: int = 1,
        output_size: int = 0,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        assert length_adaptor_n_layers > 0

        if output_size != 0:
            self.linear_in = torch.nn.Linear(input_size, output_size)
            self._output_size = output_size
        else:
            self.linear_in = torch.nn.Identity()
            self._output_size = input_size

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        length_adaptor_layers = []
        for _ in range(length_adaptor_n_layers):
            length_adaptor_layers.append(torch.nn.Conv1d(input_size, input_size, 2, 2))
            length_adaptor_layers.append(torch.nn.ReLU())

        self.length_adaptor = torch.nn.Sequential(*length_adaptor_layers)
        self.length_adaptor_ratio = 2**length_adaptor_n_layers

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

        input = input.permute(0, 2, 1)
        input = self.length_adaptor(input)
        input = input.permute(0, 2, 1)

        input_lengths = (
            input_lengths.float().div(self.length_adaptor_ratio).floor().long()
        )

        input = self.linear_in(input)

        return input, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self._output_size
