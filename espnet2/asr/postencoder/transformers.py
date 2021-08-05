#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from typeguard import check_argument_types
from typing import Tuple

import torch


class Transformers(AbsPostEncoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        try:
            from transformers import AutoModel
        except Exception as e:
            print("Error: Transformers is not properly installed.")
            print(
                "Please install Transformers: cd ${MAIN_ROOT}/tools && make transformers.done"
            )
            raise e

        model = AutoModel.from_pretrained(model_name_or_path)
        self.transformer = model.encoder
        self.linear_in = torch.nn.Linear(
            input_size, self.transformer.config.hidden_size
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input = self.linear_in(input)

        mask = (~make_pad_mask(input_lengths)).to(input.device).float()
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        output = self.transformer(input, attention_mask=mask).last_hidden_state

        return output, input_lengths

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.config.hidden_size
