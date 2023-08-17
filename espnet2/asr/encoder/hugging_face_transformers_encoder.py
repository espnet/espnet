#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersEncoder(AbsEncoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
        lang_token_id: int = -1,
    ):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        model = AutoModel.from_pretrained(model_name_or_path)

        if hasattr(model, "encoder"):
            self.transformer = model.encoder
        else:
            self.transformer = model

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        self.lang_token_id = lang_token_id

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""

        args = {"return_dict": True}

        if self.lang_token_id != -1:
            input = torch.cat(
                (
                    torch.tensor(
                        [self.lang_token_id] * input.shape[0], device=input.device
                    ).unsqueeze(1),
                    input,
                ),
                dim=-1,
            )
            input_lengths = input_lengths + 1

        args["input_ids"] = input

        mask = (~make_pad_mask(input_lengths)).to(input.device).float()
        args["attention_mask"] = mask
        output = self.transformer(**args).last_hidden_state

        return output, input_lengths

    def reload_pretrained_parameters(self):
        self.transformer.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
