#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModelForSeq2SeqLM

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersDecoder(AbsDecoder):
    """Hugging Face Transformers Decoder.

    Args:
        encoder_output_size: dimension of encoder attention
        model_name_or_path: Hugging Face Transformers model name
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
    ):
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        if hasattr(model, "model"):
            self.decoder = model.model.decoder
        else:
            self.decoder = model.decoder

        self.lm_head = model.lm_head
        self.model_name_or_path = model_name_or_path

        self.decoder_pretrained_params = copy.deepcopy(self.decoder.state_dict())
        self.lm_head_pretrained_params = copy.deepcopy(self.lm_head.state_dict())

        if encoder_output_size != self.decoder.config.hidden_size:
            self.linear_in = torch.nn.Linear(
                encoder_output_size, self.decoder.config.hidden_size
            )
        else:
            self.linear_in = torch.nn.Identity()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        args = {"return_dict": True}

        if self.decoder.__class__.__name__ == "MBartDecoder":
            ys_in_pad[:, 0] = 2

        args["input_ids"] = ys_in_pad
        mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
        args["attention_mask"] = mask

        args["encoder_hidden_states"] = self.linear_in(hs_pad)
        hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
        args["encoder_attention_mask"] = hs_mask

        x = self.decoder(**args).last_hidden_state
        x = self.lm_head(x)

        olens = mask.sum(1).to(torch.int)
        return x, olens

    def reload_pretrained_parameters(self):
        self.decoder.load_state_dict(self.decoder_pretrained_params)
        self.lm_head.load_state_dict(self.lm_head_pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")
