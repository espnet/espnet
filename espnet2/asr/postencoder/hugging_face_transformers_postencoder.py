#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError

try:
    from transformers import AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersPostEncoder(AbsPostEncoder):
    """Hugging Face Transformers PostEncoder."""

    @typechecked
    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
        length_adaptor_n_layers: int = 0,
        lang_token_id: int = -1,
    ):
        """Initialize the module."""
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

        self.lang_token_embed = None

        if hasattr(self.transformer, "embed_tokens"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.embed_tokens(torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )
            del self.transformer.embed_tokens
        if hasattr(self.transformer, "wte"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.wte(torch.tensor(lang_token_id)).detach().cpu()
                )
            del self.transformer.wte
        if hasattr(self.transformer, "word_embedding"):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    self.transformer.word_embedding(torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )
            del self.transformer.word_embedding
        if hasattr(model, "embeddings") and hasattr(
            model.embeddings, "word_embeddings"
        ):
            if lang_token_id != -1:
                self.lang_token_embed = (
                    model.embeddings.word_embeddings(torch.tensor(lang_token_id))
                    .detach()
                    .cpu()
                )

        if self.lang_token_embed is not None and hasattr(
            self.transformer, "embed_scale"
        ):
            self.lang_token_embed *= self.transformer.embed_scale

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        if (
            self.transformer.config.is_encoder_decoder
            or self.transformer.config.model_type in ["xlnet", "t5"]
        ):
            self.use_inputs_embeds = True
            self.extend_attention_mask = False
        elif self.transformer.config.model_type == "gpt2":
            self.use_inputs_embeds = True
            self.extend_attention_mask = True
        else:
            self.use_inputs_embeds = False
            self.extend_attention_mask = True

        self.linear_in = torch.nn.Linear(
            input_size, self.transformer.config.hidden_size
        )

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        if length_adaptor_n_layers > 0:
            length_adaptor_layers = []
            for _ in range(length_adaptor_n_layers):
                length_adaptor_layers.append(
                    torch.nn.Conv1d(input_size, input_size, 2, 2)
                )
                length_adaptor_layers.append(torch.nn.ReLU())
        else:
            length_adaptor_layers = [torch.nn.Identity()]

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

        if self.lang_token_embed is not None:
            lang_token_embed = (
                self.lang_token_embed.unsqueeze(0)
                .unsqueeze(0)
                .repeat(input.size(0), 1, 1)
            )
            input = torch.cat([lang_token_embed.to(input.device), input], dim=1)
            input_lengths = input_lengths + 1

        args = {"return_dict": True}

        mask = (~make_pad_mask(input_lengths)).to(input.device).float()

        if self.extend_attention_mask:
            args["attention_mask"] = _extend_attention_mask(mask)
        else:
            args["attention_mask"] = mask

        if self.use_inputs_embeds:
            args["inputs_embeds"] = input
        else:
            args["hidden_states"] = input

        if self.transformer.config.model_type == "mpnet":
            args["head_mask"] = [None for _ in self.transformer.layer]

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
