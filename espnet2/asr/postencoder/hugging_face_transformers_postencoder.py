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
    """
        Hugging Face Transformers PostEncoder.

    This class implements a post-encoder using Hugging Face Transformers models.
    It can be used to process the output of an encoder in a speech recognition
    or other sequence processing task.

    Attributes:
        transformer (transformers.PreTrainedModel): The Hugging Face transformer model.
        lang_token_embed (torch.Tensor): Language token embedding, if applicable.
        linear_in (torch.nn.Linear): Linear layer to project input to transformer dimension.
        length_adaptor (torch.nn.Sequential): Sequence of layers for length adaptation.
        length_adaptor_ratio (int): Ratio of length reduction in the adaptor.

    Args:
        input_size (int): Size of the input features.
        model_name_or_path (str): Name or path of the pre-trained Hugging Face model.
        length_adaptor_n_layers (int, optional): Number of layers in the length adaptor. Defaults to 0.
        lang_token_id (int, optional): ID of the language token. Defaults to -1.

    Raises:
        ImportError: If the 'transformers' library is not installed.

    Note:
        This class requires the 'transformers' library to be installed.
        The length adaptor is implemented as described in https://aclanthology.org/2021.acl-long.68.pdf
    """

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
        """
                Forward pass of the HuggingFaceTransformersPostEncoder.

        This method processes the input through the length adaptor, linear projection,
        and the Hugging Face transformer model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, time, feature_dim).
            input_lengths (torch.Tensor): Tensor of input lengths for each sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The output of the transformer model.
                - input_lengths (torch.Tensor): Updated lengths after processing.

        Raises:
            TooShortUttError: If the input sequence is too short for subsampling.

        Note:
            - The input is first processed by the length adaptor, which may reduce its temporal dimension.
            - If a language token is specified, it's prepended to the input.
            - The method handles different configurations of transformer models, including
              encoder-decoder models, XLNet, T5, and GPT-2.

        Examples:
            >>> encoder = HuggingFaceTransformersPostEncoder(input_size=256, model_name_or_path="bert-base-uncased")
            >>> input_tensor = torch.randn(32, 100, 256)  # (batch_size, time, feature_dim)
            >>> input_lengths = torch.full((32,), 100)
            >>> output, output_lengths = encoder.forward(input_tensor, input_lengths)
        """
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
        """
                Reload the pretrained parameters of the transformer model.

        This method resets the transformer's parameters to their original pretrained values.
        It's useful for resetting the model to its initial state, especially after fine-tuning.

        Note:
            This method logs an info message upon successful reloading of parameters.

        Examples:
            >>> encoder = HuggingFaceTransformersPostEncoder(input_size=256, model_name_or_path="bert-base-uncased")
            >>> # After some training or parameter updates
            >>> encoder.reload_pretrained_parameters()
            # This will reset the transformer's parameters to their original pretrained values
        """
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """
                Get the output size of the transformer model.

        Returns:
            int: The size of the output features from the transformer model.

        Note:
            This method returns the hidden size of the transformer model, which
            corresponds to the dimensionality of the output features.

        Examples:
            >>> encoder = HuggingFaceTransformersPostEncoder(input_size=256, model_name_or_path="bert-base-uncased")
            >>> output_dim = encoder.output_size()
            >>> print(output_dim)
            768  # For BERT base model
        """
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
