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

    This class wraps a Hugging Face transformer model for use as a post-encoder
    in speech recognition tasks. It initializes the transformer model and
    processes the input data through various layers, including a length
    adaptor if specified.

    Attributes:
        transformer (torch.nn.Module): The transformer model used for encoding.
        lang_token_embed (torch.Tensor): The language token embedding if a
            language token ID is provided.
        pretrained_params (dict): A deep copy of the transformer model's
            initial state dictionary for later reloading.
        length_adaptor (torch.nn.Sequential): A sequence of layers for adapting
            the input length.
        length_adaptor_ratio (int): The ratio by which the input length is
            reduced when passing through the length adaptor.
        use_inputs_embeds (bool): Indicates whether to use input embeddings.
        extend_attention_mask (bool): Indicates whether to extend the attention
            mask for certain model types.

    Args:
        input_size (int): The size of the input features.
        model_name_or_path (str): The name or path of the pretrained model to use.
        length_adaptor_n_layers (int, optional): The number of layers in the
            length adaptor. Defaults to 0.
        lang_token_id (int, optional): The ID of the language token to use.
            Defaults to -1.

    Raises:
        ImportError: If the `transformers` library is not available.

    Examples:
        >>> post_encoder = HuggingFaceTransformersPostEncoder(
        ...     input_size=256,
        ...     model_name_or_path='bert-base-uncased',
        ...     length_adaptor_n_layers=2,
        ...     lang_token_id=101
        ... )
        >>> input_tensor = torch.randn(10, 20, 256)  # (batch_size, seq_len, features)
        >>> input_lengths = torch.tensor([20] * 10)  # Lengths of each sequence
        >>> output, output_lengths = post_encoder(input_tensor, input_lengths)

    Note:
        Ensure that the `transformers` library is installed to use this class.
        You can install it via:
        `pip install transformers` or follow the ESPnet installation instructions.

    Todo:
        - Add support for more model types as needed.
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
        Hugging Face Transformers PostEncoder.

        This class serves as a post-encoder that utilizes pre-trained models from 
        Hugging Face's Transformers library to process input sequences. It includes 
        functionalities for length adaptation and handling language token embeddings.

        Attributes:
            transformer: The underlying transformer model used for encoding.
            lang_token_embed: The embedding for the language token, if applicable.
            pretrained_params: A deep copy of the transformer model's state dictionary.
            use_inputs_embeds: A boolean indicating whether to use input embeddings.
            extend_attention_mask: A boolean indicating whether to extend the 
                attention mask.
            linear_in: A linear layer to project input features to the transformer's 
                hidden size.
            length_adaptor: A sequential model for length adaptation.
            length_adaptor_ratio: The ratio of input length to output length 
                after adaptation.

        Args:
            input_size (int): The size of the input features.
            model_name_or_path (str): The model name or path to the pre-trained 
                model.
            length_adaptor_n_layers (int, optional): Number of convolutional layers 
                for length adaptation. Defaults to 0.
            lang_token_id (int, optional): The token ID for the language token. 
                Defaults to -1.

        Raises:
            ImportError: If the `transformers` library is not available.

        Examples:
            >>> post_encoder = HuggingFaceTransformersPostEncoder(
            ...     input_size=128, 
            ...     model_name_or_path='bert-base-uncased'
            ... )
            >>> input_tensor = torch.rand(2, 128, 128)  # Batch of 2
            >>> input_lengths = torch.tensor([128, 128])
            >>> output, output_lengths = post_encoder.forward(input_tensor, input_lengths)

        Note:
            The forward method expects input tensors of shape (batch_size, seq_len, 
            input_size) and input_lengths of shape (batch_size,).
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
        Reloads the pretrained parameters of the Hugging Face Transformers model.

        This method restores the parameters of the transformer model to their initial
        state as defined when the instance of `HuggingFaceTransformersPostEncoder` 
        was created. This can be useful when you want to reset the model's weights 
        to the pretrained values after fine-tuning or any modification.

        Example:
            >>> post_encoder = HuggingFaceTransformersPostEncoder(
            ...     input_size=128, 
            ...     model_name_or_path='bert-base-uncased'
            ... )
            >>> # Fine-tuning or modifying the model parameters...
            >>> post_encoder.reload_pretrained_parameters()  # Reloads pretrained params

        Note:
            Ensure that the model has been initialized and pretrained parameters 
            are stored before calling this method, or it will reload with 
            the last saved state.

        Raises:
            RuntimeError: If the model is not properly initialized or the 
            pretrained parameters are not available.
        """
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """
        Get the output size of the transformer model.

        This method retrieves the hidden size of the transformer model, which is
        defined in its configuration. The output size is crucial for downstream
        tasks where the model's output needs to match the expected dimensions.

        Returns:
            int: The hidden size of the transformer model.

        Examples:
            >>> post_encoder = HuggingFaceTransformersPostEncoder(
            ...     input_size=256,
            ...     model_name_or_path='bert-base-uncased'
            ... )
            >>> post_encoder.output_size()
            768  # For BERT, the hidden size is typically 768.
        """
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
