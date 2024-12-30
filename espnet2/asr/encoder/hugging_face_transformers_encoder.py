#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

import copy
import logging
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersEncoder(AbsEncoder):
    """
    Hugging Face Transformers Encoder for Automatic Speech Recognition.

    This class serves as an encoder that utilizes pre-trained models from the
    Hugging Face Transformers library for processing input sequences in
    automatic speech recognition tasks. It supports optional language token
    integration and manages attention masks to ensure effective sequence
    processing.

    Attributes:
        transformer (transformers.PreTrainedModel): The underlying transformer
            model for encoding.
        pretrained_params (dict): A copy of the model's parameters for
            reloading purposes.
        lang_token_id (int): The token ID for the language token, if used.

    Args:
        input_size (int): The size of the input feature vector.
        model_name_or_path (str): The model identifier from Hugging Face's model
            hub or a local path to a model.
        lang_token_id (int, optional): The token ID for the language token to
            prepend to inputs. Defaults to -1 (disabled).

    Raises:
        ImportError: If the `transformers` library is not available.

    Examples:
        >>> encoder = HuggingFaceTransformersEncoder(
        ...     input_size=512,
        ...     model_name_or_path='bert-base-uncased',
        ...     lang_token_id=101
        ... )
        >>> input_tensor = torch.randint(0, 1000, (3, 512))
        >>> input_lengths = torch.tensor([512, 300, 250])
        >>> output, lengths = encoder(input_tensor, input_lengths)
        >>> print(output.shape)  # Output shape will depend on the model
        >>> print(lengths)  # Adjusted input lengths after processing

    Note:
        Ensure that the `transformers` library is installed before using this
        class. You can install it via `pip install transformers`.

    Todo:
        - Implement support for more transformer model variants.
        - Add more extensive error handling for input dimensions.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
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

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        self.lang_token_id = lang_token_id

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Hugging Face Transformers encoder.

        This method processes the input tensor through the transformer model,
        optionally prepending a language token if `lang_token_id` is specified.
        It also generates an attention mask based on the input lengths to
        ensure that padding tokens are ignored during the attention
        computation.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size,
                sequence_length) containing token IDs.
            input_lengths (torch.Tensor): A tensor of shape (batch_size,)
                containing the lengths of each input sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple where the first element
            is the output tensor of shape (batch_size, sequence_length,
            hidden_size) from the transformer, and the second element is
            the updated input_lengths tensor.

        Examples:
            >>> encoder = HuggingFaceTransformersEncoder(768, "bert-base-uncased")
            >>> input_tensor = torch.tensor([[101, 2009, 2003, 102]])
            >>> input_lengths = torch.tensor([4])
            >>> output, lengths = encoder.forward(input_tensor, input_lengths)
            >>> print(output.shape)  # Output: torch.Size([1, 4, 768])
            >>> print(lengths)  # Output: tensor([4])

        Note:
            Ensure that the `transformers` library is installed before
            using this method. If the library is not available, an
            ImportError will be raised during the initialization of the
            encoder.

        Raises:
            ImportError: If the `transformers` library is not available.

        Todo:
            Add support for additional configuration options for the
            transformer model in the future.
        """

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
        """
        Reload the pretrained parameters of the transformer model.

        This method restores the model's parameters to their initial state
        that was saved during the initialization of the encoder. It can be
        useful for resetting the model's weights to the pretrained values
        after fine-tuning or training on a specific task.

        Example:
            >>> encoder = HuggingFaceTransformersEncoder(input_size=512,
            ...     model_name_or_path='bert-base-uncased')
            >>> # After some training or modifications
            >>> encoder.reload_pretrained_parameters()
            Pretrained Transformers model parameters reloaded!

        Note:
            Ensure that the `transformers` library is installed to utilize
            this functionality. If the model parameters have not been set
            (e.g., after initialization), calling this method will reload
            the parameters to their original pretrained state.

        Raises:
            ValueError: If the pretrained parameters are not set or if
            there is a mismatch between the model architecture and the
            loaded parameters.
        """
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """
        Get the output size of the transformer model.

        This method retrieves the hidden size of the transformer model, which
        corresponds to the dimensionality of the output embeddings produced
        by the model.

        Returns:
            int: The hidden size of the transformer model.

        Examples:
            >>> encoder = HuggingFaceTransformersEncoder(
            ...     input_size=128,
            ...     model_name_or_path='bert-base-uncased'
            ... )
            >>> encoder.output_size()
            768  # For BERT base model

        Note:
            The output size may vary depending on the specific transformer model
            architecture being used.
        """
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
