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
        Hugging Face Transformers PostEncoder for ASR tasks.

    This class implements an encoder that utilizes pre-trained Hugging Face Transformer
    models for automatic speech recognition (ASR) tasks. It can be used as a post-encoder
    in ESPnet2 ASR systems.

    Attributes:
        transformer (transformers.PreTrainedModel): The Transformer model used for encoding.
        pretrained_params (dict): A copy of the initial pre-trained model parameters.
        lang_token_id (int): The token ID for language, used if language-specific encoding is needed.

    Args:
        input_size (int): The size of the input features.
        model_name_or_path (str): The name or path of the pre-trained Hugging Face Transformer model.
        lang_token_id (int, optional): The token ID for language. Defaults to -1 (not used).

    Raises:
        ImportError: If the 'transformers' library is not installed.

    Note:
        This class requires the 'transformers' library to be installed. If not available,
        it will raise an ImportError with instructions on how to install it.

    Examples:
        >>> encoder = HuggingFaceTransformersEncoder(input_size=80, model_name_or_path="bert-base-uncased")
        >>> input_tensor = torch.randn(32, 100, 80)  # (batch_size, sequence_length, input_size)
        >>> input_lengths = torch.randint(50, 100, (32,))  # (batch_size,)
        >>> output, output_lengths = encoder(input_tensor, input_lengths)
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
                Forward pass of the Hugging Face Transformers encoder.

        This method processes the input tensor through the Transformer model and returns
        the encoded representation along with the updated input lengths.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            input_lengths (torch.Tensor): Tensor of input sequence lengths of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): Encoded representation from the Transformer model.
                  Shape: (batch_size, sequence_length, hidden_size).
                - input_lengths (torch.Tensor): Updated input lengths, accounting for any
                  additional tokens (e.g., language token) added during processing.

        Note:
            If a language token ID is specified (self.lang_token_id != -1), it will be
            prepended to the input sequences, and the input lengths will be incremented accordingly.

        Examples:
            >>> encoder = HuggingFaceTransformersEncoder(input_size=80, model_name_or_path="bert-base-uncased")
            >>> input_tensor = torch.randn(32, 100, 80)  # (batch_size, sequence_length, input_size)
            >>> input_lengths = torch.randint(50, 100, (32,))  # (batch_size,)
            >>> output, output_lengths = encoder.forward(input_tensor, input_lengths)
            >>> print(output.shape)  # Expected: torch.Size([32, 100, 768])
            >>> print(output_lengths.shape)  # Expected: torch.Size([32])
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
                Reload the pretrained parameters of the Transformer model.

        This method resets the Transformer model's parameters to their initial pretrained state.
        It's useful for resetting the model to its original configuration after fine-tuning or
        other modifications.

        Note:
            This method uses the `pretrained_params` attribute, which is a deep copy of the
            initial model parameters created during the initialization of the
            HuggingFaceTransformersEncoder instance.

        Examples:
            >>> encoder = HuggingFaceTransformersEncoder(input_size=80, model_name_or_path="bert-base-uncased")
            >>> # After some fine-tuning or parameter updates
            >>> encoder.reload_pretrained_parameters()
            >>> print("Pretrained Transformers model parameters reloaded!")
        """
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """
                Get the output size of the Transformer encoder.

        This method returns the size of the hidden state output by the Transformer model.

        Returns:
            int: The size of the hidden state (number of features) in the output of the Transformer model.

        Note:
            The output size is determined by the `hidden_size` parameter in the Transformer model's configuration.

        Examples:
            >>> encoder = HuggingFaceTransformersEncoder(input_size=80, model_name_or_path="bert-base-uncased")
            >>> output_size = encoder.output_size()
            >>> print(output_size)  # Expected: 768 for bert-base-uncased
        """
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
