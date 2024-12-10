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
    """
    Length Adaptor PostEncoder.

    This class implements a Length Adaptor PostEncoder which is a component 
    designed to adapt the length of input sequences through convolutional 
    layers, based on the specified parameters. It is particularly useful in 
    asynchronous speech recognition systems where input lengths may vary.

    Attributes:
        embed (torch.nn.Sequential): The embedding layer for input processing.
        out_sz (int): The output size of the encoder.
        length_adaptor (torch.nn.Sequential): The sequential layers for length 
            adaptation.
        length_adaptor_ratio (int): The ratio by which input lengths are 
            adjusted.
        return_int_enc (bool): A flag to determine if the integer encoding should 
            be returned.

    Args:
        input_size (int): The size of the input features.
        length_adaptor_n_layers (int, optional): The number of convolutional 
            layers in the length adaptor. Defaults to 0.
        input_layer (Optional[str], optional): Type of input layer ('linear' or 
            None). Defaults to None.
        output_size (Optional[int], optional): The size of the output features 
            if input_layer is 'linear'. Defaults to None.
        dropout_rate (float, optional): The dropout rate for regularization. 
            Defaults to 0.1.
        return_int_enc (bool, optional): Whether to return integer encoding. 
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The adapted input tensor and the 
        updated lengths of the input sequences.

    Raises:
        TooShortUttError: If the input sequence length is shorter than the 
            required length for subsampling.

    Examples:
        >>> post_encoder = LengthAdaptorPostEncoder(input_size=128, 
        ...                                          length_adaptor_n_layers=2, 
        ...                                          output_size=256)
        >>> input_tensor = torch.randn(10, 50, 128)  # (batch_size, seq_len, features)
        >>> input_lengths = torch.tensor([50] * 10)  # Lengths of each input
        >>> output, new_lengths = post_encoder(input_tensor, input_lengths)
        >>> print(output.shape)  # Should reflect the adapted length
        >>> print(new_lengths)  # Updated lengths after adaptation

    Note:
        This implementation follows the design described in the paper 
        "Length Adaptor for End-to-End ASR" (ACL 2021).

    Todo:
        - Extend the functionality to allow for more input layer types.
        - Implement additional error handling for various edge cases.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        length_adaptor_n_layers: int = 0,
        input_layer: Optional[str] = None,
        output_size: Optional[int] = None,
        dropout_rate: float = 0.1,
        return_int_enc: bool = False,
    ):
        """Initialize the module."""
        super().__init__()

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            self.out_sz = output_size
        else:
            self.embed = None
            self.out_sz = input_size

        # Length Adaptor as in https://aclanthology.org/2021.acl-long.68.pdf

        if length_adaptor_n_layers > 0:
            length_adaptor_layers = []
            for _ in range(length_adaptor_n_layers):
                length_adaptor_layers.append(
                    torch.nn.Conv1d(self.out_sz, self.out_sz, 2, 2)
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
        """
        Forward pass through the LengthAdaptorPostEncoder.

        This method takes an input tensor and its corresponding lengths, applies
        the embedding layer if specified, processes the input through the length
        adaptor, and returns the transformed input along with updated lengths.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, 
                                  input_size, sequence_length).
            input_lengths (torch.Tensor): Tensor of shape (batch_size,) 
                                           containing the lengths of each 
                                           input sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): Transformed output tensor of shape 
                                         (batch_size, output_size, 
                                         new_sequence_length).
                - output_lengths (torch.Tensor): Updated lengths of the output 
                                                   sequences.

        Raises:
            TooShortUttError: If the input sequence is shorter than the 
                              required length for subsampling.

        Examples:
            >>> encoder = LengthAdaptorPostEncoder(input_size=128, 
            ...                                     length_adaptor_n_layers=2)
            >>> input_tensor = torch.randn(10, 128, 20)  # batch_size=10
            >>> input_lengths = torch.tensor([20] * 10)  # all sequences have length 20
            >>> output, output_lengths = encoder.forward(input_tensor, input_lengths)

        Note:
            The length adaptor reduces the sequence length by a factor of 
            `2 ** length_adaptor_n_layers`. Ensure that the input sequences 
            are sufficiently long to avoid raising the TooShortUttError.
        """
        if input.size(1) < self.length_adaptor_ratio:
            raise TooShortUttError(
                f"has {input.size(1)} frames and is too short for subsampling "
                + f"(it needs at least {self.length_adaptor_ratio} frames), "
                + "return empty results",
                input.size(1),
                self.length_adaptor_ratio,
            )

        if self.embed is not None:
            input = self.embed(input)

        input = input.permute(0, 2, 1)
        input = self.length_adaptor(input)
        input = input.permute(0, 2, 1)

        input_lengths = (
            input_lengths.float().div(self.length_adaptor_ratio).floor().long()
        )

        return input, input_lengths

    def output_size(self) -> int:
        """
        Get the output size.

        This method returns the output size of the LengthAdaptorPostEncoder,
        which is determined during initialization. The output size is either
        set explicitly through the `output_size` parameter or defaults to the
        `input_size` if no embedding layer is used.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> encoder = LengthAdaptorPostEncoder(input_size=256, output_size=128)
            >>> encoder.output_size()
            128

            >>> encoder_no_embed = LengthAdaptorPostEncoder(input_size=256)
            >>> encoder_no_embed.output_size()
            256

        Note:
            The output size is important for downstream tasks and should
            be configured based on the model architecture requirements.
        """
        return self.out_sz
