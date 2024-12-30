# x-vector, cross checked with SpeechBrain implementation:
# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/Xvector.py
# adapted for ESPnet-SPK by Jee-weon Jung
from typing import List

import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class XvectorEncoder(AbsEncoder):
    """
    X-vector encoder. Extracts frame-level x-vector embeddings from features.

    This class implements the X-vector model for speaker recognition as described
    in the paper by D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker
    recognition," presented at IEEE ICASSP, 2018. The model takes input features
    and processes them through a series of convolutional layers to produce
    speaker embeddings.

    Attributes:
        layers (nn.ModuleList): A list of convolutional layers, ReLU activations,
            and batch normalization layers.
        _output_size (int): The output embedding dimension.

    Args:
        input_size (int): Input feature dimension.
        ndim (int, optional): Dimensionality of the hidden representation.
            Defaults to 512.
        output_size (int, optional): Output embedding dimension. Defaults to 1500.
        kernel_sizes (List, optional): List of kernel sizes for each convolutional
            layer. Defaults to [5, 3, 3, 1, 1].
        paddings (List, optional): List of padding sizes for each convolutional
            layer. Defaults to [2, 1, 1, 0, 0].
        dilations (List, optional): List of dilation rates for each convolutional
            layer. Defaults to [1, 2, 3, 1, 1].
        **kwargs: Additional keyword arguments.

    Examples:
        >>> encoder = XvectorEncoder(input_size=40)
        >>> input_tensor = torch.randn(10, 100, 40)  # (Batch, Sequence, Features)
        >>> output = encoder(input_tensor)
        >>> print(output.shape)  # Output shape will depend on the configuration

    Note:
        This implementation is adapted for ESPnet-SPK by Jee-weon Jung, and
        cross-checked with the SpeechBrain implementation:
        https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/Xvector.py
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        ndim: int = 512,
        output_size: int = 1500,
        kernel_sizes: List = [5, 3, 3, 1, 1],
        paddings: List = [2, 1, 1, 0, 0],
        dilations: List = [1, 2, 3, 1, 1],
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        in_channels = [input_size] + [ndim] * 4
        out_channels = [ndim] * 4 + [output_size]

        self.layers = nn.ModuleList()
        for idx in range(5):
            self.layers.append(
                nn.Conv1d(
                    in_channels[idx],
                    out_channels[idx],
                    kernel_sizes[idx],
                    dilation=dilations[idx],
                    padding=paddings[idx],
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(out_channels[idx]))

    def output_size(self) -> int:
        """
            X-vector encoder. Extracts frame-level x-vector embeddings from features.

        Paper: D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker
        recognition," in Proc. IEEE ICASSP, 2018.

        Args:
            input_size: Input feature dimension.
            ndim: Dimensionality of the hidden representation.
            output_size: Output embedding dimension.

        Attributes:
            layers: A list of convolutional layers, activation functions, and batch
                normalization layers.

        Returns:
            int: The output embedding dimension.

        Examples:
            # Creating an instance of the XvectorEncoder
            encoder = XvectorEncoder(input_size=40, ndim=512, output_size=1500)

            # Accessing the output size
            print(encoder.output_size())  # Outputs: 1500

        Note:
            This class is adapted for ESPnet-SPK by Jee-weon Jung, and it is
            cross-checked with the SpeechBrain implementation.
        """
        return self._output_size

    def forward(self, x):
        """
            Forward pass of the X-vector encoder.

        This method processes the input tensor through the layers of the encoder,
        transforming the input features into frame-level x-vector embeddings. The
        input tensor should have the shape (B, S, D), where B is the batch size,
        S is the sequence length, and D is the feature dimension. The output will
        have the shape (B, output_size, new_S), where new_S is determined by the
        convolutional layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D).

        Returns:
            torch.Tensor: Output tensor after passing through the encoder layers,
            with shape (B, output_size, new_S).

        Examples:
            >>> import torch
            >>> encoder = XvectorEncoder(input_size=40)
            >>> input_tensor = torch.randn(10, 100, 40)  # (B, S, D)
            >>> output_tensor = encoder.forward(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([10, 1500, new_S])  # Output shape will vary based on new_S
        """
        x = x.permute(0, 2, 1)  # (B, S, D) -> (B, D, S)
        for layer in self.layers:
            x = layer(x)

        return x
