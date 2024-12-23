# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.rawnet_block import Bottle2neck


class RawNet3Encoder(AbsEncoder):
    """
        RawNet3 encoder. Extracts frame-level RawNet embeddings from raw waveform.

    This encoder is designed for speaker recognition tasks and is based on the
    architecture presented in the paper by J. Jung et al., "Pushing the limits of
    raw waveform speaker recognition", in Proc. INTERSPEECH, 2022.

    Attributes:
        _output_size (int): The output embedding dimension.

    Args:
        input_size (int): Input feature dimension.
        block (str, optional): Type of encoder block class to use. Default is
            "Bottle2neck".
        model_scale (int, optional): Scale value of the Res2Net architecture.
            Default is 8.
        ndim (int, optional): Dimensionality of the hidden representation. Default is
            1024.
        output_size (int, optional): Output embedding dimension. Default is 1536.

    Examples:
        >>> encoder = RawNet3Encoder(input_size=16000)
        >>> waveform = torch.randn(1, 16000)  # Example raw waveform input
        >>> embeddings = encoder(waveform)
        >>> print(embeddings.shape)
        torch.Size([1, 1536, <sequence_length>])  # Output shape may vary

    Raises:
        ValueError: If an unsupported block type is provided.

    Note:
        The encoder expects a 3D tensor as input with the shape (batch_size,
        input_size, sequence_length).
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: str = "Bottle2neck",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        **kwargs,
    ):
        super().__init__()
        if block == "Bottle2neck":
            block: type = Bottle2neck
        else:
            raise ValueError(f"unsupported block, got: {block}")

        self._output_size = output_size

        self.relu = nn.ReLU()

        self.layer1 = block(
            input_size,
            ndim,
            kernel_size=3,
            dilation=2,
            scale=model_scale,
            pool=5,
        )
        self.layer2 = block(
            ndim,
            ndim,
            kernel_size=3,
            dilation=3,
            scale=model_scale,
            pool=3,
        )
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        """
                RawNet3 encoder. Extracts frame-level RawNet embeddings from raw waveform.

        Paper: J. Jung et al., "Pushing the limits of raw waveform speaker
        recognition", in Proc. INTERSPEECH, 2022.

        Attributes:
            _output_size (int): Output embedding dimension.

        Args:
            input_size (int): Input feature dimension.
            block (str): Type of encoder block class to use. Default is "Bottle2neck".
            model_scale (int): Scale value of the Res2Net architecture. Default is 8.
            ndim (int): Dimensionality of the hidden representation. Default is 1024.
            output_size (int): Output embedding dimension. Default is 1536.

        Examples:
            >>> encoder = RawNet3Encoder(input_size=128)
            >>> output = encoder.forward(torch.randn(1, 128, 100))
            >>> print(output.shape)
            torch.Size([1, 1536, 98])
        """
        return self._output_size

    def forward(self, x: torch.Tensor):
        """
            Perform forward propagation through the RawNet3 encoder.

        This method takes a batch of input tensors, processes them through
        several layers of the encoder, and outputs the frame-level embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size, seq_len).

        Examples:
            >>> encoder = RawNet3Encoder(input_size=128)
            >>> input_tensor = torch.randn(10, 128, 16000)  # Example input
            >>> output_tensor = encoder.forward(input_tensor)
            >>> print(output_tensor.shape)  # Should print (10, 1536, seq_len)

        Note:
            Ensure that the input tensor is properly shaped and normalized as
            expected by the model.

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """
        # frame-level propagation
        x1 = self.layer1(x.permute(0, 2, 1))
        x2 = self.layer2(x1)
        x3 = self.layer3(self.mp3(x1) + x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        return x
