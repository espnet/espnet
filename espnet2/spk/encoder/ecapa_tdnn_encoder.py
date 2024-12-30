# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""
ECAPA-TDNN Encoder
"""

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.ecapa_block import EcapaBlock


class EcapaTdnnEncoder(AbsEncoder):
    """
        ECAPA-TDNN Encoder

    This class implements the ECAPA-TDNN encoder, which extracts frame-level
    ECAPA-TDNN embeddings from mel-filterbank energy or MFCC features. It is
    based on the paper: B Desplanques et al., ``ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification,''
    in Proc. INTERSPEECH, 2020.

    Attributes:
        _output_size (int): The output embedding dimension of the encoder.
        conv (nn.Conv1d): Convolutional layer for initial feature extraction.
        relu (nn.ReLU): Activation function applied after convolution.
        bn (nn.BatchNorm1d): Batch normalization layer for feature scaling.
        layer1 (block): First ECAPA block layer.
        layer2 (block): Second ECAPA block layer.
        layer3 (block): Third ECAPA block layer.
        layer4 (nn.Conv1d): Final convolutional layer for output embedding.
        mp3 (nn.MaxPool1d): Max pooling layer with a kernel size of 3.

    Args:
        input_size (int): Input feature dimension.
        block (str): Type of encoder block class to use (default: "EcapaBlock").
        model_scale (int): Scale value of the Res2Net architecture (default: 8).
        ndim (int): Dimensionality of the hidden representation (default: 1024).
        output_size (int): Output embedding dimension (default: 1536).
        **kwargs: Additional keyword arguments for further customization.

    Examples:
        >>> encoder = EcapaTdnnEncoder(input_size=80)
        >>> input_tensor = torch.randn(16, 100, 80)  # Batch size 16, 100 frames
        >>> output = encoder(input_tensor)
        >>> print(output.shape)  # Should output: torch.Size([16, 100, 1536])

    Raises:
        ValueError: If an unsupported block type is provided.

    Note:
        The encoder is designed for speaker verification tasks and utilizes
        a series of convolutional layers followed by ECAPA blocks to enhance
        the feature extraction process.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: str = "EcapaBlock",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        **kwargs,
    ):
        super().__init__()
        if block == "EcapaBlock":
            block: type = EcapaBlock
        else:
            raise ValueError(f"unsupported block, got: {block}")
        self._output_size = output_size

        self.conv = nn.Conv1d(input_size, ndim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ndim)

        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=2, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=3, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        """
                ECAPA-TDNN encoder. Extracts frame-level ECAPA-TDNN embeddings from mel-filterbank
        energy or MFCC features.

        Paper: B Desplanques et al., ``ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification,'' in Proc.
        INTERSPEECH, 2020.

        Attributes:
            input_size (int): Input feature dimension.
            block (str): Type of encoder block class to use.
            model_scale (int): Scale value of the Res2Net architecture.
            ndim (int): Dimensionality of the hidden representation.
            output_size (int): Output embedding dimension.

        Args:
            input_size: Input feature dimension.
            block: Type of encoder block class to use.
            model_scale: Scale value of the Res2Net architecture.
            ndim: Dimensionality of the hidden representation.
            output_size: Output embedding dimension.

        Examples:
            encoder = EcapaTdnnEncoder(input_size=80, output_size=1536)
            output = encoder(torch.randn(32, 100, 80))  # Example input tensor

        Note:
            Ensure that the input tensor has the shape (#batch, L, input_size).

        Raises:
            ValueError: If an unsupported block type is provided.
        """
        return self._output_size

    def forward(self, x: torch.Tensor):
        """
                Calculate forward propagation through the ECAPA-TDNN encoder.

        This method processes the input tensor through several convolutional and
        layer blocks to produce an output tensor representing the frame-level
        ECAPA-TDNN embeddings.

        Args:
            x (torch.Tensor): Input tensor with shape (#batch, L, input_size),
                              where L is the sequence length and input_size is the
                              dimension of the input features.

        Returns:
            torch.Tensor: Output tensor with shape (#batch, L, output_size),
                          where output_size is the dimension of the extracted
                          embeddings.

        Examples:
            >>> encoder = EcapaTdnnEncoder(input_size=80)
            >>> input_tensor = torch.randn(32, 100, 80)  # Example input
            >>> output_tensor = encoder.forward(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([32, 100, 1536])  # Example output shape
        """
        x = self.conv(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.bn(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        return x
