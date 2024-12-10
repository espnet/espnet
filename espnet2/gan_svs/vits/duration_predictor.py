# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor modules in VISinger.
"""

import torch

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    """
        DurationPredictor is a module that predicts durations for audio signals in the
    VISinger framework.

    This class utilizes convolutional layers and normalization to process input
    features and predict the duration of each time step in the sequence. It can
    optionally take global conditioning inputs, which can be used for multi-singer
    applications.

    Attributes:
        in_channels (int): Number of input channels.
        filter_channels (int): Number of filter channels for convolutional layers.
        kernel_size (int): Size of the convolutional kernel.
        dropout_rate (float): Rate at which to drop units during training.

    Args:
        channels (int): Number of input channels.
        filter_channels (int): Number of filter channels.
        kernel_size (int): Size of the convolutional kernel.
        dropout_rate (float): Dropout rate.
        global_channels (int, optional): Number of global conditioning channels.

    Methods:
        forward(x, x_mask, g=None):
            Forward pass through the duration predictor module.

    Returns:
        Tensor: Predicted duration tensor of shape (B, 2, T), where B is the batch
        size and T is the length of the input sequence.

    Examples:
        >>> predictor = DurationPredictor(128, 256, 3, 0.1)
        >>> x = torch.randn(32, 128, 100)  # Example input tensor
        >>> x_mask = torch.ones(32, 1, 100)  # Example mask tensor
        >>> output = predictor(x, x_mask)
        >>> print(output.shape)  # Should output: torch.Size([32, 2, 100])

    Note:
        This module is designed for use within the ESPnet framework, specifically
        for voice synthesis tasks. The input tensor should be appropriately shaped
        and masked to avoid influencing the output predictions with padded values.
    """

    def __init__(
        self,
        channels,
        filter_channels,
        kernel_size,
        dropout_rate,
        global_channels=0,
    ):
        """Initialize duration predictor module.

        Args:
            channels (int): Number of input channels.
            filter_channels (int): Number of filter channels.
            kernel_size (int): Size of the convolutional kernel.
            dropout_rate (float): Dropout rate.
            global_channels (int, optional): Number of global conditioning channels.

        """
        super().__init__()

        self.in_channels = channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.drop = torch.nn.Dropout(dropout_rate)
        self.conv_1 = torch.nn.Conv1d(
            channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels, dim=1)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels, dim=1)
        self.conv_3 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_3 = LayerNorm(filter_channels, dim=1)
        self.proj = torch.nn.Conv1d(filter_channels, 2, 1)

        if global_channels > 0:
            self.conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        """
            Forward pass through the duration predictor module.

        This method processes the input tensor through a series of convolutional
        layers, applying layer normalization and dropout, and optionally includes
        global conditioning. It produces a predicted duration tensor that can be
        used for various downstream tasks in duration prediction.

        Args:
            x (Tensor): Input tensor of shape (B, in_channels, T), where B is the
                batch size, in_channels is the number of input channels, and T is
                the sequence length.
            x_mask (Tensor): Mask tensor of shape (B, 1, T) used to mask out
                padding or irrelevant parts of the input during processing.
            g (Tensor, optional): Global condition tensor of shape (B, global_channels, 1)
                used for multi-singer scenarios. If provided, this tensor is added
                to the input tensor after being processed through a convolutional
                layer. Defaults to None.

        Returns:
            Tensor: Predicted duration tensor of shape (B, 2, T), where the second
            dimension corresponds to the predicted durations.

        Examples:
            >>> duration_predictor = DurationPredictor(channels=256,
            ...                                          filter_channels=512,
            ...                                          kernel_size=3,
            ...                                          dropout_rate=0.1)
            >>> x = torch.randn(10, 256, 100)  # Batch of 10, 256 channels, length 100
            >>> x_mask = torch.ones(10, 1, 100)  # No masking
            >>> output = duration_predictor(x, x_mask)
            >>> print(output.shape)
            torch.Size([10, 2, 100])  # Expected output shape
        """

        # multi-singer
        if g is not None:
            g = torch.detach(g)
            x = x + self.conv(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask
