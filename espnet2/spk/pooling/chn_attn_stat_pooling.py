import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling


class ChnAttnStatPooling(AbsPooling):
    """
        Aggregates frame-level features to a single utterance-level feature.

    This pooling method is proposed in B. Desplanques et al., "ECAPA-TDNN:
    Emphasized Channel Attention, Propagation and Aggregation in TDNN Based
    Speaker Verification".

    Attributes:
        _output_size (int): The output dimensionality, which is double the
            input size.

    Args:
        input_size (int): Dimensionality of the input frame-level embeddings.
            This is determined by the encoder hyperparameter. The output
            dimensionality will be double the input_size.

    Returns:
        torch.Tensor: The pooled utterance-level feature.

    Raises:
        ValueError: If `task_tokens` is not None, as ChannelAttentiveStatisticsPooling
            is not adequate for task tokens.

    Examples:
        >>> pooling_layer = ChnAttnStatPooling(input_size=1536)
        >>> frame_level_features = torch.randn(10, 1536, 20)  # (batch_size, input_size, time_steps)
        >>> pooled_features = pooling_layer(frame_level_features)
        >>> print(pooled_features.shape)  # Output shape will be (10, 3072)
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self._output_size = input_size * 2

    def output_size(self):
        """
                Aggregates frame-level features to a single utterance-level feature.

        Proposed in B.Desplanques et al., "ECAPA-TDNN: Emphasized Channel
        Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

        Attributes:
            attention (nn.Sequential): A sequential container that applies a series of
                layers including convolution, activation, batch normalization, and
                softmax to compute attention weights.

        Args:
            input_size (int): Dimensionality of the input frame-level embeddings.
                Determined by encoder hyperparameter. For this pooling layer, the
                output dimensionality will be double the input_size.

        Returns:
            int: The output size of the pooling layer, which is double the input size.

        Examples:
            >>> pooling_layer = ChnAttnStatPooling(input_size=512)
            >>> output_size = pooling_layer.output_size()
            >>> print(output_size)
            1024

        Raises:
            ValueError: If task_tokens is not None during the forward pass, indicating
                that ChannelAttentiveStatisticsPooling is not adequate for task tokens.
        """
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        """
            Performs the forward pass of the ChnAttnStatPooling layer, aggregating
        frame-level features into a single utterance-level feature representation.

        This method computes a weighted combination of the input features and
        statistical summaries (mean and standard deviation) to produce a compact
        representation. It uses channel attention to emphasize important features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size, time).
            task_tokens (torch.Tensor, optional): An optional tensor for task-specific
                tokens. If provided, a ValueError will be raised, as this pooling
                method does not support task tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size), where
            output_size is double the input_size.

        Raises:
            ValueError: If task_tokens is provided, indicating that this pooling
            method is not suitable for task tokens.

        Examples:
            >>> pooling_layer = ChnAttnStatPooling(input_size=1536)
            >>> input_tensor = torch.randn(10, 1536, 100)  # (batch_size, input_size, time)
            >>> output = pooling_layer.forward(input_tensor)
            >>> print(output.shape)
            torch.Size([10, 3072])  # (batch_size, output_size)

        Note:
            The output size of this pooling layer is always double the input size,
            which is achieved by concatenating the mean and standard deviation of the
            input features.

        Todo:
            Consider extending the functionality to support task tokens if necessary.
        """
        if task_tokens is not None:
            raise ValueError(
                "ChannelAttentiveStatisticsPooling is not adequate for task_tokens"
            )
        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)
                ).repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x
