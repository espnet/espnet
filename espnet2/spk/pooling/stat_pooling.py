import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class StatsPooling(AbsPooling):
    """
    Aggregates frame-level features to single utterance-level feature.

    This class implements the statistical pooling method as proposed in D. Snyder
    et al., "X-vectors: Robust dnn embeddings for speaker recognition". It takes
    frame-level embeddings and computes the mean and standard deviation across
    the time dimension, concatenating them to form a single utterance-level
    feature.

    Attributes:
        _output_size (int): The dimensionality of the output feature, which is
            double the input size.

    Args:
        input_size (int): Dimensionality of the input frame-level embeddings.
            This is determined by the encoder hyperparameter. The output
            dimensionality will be double the input_size.

    Raises:
        ValueError: If `task_tokens` is provided during the forward pass,
            as this pooling method is not suitable for task-specific tokens.

    Examples:
        >>> pooling_layer = StatsPooling(input_size=512)
        >>> frame_level_embeddings = torch.randn(10, 512)  # (time, features)
        >>> output = pooling_layer.forward(frame_level_embeddings)
        >>> print(output.shape)  # Output will be (10, 1024), where 1024 = 2 * 512
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size * 2

    def output_size(self):
        """
                Aggregates frame-level features to single utterance-level feature.

        Proposed in D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker
        recognition".

        Attributes:
            input_size (int): Dimensionality of the input frame-level embeddings.
                Determined by encoder hyperparameter.
                For this pooling layer, the output dimensionality will be double of
                the input_size.

        Args:
            input_size: Dimensionality of the input frame-level embeddings.

        Returns:
            int: The output size, which is double the input_size.

        Examples:
            >>> pooling = StatsPooling(input_size=128)
            >>> pooling.output_size()
            256

        Note:
            The output size is fixed at double the input size for this pooling layer.
        """
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        """
            Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames,
                input_size), where num_frames is the number of frames and
                input_size is the dimensionality of the input embeddings.
            task_tokens (torch.Tensor, optional): A tensor of task tokens. If
                provided, a ValueError will be raised since this pooling method
                does not support task tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size),
                where output_size is double the input_size. The output contains
                the mean and standard deviation of the input embeddings across
                the frame dimension.

        Raises:
            ValueError: If task_tokens is not None, as this pooling method does
                not support task-specific tokens.

        Examples:
            >>> pooling = StatsPooling(input_size=1536)
            >>> input_tensor = torch.rand(10, 5, 1536)  # Batch of 10, 5 frames
            >>> output = pooling.forward(input_tensor)
            >>> output.shape
            torch.Size([10, 3072])  # Output size is 2 * input_size

        Note:
            This method aggregates frame-level features into a single
            utterance-level feature by calculating the mean and standard
            deviation across the frame dimension.
        """
        if task_tokens is not None:
            raise ValueError("StatisticsPooling is not adequate for task_tokens")
        mu = torch.mean(x, dim=-1)
        st = torch.std(x, dim=-1)

        x = torch.cat((mu, st), dim=1)

        return x
