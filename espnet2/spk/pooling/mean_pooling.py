import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """
    Average frame-level features to a single utterance-level feature.

    This class implements mean pooling for aggregating frame-level
    embeddings into a single vector that represents the entire utterance.

    Attributes:
        input_size (int): Dimensionality of the input frame-level embeddings,
            determined by the encoder hyperparameter.

    Args:
        input_size (int): Dimensionality of the input frame-level embeddings.
            Defaults to 1536.

    Raises:
        ValueError: If `task_tokens` is provided, as MeanPooling is not
            suitable for task-specific tokens.

    Examples:
        >>> import torch
        >>> mean_pooling = MeanPooling(input_size=128)
        >>> frame_embeddings = torch.rand(10, 128)  # 10 frames, 128 features
        >>> utterance_embedding = mean_pooling.forward(frame_embeddings)
        >>> print(utterance_embedding.shape)
        torch.Size([128])  # Resulting shape after mean pooling

    Note:
        The input tensor `x` should have shape (N, C, L), where N is the
        batch size, C is the number of channels (features), and L is the
        number of frames. The mean will be computed over the last dimension (L).
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size

    def output_size(self):
        """
            Average frame-level features to a single utterance-level feature.

        This class implements mean pooling over input embeddings to reduce the
        dimensionality of the data from frame-level to utterance-level. The
        output size is determined by the dimensionality of the input frame-level
        embeddings, which is specified during initialization.

        Attributes:
            output_size (int): The dimensionality of the output features.

        Args:
            input_size (int): Dimensionality of the input frame-level embeddings.
                This is determined by the encoder hyperparameter and defaults to
                1536.

        Returns:
            int: The output size of the pooled feature.

        Raises:
            ValueError: If `task_tokens` is not None during the forward pass, as
                MeanPooling is not designed to handle task-specific tokens.

        Examples:
            >>> mean_pooling = MeanPooling(input_size=512)
            >>> pooled_output = mean_pooling.forward(torch.randn(10, 512))
            >>> print(pooled_output.shape)
            torch.Size([10])  # Output shape after pooling

        Note:
            This pooling method averages over the last dimension of the input tensor.

        Todo:
            Consider implementing support for task_tokens in future versions.
        """
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        """
            Computes the mean of frame-level features to produce a single
        utterance-level feature.

        This method takes a tensor of frame-level embeddings and returns
        the mean value across the specified dimension. It is primarily
        used in scenarios where the average representation of a sequence
        is required.

        Args:
            x (torch.Tensor): A tensor containing frame-level embeddings
                of shape (batch_size, num_frames, input_size).
            task_tokens (torch.Tensor, optional): A tensor for task-specific
                tokens. If provided, a ValueError is raised as MeanPooling
                is not designed to handle task tokens.

        Returns:
            torch.Tensor: A tensor containing the mean of the input
            embeddings, with shape (batch_size, input_size).

        Raises:
            ValueError: If task_tokens is provided, as MeanPooling does not
            support task-specific tokens.

        Examples:
            >>> import torch
            >>> mean_pooling = MeanPooling(input_size=1536)
            >>> frame_embeddings = torch.rand(10, 5, 1536)  # 10 samples, 5 frames
            >>> output = mean_pooling.forward(frame_embeddings)
            >>> print(output.shape)  # Output: torch.Size([10, 1536])
        """
        if task_tokens is not None:
            raise ValueError("MeanPooling is not adequate for task_tokens")
        x = torch.mean(x, dim=-1)

        return x
