import torch

from espnet2.layers.label_aggregation import LabelAggregate


class LabelProcessor(torch.nn.Module):
    """
    LabelProcessor is a PyTorch module that aggregates labels for speaker
    diarization tasks.

    This class utilizes the LabelAggregate layer to perform label aggregation
    over specified window and hop lengths. It is designed to process input
    label tensors and their corresponding lengths, outputting aggregated
    labels along with their lengths.

    Attributes:
        label_aggregator (LabelAggregate): An instance of the LabelAggregate
        class responsible for aggregating labels based on specified parameters.

    Args:
        win_length (int): The length of the window for aggregation. Defaults to
            512.
        hop_length (int): The hop length for the sliding window. Defaults to
            128.
        center (bool): Whether to center the window. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The aggregated label tensor of shape
            (Batch, Frames, Label_dim).
            - olens (torch.Tensor): The lengths of the output sequences of
            shape (Batch).

    Examples:
        >>> label_processor = LabelProcessor(win_length=256, hop_length=64)
        >>> input_tensor = torch.randn(10, 1000, 20)  # Batch of 10, 1000 samples, 20 labels
        >>> ilens_tensor = torch.tensor([1000] * 10)  # Lengths for each batch
        >>> output, olens = label_processor(input_tensor, ilens_tensor)
        >>> print(output.shape)  # Should output (10, Frames, 20)
        >>> print(olens.shape)   # Should output (10,)

    Note:
        This module is specifically designed for use in speaker diarization
        systems, and the choice of window and hop lengths can significantly
        affect the performance of the aggregation.

    Todo:
        - Implement additional methods for enhanced label processing if needed.
    """

    def __init__(
        self, win_length: int = 512, hop_length: int = 128, center: bool = True
    ):
        super().__init__()
        self.label_aggregator = LabelAggregate(win_length, hop_length, center)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
            Forward pass for the LabelProcessor module.

        This method processes the input tensor containing labels and their respective
        lengths. It utilizes the label aggregator to produce an output tensor that
        represents the aggregated labels over the specified window and hop lengths.

        Args:
            input (torch.Tensor): A tensor of shape (Batch, Nsamples, Label_dim)
                representing the input labels for each sample in the batch.
            ilens (torch.Tensor): A tensor of shape (Batch) containing the lengths
                of each input sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): A tensor of shape (Batch, Frames, Label_dim)
                  representing the aggregated output labels.
                - olens (torch.Tensor): A tensor of shape (Batch) containing the
                  lengths of the output sequences.

        Examples:
            >>> label_processor = LabelProcessor(win_length=512, hop_length=128)
            >>> input_tensor = torch.randn(2, 1000, 10)  # Batch of 2, 1000 samples, 10 labels
            >>> ilens_tensor = torch.tensor([1000, 800])  # Lengths of each input
            >>> output, olens = label_processor(input_tensor, ilens_tensor)
            >>> print(output.shape)  # Output shape will depend on aggregation
            >>> print(olens)  # Output lengths for each batch

        Note:
            Ensure that the input tensor and ilens are correctly formatted
            according to the expected shapes.

        Raises:
            ValueError: If the input tensor dimensions do not match the expected
            shape or if ilens contains invalid lengths.
        """

        output, olens = self.label_aggregator(input, ilens)

        return output, olens
