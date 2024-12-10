from typing import Optional, Tuple

import torch


class AbsSpecAug(torch.nn.Module):
    """
    Abstract base class for spectrogram augmentation in speech processing.

    This class serves as a blueprint for implementing various spectrogram
    augmentation techniques. The augmentation process is typically part of
    a speech recognition pipeline that includes frontend processing,
    spectrogram augmentation, normalization, encoding, and decoding.

    Attributes:
        None

    Args:
        None

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            A tuple containing the augmented spectrogram tensor and
            optionally the lengths of the input sequences.

    Yields:
        None

    Raises:
        NotImplementedError: If the forward method is not implemented by
        the subclass.

    Examples:
        To implement a specific spectrogram augmentation, subclass
        AbsSpecAug and define the forward method:

        ```python
        class MySpecAug(AbsSpecAug):
            def forward(self, x, x_lengths=None):
                # Implement the augmentation logic here
                return augmented_x, x_lengths
        ```

    Note:
        This class is intended to be subclassed, and the forward method
        must be overridden to provide specific augmentation behavior.

    Todo:
        Implement additional methods or properties as needed for specific
        augmentation strategies.
    """

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs the forward pass of the spectrogram augmentation.

        This method takes an input tensor representing the spectrogram and
        optionally its lengths. It processes the input through the augmentation
        pipeline, returning the augmented spectrogram and the updated lengths.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_channels,
                time_steps) representing the input spectrogram.
            x_lengths (torch.Tensor, optional): A tensor of shape (batch_size,)
                containing the lengths of each input in the batch. If None,
                lengths are assumed to be the maximum length of the inputs.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple where the first
            element is the augmented spectrogram tensor of shape
            (batch_size, num_channels, time_steps) and the second element is
            the updated lengths tensor, or None if lengths were not provided.

        Raises:
            NotImplementedError: This method should be implemented in a
            subclass of AbsSpecAug.

        Examples:
            >>> model = MySpecAug()  # MySpecAug is a subclass of AbsSpecAug
            >>> input_tensor = torch.randn(2, 1, 100)  # Batch of 2, 1 channel, 100 time steps
            >>> lengths = torch.tensor([100, 90])  # Lengths of the inputs
            >>> output, updated_lengths = model.forward(input_tensor, lengths)
            >>> print(output.shape)  # Should print: torch.Size([2, 1, 100])
            >>> print(updated_lengths)  # Should print: tensor([100, 90]) or modified lengths

        Note:
            This method is intended to be overridden in subclasses to provide
            specific augmentation logic.
        """
        raise NotImplementedError
