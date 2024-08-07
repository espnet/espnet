from typing import Optional, Tuple

import torch


class AbsSpecAug(torch.nn.Module):
    """
        Abstract base class for spectrogram augmentation.

    This class defines the interface for spectrogram augmentation modules in a
    neural network pipeline. Spectrogram augmentation is typically applied after
    the frontend processing and before normalization, encoding, and decoding steps.

    Attributes:
        None

    Note:
        Subclasses must implement the `forward` method to define specific
        augmentation techniques.

    Examples:
        >>> class MySpecAug(AbsSpecAug):
        ...     def forward(self, x, x_lengths=None):
        ...         # Implement augmentation logic here
        ...         return augmented_x, x_lengths
        ...
        >>> spec_aug = MySpecAug()
        >>> augmented_spec, lengths = spec_aug(input_spec, input_lengths)
    """

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
                Apply spectrogram augmentation to the input tensor.

        This method should be implemented by subclasses to perform specific
        augmentation techniques on the input spectrogram.

        Args:
            x (torch.Tensor): Input spectrogram tensor.
            x_lengths (torch.Tensor, optional): Tensor containing the lengths of each
                sequence in the batch. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - The augmented spectrogram tensor.
                - The updated lengths tensor (if x_lengths was provided, otherwise None).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Examples:
            >>> class MySpecAug(AbsSpecAug):
            ...     def forward(self, x, x_lengths=None):
            ...         # Apply augmentation (e.g., time warping, frequency masking)
            ...         augmented_x = some_augmentation_function(x)
            ...         return augmented_x, x_lengths
            ...
            >>> spec_aug = MySpecAug()
            >>> input_spec = torch.randn(32, 80, 100)  # (batch_size, num_mels, time_steps)
            >>> input_lengths = torch.full((32,), 100)
            >>> augmented_spec, updated_lengths = spec_aug(input_spec, input_lengths)
        """
        raise NotImplementedError
