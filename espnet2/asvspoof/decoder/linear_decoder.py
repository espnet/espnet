from typing import Optional

import torch

from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """
    Linear decoder for speaker diarization.

    This class implements a linear decoder used in the context of speaker 
    diarization. It is responsible for transforming the encoder's output into 
    a suitable representation for further processing or classification.

    Attributes:
        encoder_output_size (int): The size of the output from the encoder.

    Args:
        encoder_output_size (int): Size of the encoder's output dimension.

    Methods:
        forward(input: torch.Tensor, ilens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            Processes the input tensor and applies a linear projection.

    Examples:
        >>> decoder = LinearDecoder(encoder_output_size=256)
        >>> input_tensor = torch.randn(10, 100, 256)  # Batch of 10
        >>> input_lengths = torch.tensor([100] * 10)  # All sequences are of length 100
        >>> output = decoder.forward(input_tensor, input_lengths)
        >>> print(output.shape)  # Shape depends on the implementation details

    Note:
        This class currently contains placeholder TODOs for implementation.
        The forward method is expected to compute the mean over the time 
        dimension and apply a linear projection layer.

    Raises:
        ValueError: If the input tensor has an incorrect shape.
    
    Todo:
        - Implement the linear projection layer.
        - Compute mean over the time-domain (dimension 1).
        - Update the return value to return the processed tensor.
    """

    def __init__(
        self,
        encoder_output_size: int,
    ):
        super().__init__()
        # TODO(checkpoint3): initialize a linear projection layer

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor]):
        """
        Perform the forward pass of the LinearDecoder.

        This method takes the encoder output and computes the linear projection 
        for speaker diarization. It processes the input tensor and utilizes 
        the specified input lengths to ensure proper handling of variable-length 
        sequences.

        Args:
            input (torch.Tensor): A tensor representing the hidden space with 
                shape [Batch, T, F], where Batch is the number of samples, 
                T is the sequence length, and F is the feature dimension.
            ilens (Optional[torch.Tensor]): A tensor containing the lengths 
                of the input sequences with shape [Batch]. This is used to 
                handle variable-length inputs properly.

        Returns:
            torch.Tensor: The output of the linear projection layer, which 
            will have shape [Batch, F_out], where F_out is the size of the 
            output features after applying the linear projection.

        Raises:
            ValueError: If the input tensor shape does not match the expected 
            dimensions or if ilens is not compatible with input.

        Examples:
            >>> decoder = LinearDecoder(encoder_output_size=128)
            >>> input_tensor = torch.randn(32, 10, 128)  # Batch of 32, T=10, F=128
            >>> ilens = torch.tensor([10] * 32)  # All sequences are of length 10
            >>> output = decoder.forward(input_tensor, ilens)
            >>> print(output.shape)  # Should print: torch.Size([32, F_out])

        Note:
            The actual implementation of the forward pass is yet to be 
            completed. This includes computing the mean over the time 
            dimension and applying the projection layer.
        """
        # TODO(checkpoint3): compute mean over time-domain (dimension 1)

        # TODO(checkpoint3): apply the projection layer

        # TODO(checkpoint3): change the return value
        return None
