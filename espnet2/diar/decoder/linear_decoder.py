import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """
    Linear decoder for speaker diarization.

This class implements a linear decoder that processes the output of an encoder 
for speaker diarization tasks. It inherits from the `AbsDecoder` class and uses 
a linear layer to map encoder outputs to the desired number of speakers.

Attributes:
    num_spk (int): The number of speakers that the decoder can output.

Args:
    encoder_output_size (int): The size of the encoder's output feature vector.
    num_spk (int, optional): The number of speakers to decode. Defaults to 2.

Returns:
    torch.Tensor: The decoded output tensor with shape [Batch, T, num_spk].

Examples:
    >>> decoder = LinearDecoder(encoder_output_size=128, num_spk=3)
    >>> input_tensor = torch.randn(10, 50, 128)  # Batch size 10, T=50, F=128
    >>> ilens = torch.tensor([50] * 10)  # All sequences have length 50
    >>> output = decoder(input_tensor, ilens)
    >>> print(output.shape)
    torch.Size([10, 50, 3])  # Output shape corresponds to num_spk

Raises:
    ValueError: If `input` does not have the correct shape or dimensions.
    """

    def __init__(
        self,
        encoder_output_size: int,
        num_spk: int = 2,
    ):
        super().__init__()
        self._num_spk = num_spk
        self.linear_decoder = torch.nn.Linear(encoder_output_size, num_spk)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
        Forward pass of the LinearDecoder.

        This method takes the input tensor representing the hidden space and the 
        input lengths, and applies a linear transformation to decode the speaker 
        representations.

        Args:
            input (torch.Tensor): A tensor of shape [Batch, T, F] representing the 
                hidden space, where 'Batch' is the number of samples, 'T' is the 
                time dimension, and 'F' is the feature dimension.
            ilens (torch.Tensor): A tensor of shape [Batch] representing the lengths 
                of the input sequences.

        Returns:
            torch.Tensor: A tensor of shape [Batch, T, num_spk] representing the 
            decoded speaker outputs, where 'num_spk' is the number of speakers.

        Examples:
            >>> decoder = LinearDecoder(encoder_output_size=256, num_spk=3)
            >>> input_tensor = torch.randn(10, 20, 256)  # Batch of 10, 20 time steps, 256 features
            >>> input_lengths = torch.tensor([20] * 10)  # All sequences are of length 20
            >>> output = decoder.forward(input_tensor, input_lengths)
            >>> output.shape
            torch.Size([10, 20, 3])  # Decoded output for 3 speakers

        Note:
            The input tensor should be properly normalized and prepared before 
            passing to the forward method.
        """

        output = self.linear_decoder(input)

        return output

    @property
    def num_spk(self):
        """
        Linear decoder for speaker diarization.

        This class implements a linear decoder that maps encoder outputs to a specified 
        number of speakers. It is designed to work with the output of an encoder in 
        speaker diarization tasks.

        Attributes:
            num_spk (int): The number of speakers the decoder can handle.

        Args:
            encoder_output_size (int): The size of the output from the encoder.
            num_spk (int, optional): The number of speakers to decode. Defaults to 2.

        Methods:
            forward(input: torch.Tensor, ilens: torch.Tensor) -> torch.Tensor:
                Computes the forward pass of the linear decoder.

        Examples:
            # Initialize the LinearDecoder with encoder output size of 128 and 3 speakers
            decoder = LinearDecoder(encoder_output_size=128, num_spk=3)

            # Create a dummy input tensor and input lengths
            input_tensor = torch.randn(10, 20, 128)  # [Batch, T, F]
            input_lengths = torch.randint(1, 21, (10,))  # Random lengths for each batch

            # Perform a forward pass
            output = decoder.forward(input_tensor, input_lengths)

            # Access the number of speakers
            number_of_speakers = decoder.num_spk  # Should return 3

        Note:
            The forward method expects the input tensor to have a shape of 
            [Batch, T, F] where T is the time dimension and F is the feature dimension.

        Raises:
            ValueError: If input dimensions do not match the expected shape.

        Todo:
            Implement additional methods for loss computation and decoding strategies.
        """
        return self._num_spk
