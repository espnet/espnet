import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class NullEncoder(AbsEncoder):
    """
    NullEncoder is a simple implementation of an encoder that performs no
operations on the input. It is a subclass of AbsEncoder and serves as a
placeholder encoder.

Attributes:
    output_dim (int): The output dimension of the encoder, which is fixed at 1.

Args:
    input (torch.Tensor): Mixed speech tensor of shape [Batch, sample].
    ilens (torch.Tensor): Tensor representing the lengths of the input 
        sequences, shape [Batch].
    fs (int, optional): Sampling rate in Hz. This parameter is not used in 
        this encoder.

Returns:
    tuple: A tuple containing:
        - torch.Tensor: The input tensor unchanged.
        - torch.Tensor: The input lengths unchanged.

Examples:
    >>> encoder = NullEncoder()
    >>> mixed_speech = torch.randn(5, 16000)  # Example mixed speech
    >>> input_lengths = torch.tensor([16000] * 5)  # Example input lengths
    >>> output, lengths = encoder.forward(mixed_speech, input_lengths)
    >>> assert torch.equal(output, mixed_speech)
    >>> assert torch.equal(lengths, input_lengths)

Note:
    This encoder is primarily used for testing or when no processing is 
    required.

Todo:
    - Consider extending functionality if needed for future use cases.
    """

    def __init__(self):
        super().__init__()

    @property
    def output_dim(self) -> int:
        """
        Null encoder.

    This encoder serves as a placeholder and does not perform any actual encoding.
    It simply returns the input as is.

    Attributes:
        output_dim (int): The dimensionality of the output, which is always 1.

    Methods:
        forward(input: torch.Tensor, ilens: torch.Tensor, fs: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Passes the input tensor and its lengths through without modification.

    Examples:
        encoder = NullEncoder()
        print(encoder.output_dim)  # Output: 1

        input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ilens_tensor = torch.tensor([3, 3])
        output, lengths = encoder.forward(input_tensor, ilens_tensor)
        print(output)  # Output: tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        print(lengths)  # Output: tensor([3, 3])
        """
        return 1

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """
        Forward pass of the NullEncoder.

    This method takes the mixed speech input and its corresponding lengths, 
    and returns them without any modifications. It is primarily used in 
    scenarios where an encoder is required, but no actual processing is 
    needed. This is useful for testing or as a placeholder in models.

    Args:
        input (torch.Tensor): Mixed speech tensor of shape [Batch, sample].
        ilens (torch.Tensor): Tensor containing the lengths of each input 
            sequence, shape [Batch].
        fs (int, optional): Sampling rate in Hz (Not used). Default is None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Returns the input tensor and the 
            input lengths tensor as a tuple.

    Examples:
        >>> encoder = NullEncoder()
        >>> mixed_speech = torch.randn(5, 16000)  # 5 samples, 16000 samples each
        >>> lengths = torch.tensor([16000, 16000, 16000, 16000, 16000])
        >>> output, output_lengths = encoder.forward(mixed_speech, lengths)
        >>> output.shape
        torch.Size([5, 16000])
        >>> output_lengths.shape
        torch.Size([5])

    Note:
        This encoder does not perform any transformations on the input data. 
        It simply returns the inputs as they are.
        """
        return input, ilens
