import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class NullDecoder(AbsDecoder):
    """
        NullDecoder is a class that serves as a placeholder decoder, returning the
    input arguments unchanged. It inherits from the abstract base class
    AbsDecoder.

    Attributes:
        None

    Args:
        None

    Methods:
        forward(input: torch.Tensor, ilens: torch.Tensor, fs: int = None) ->
        Tuple[torch.Tensor, torch.Tensor]:
            Processes the input waveform and lengths, returning them as-is.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The input waveform and its lengths
        unchanged.

    Examples:
        >>> decoder = NullDecoder()
        >>> input_waveform = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> input_lengths = torch.tensor([3, 3])
        >>> output_waveform, output_lengths = decoder.forward(input_waveform,
        ... input_lengths)
        >>> output_waveform
        tensor([[0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]])
        >>> output_lengths
        tensor([3, 3])

    Note:
        This decoder does not perform any processing on the input data and is
        primarily used for testing or as a fallback option.

    Todo:
        - Consider adding functionality if needed in future versions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """
            Forward pass for the NullDecoder. This method returns the input waveform
        and its corresponding lengths without any modifications. The input should
        be the waveform already provided.

        Args:
            input (torch.Tensor): The waveform input of shape [Batch, sample].
            ilens (torch.Tensor): The lengths of the input waveforms of shape [Batch].
            fs (int, optional): The sampling rate in Hz. This argument is not used
                in the computation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input (torch.Tensor): The original waveform input.
                - ilens (torch.Tensor): The original input lengths.

        Examples:
            >>> decoder = NullDecoder()
            >>> input_waveform = torch.randn(2, 16000)  # Example batch of waveforms
            >>> input_lengths = torch.tensor([16000, 16000])  # Lengths of waveforms
            >>> output_waveform, output_lengths = decoder.forward(input_waveform, input_lengths)
            >>> assert torch.equal(output_waveform, input_waveform)
            >>> assert torch.equal(output_lengths, input_lengths)

        Note:
            This decoder does not perform any decoding operation and simply
            returns the input as is. It can be useful for testing or as a
            placeholder in a processing pipeline.
        """
        return input, ilens
