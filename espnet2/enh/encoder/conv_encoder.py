import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class ConvEncoder(AbsEncoder):
    """
    Convolutional encoder for speech enhancement and separation.

    This class implements a convolutional encoder using 1D convolutional layers.
    It is designed to process mixed speech inputs for tasks such as speech
    enhancement and separation.

    Attributes:
        output_dim (int): The dimension of the output features after encoding.

    Args:
        channel (int): The number of output channels in the convolutional layer.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution.

    Returns:
        feature (torch.Tensor): Mixed feature after encoder
            with shape [Batch, flens, channel].
        flens (torch.Tensor): Output lengths after encoding.

    Raises:
        AssertionError: If the input tensor does not have the correct dimensions.

    Examples:
        >>> import torch
        >>> input_audio = torch.randn((2, 100))
        >>> ilens = torch.LongTensor([100, 98])
        >>> encoder = ConvEncoder(kernel_size=32, stride=10, channel=16)
        >>> frames, flens = encoder(input_audio, ilens)

        >>> splited = encoder.streaming_frame(input_audio)
        >>> sframes = [encoder.forward_streaming(s) for s in splited]
        >>> sframes = torch.cat(sframes, dim=1)
        >>> torch.testing.assert_allclose(sframes, frames)

    Note:
        The `fs` argument in the `forward` method is not used and can be set to
        `None`.

    Todo:
        - Implement additional functionality for multi-channel inputs.
    """

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            1, channel, kernel_size=kernel_size, stride=stride, bias=False
        )
        self.stride = stride
        self.kernel_size = kernel_size

        self._output_dim = channel

    @property
    def output_dim(self) -> int:
        """
        Get the output dimension of the ConvEncoder.

        This property returns the number of output channels
        produced by the convolutional layer in the encoder.

        Returns:
            int: The output dimension, which corresponds to the
            number of channels specified during initialization.

        Examples:
            encoder = ConvEncoder(kernel_size=3, stride=1, channel=16)
            print(encoder.output_dim)  # Output: 16

        Note:
            This property is primarily used to obtain the output
            dimension after the convolutional processing of the input.
        """
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """
            Forward pass of the convolutional encoder.

        This method processes the input mixed speech signal through a 1D convolutional
        layer, applying ReLU activation and returning the encoded features along with
        the calculated lengths of the output features.

        Args:
            input (torch.Tensor): Mixed speech input of shape [Batch, sample].
            ilens (torch.Tensor): Lengths of the input sequences of shape [Batch].
            fs (int, optional): Sampling rate in Hz (not used in current implementation).

        Returns:
            feature (torch.Tensor): Encoded mixed feature of shape [Batch, flens, channel],
                where flens is the calculated output length after processing.
            flens (torch.Tensor): Lengths of the output features of shape [Batch].

        Raises:
            AssertionError: If the input tensor does not have 2 dimensions.

        Examples:
            >>> input_audio = torch.randn((2, 100))
            >>> ilens = torch.LongTensor([100, 98])
            >>> encoder = ConvEncoder(kernel_size=32, stride=10, channel=16)
            >>> feature, flens = encoder(input_audio, ilens)
            >>> print(feature.shape)  # Output shape: [Batch, flens, channel]
            >>> print(flens)  # Output lengths for each batch

        Note:
            The input tensor is expected to be a single-channel tensor.
        """
        assert input.dim() == 2, "Currently only support single channel input"

        input = torch.unsqueeze(input, 1)

        feature = self.conv1d(input)
        feature = torch.nn.functional.relu(feature)
        feature = feature.transpose(1, 2)

        flens = (
            torch.div(ilens - self.kernel_size, self.stride, rounding_mode="trunc") + 1
        )

        return feature, flens

    def forward_streaming(self, input: torch.Tensor):
        """
            Perform the forward pass for streaming input.

        This method is designed to handle streaming audio inputs by utilizing the
        `forward` method. It takes a tensor representing audio data and processes
        it through the convolutional encoder to produce the output features.

        Args:
            input (torch.Tensor): Input tensor representing mixed speech with shape
                [Batch, sample].

        Returns:
            output (torch.Tensor): Output tensor containing mixed features after
                encoding with shape [Batch, flens, channel].

        Examples:
            >>> encoder = ConvEncoder(kernel_size=32, stride=10, channel=16)
            >>> input_audio = torch.randn((2, 100))
            >>> output = encoder.forward_streaming(input_audio)
            >>> print(output.shape)
            torch.Size([2, 7, 16])  # Example output shape based on kernel and stride

        Note:
            The `ilens` parameter is not utilized in this method, and it defaults
            to a fixed behavior from the `forward` method.
        """
        output, _ = self.forward(input, 0)
        return output

    def streaming_frame(self, audio: torch.Tensor):
        """
        Stream frame.

        It splits the continuous audio into frame-level audio chunks in the
        streaming simulation. This function takes the entire long audio as input
        for a streaming simulation. You may refer to this function to manage your
        streaming input buffer in a real streaming application.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (B, T), where B is
            the batch size and T is the total length of the audio.

        Returns:
            List[torch.Tensor]: A list of chunked audio tensors, each of shape
            (B, frame_size), where frame_size is determined by the kernel size.

        Examples:
            >>> encoder = ConvEncoder(kernel_size=32, stride=10, channel=16)
            >>> audio_input = torch.randn((2, 100))
            >>> frames = encoder.streaming_frame(audio_input)
            >>> for frame in frames:
            ...     print(frame.shape)
            torch.Size([2, 32])
            torch.Size([2, 32])
            ...
        """
        batch_size, audio_len = audio.shape

        hop_size = self.stride
        frame_size = self.kernel_size

        audio = [
            audio[:, i * hop_size : i * hop_size + frame_size]
            for i in range((audio_len - frame_size) // hop_size + 1)
        ]

        return audio


if __name__ == "__main__":
    input_audio = torch.randn((2, 100))
    ilens = torch.LongTensor([100, 98])

    nfft = 32
    win_length = 28
    hop = 10

    encoder = ConvEncoder(kernel_size=nfft, stride=hop, channel=16)
    frames, flens = encoder(input_audio, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]

    sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_allclose(sframes, frames)
