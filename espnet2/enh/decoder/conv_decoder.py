import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class ConvDecoder(AbsDecoder):
    """
        ConvDecoder is a transposed convolutional decoder for speech enhancement and
    separation.

    This class extends the AbsDecoder and provides functionality to decode the
    output of a convolutional encoder into a time-domain waveform. The decoder
    utilizes a transposed convolutional layer to perform the decoding operation,
    which is crucial in tasks such as speech enhancement and separation.

    Attributes:
        convtrans1d (torch.nn.ConvTranspose1d): The transposed convolutional layer
            used for decoding.
        kernel_size (int): The size of the kernel used in the transposed
            convolution.
        stride (int): The stride of the transposed convolution.

    Args:
        channel (int): The number of input channels for the transposed convolution.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride for the transposed convolution.

    Methods:
        forward(input: torch.Tensor, ilens: torch.Tensor, fs: int = None) ->
            Tuple[torch.Tensor, torch.Tensor]:
                Performs the forward pass, decoding the input tensor into a
                waveform.

        forward_streaming(input_frame: torch.Tensor) -> torch.Tensor:
            Performs streaming forward pass for the input frame.

        streaming_merge(chunks: torch.Tensor, ilens: torch.Tensor = None) ->
            torch.Tensor:
                Merges frame-level processed audio chunks in a streaming
                simulation.

    Raises:
        ValueError: If input tensor dimensions do not match expected shapes.

    Examples:
        >>> import torch
        >>> input_audio = torch.randn((1, 100))
        >>> ilens = torch.LongTensor([100])
        >>> kernel_size = 32
        >>> stride = 16
        >>> decoder = ConvDecoder(kernel_size=kernel_size, stride=stride, channel=16)
        >>> wav, ilens = decoder(input_audio, ilens)

    Note:
        The `fs` parameter in the forward method is currently not utilized.

    Todo:
        Implement additional error handling for input dimensions in the forward
        method.
    """

    def __init__(
        self,
        channel: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        self.convtrans1d = torch.nn.ConvTranspose1d(
            channel, 1, kernel_size, bias=False, stride=stride
        )

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, fs: int = None):
        """Forward.

        Args:
            input (torch.Tensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            fs (int): sampling rate in Hz (Not used)
        """
        input = input.transpose(1, 2)
        batch_size = input.shape[0]
        wav = self.convtrans1d(input, output_size=(batch_size, 1, ilens.max()))
        wav = wav.squeeze(1)

        return wav, ilens

    def forward_streaming(self, input_frame: torch.Tensor):
        """
        Forward streaming of audio frames through the ConvDecoder.

        This method processes a single input frame and returns the output
        waveform corresponding to that frame. It is primarily used for
        streaming applications where audio is processed in small chunks.

        Args:
            input_frame (torch.Tensor): A tensor representing the input frame
                of audio to be processed. The shape should be [B, F], where
                B is the batch size and F is the frame size.

        Returns:
            torch.Tensor: The output waveform after processing the input frame,
            with the shape [B, T], where T is the length of the output waveform.

        Examples:
            >>> decoder = ConvDecoder(channel=16, kernel_size=32, stride=16)
            >>> input_frame = torch.randn(1, 32)  # Example input frame
            >>> output_waveform = decoder.forward_streaming(input_frame)
            >>> print(output_waveform.shape)  # Output shape will be [1, T]
        """
        return self.forward(input_frame, ilens=torch.LongTensor([self.kernel_size]))[0]

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """
        Stream Merge.

        It merges the frame-level processed audio chunks in the streaming
        simulation. It is noted that, in real applications, the processed
        audio should be sent to the output channel frame by frame. You may
        refer to this function to manage your streaming output buffer.

        Args:
            chunks (torch.Tensor): A list of tensors where each tensor has the
                shape (B, frame_size), representing processed audio chunks.
            ilens (torch.Tensor, optional): A tensor of shape [B] containing
                the lengths of each batch. If not provided, the maximum length
                will be calculated based on the number of chunks.

        Returns:
            torch.Tensor: A tensor of shape [B, T] representing the merged audio
                output, where T is the total length of the merged audio.

        Examples:
            >>> decoder = ConvDecoder(channel=16, kernel_size=32, stride=16)
            >>> chunks = [torch.randn(1, 32) for _ in range(5)]
            >>> merged_audio = decoder.streaming_merge(chunks)
            >>> print(merged_audio.shape)
            torch.Size([1, 128])  # Example output shape based on the chunks

        Note:
            The `chunks` should be provided in the order they were processed,
            and the merging assumes that the frames overlap according to the
            defined `stride`.
        """
        hop_size = self.stride
        frame_size = self.kernel_size

        num_chunks = len(chunks)
        batch_size = chunks[0].shape[0]
        audio_len = (
            int(hop_size * num_chunks + frame_size - hop_size)
            if not ilens
            else ilens.max()
        )

        output = torch.zeros((batch_size, audio_len), dtype=chunks[0].dtype).to(
            chunks[0].device
        )

        for i, chunk in enumerate(chunks):
            output[:, i * hop_size : i * hop_size + frame_size] += chunk

        return output


if __name__ == "__main__":
    from espnet2.enh.encoder.conv_encoder import ConvEncoder

    input_audio = torch.randn((1, 100))
    ilens = torch.LongTensor([100])

    kernel_size = 32
    stride = 16

    encoder = ConvEncoder(kernel_size=kernel_size, stride=stride, channel=16)
    decoder = ConvDecoder(kernel_size=kernel_size, stride=stride, channel=16)
    frames, flens = encoder(input_audio, ilens)
    wav, ilens = decoder(frames, ilens)

    splited = encoder.streaming_frame(input_audio)

    sframes = [encoder.forward_streaming(s) for s in splited]
    swavs = [decoder.forward_streaming(s) for s in sframes]
    merged = decoder.streaming_merge(swavs, ilens)

    sframes = torch.cat(sframes, dim=1)

    torch.testing.assert_allclose(sframes, frames)
    torch.testing.assert_allclose(wav, merged)
