import math

import torch

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class ConvDecoder(AbsDecoder):
    """Transposed Convolutional decoder for speech enhancement and separation"""

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

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
        input (torch.Tensor): spectrum [Batch, T, F]
        ilens (torch.Tensor): input lengths [Batch]
        """
        input = input.transpose(1, 2)
        batch_size = input.shape[0]
        wav = self.convtrans1d(input, output_size=(batch_size, 1, ilens.max()))
        wav = wav.squeeze(1)

        return wav, ilens

    def forward_streaming(self, input_frame: torch.Tensor):
        return self.forward(input_frame, ilens=torch.LongTensor([self.kernel_size]))[0]

    def streaming_merge(self, chunks: torch.Tensor, ilens: torch.tensor = None):
        """streaming_merge. It merges the frame-level processed audio chunks
        in the streaming *simulation*. It is noted that, in real applications,
        the processed audio should be sent to the output channel frame by frame.
        You may refer to this function to manage your streaming output buffer.

        Args:
            chunks: List [(B, frame_size),]
            ilens: [B]
        Returns:
            merge_audio: [B, T]
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
