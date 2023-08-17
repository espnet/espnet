import math

import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class ConvEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

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
        return self._output_dim

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.

        Args:
            input (torch.Tensor): mixed speech [Batch, sample]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            feature (torch.Tensor): mixed feature after encoder [Batch, flens, channel]
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
        output, _ = self.forward(input, 0)
        return output

    def streaming_frame(self, audio: torch.Tensor):
        """streaming_frame. It splits the continuous audio into frame-level
        audio chunks in the streaming *simulation*. It is noted that this
        function takes the entire long audio as input for a streaming simulation.
        You may refer to this function to manage your streaming input
        buffer in a real streaming application.

        Args:
            audio: (B, T)
        Returns:
            chunked: List [(B, frame_size),]
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
