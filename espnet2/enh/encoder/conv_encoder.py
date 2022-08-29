import torch

from espnet2.enh.encoder.abs_encoder import AbsEncoder


class ConvEncoder(AbsEncoder):
    """Convolutional encoder for speech enhancement and separation"""

    def __init__(
        self, channel: int, kernel_size: int, stride: int,
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

        flens = (ilens - self.kernel_size) // self.stride + 1

        return feature, flens
