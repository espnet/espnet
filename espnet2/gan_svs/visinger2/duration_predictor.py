import torch
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    def __init__(
        self,
        channels,
        filter_channels,
        kernel_size,
        dropout_rate,
        global_channels=0,
    ):
        super().__init__()

        self.in_channels = channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = dropout_rate

        self.drop = torch.nn.Dropout(dropout_rate)
        self.conv_1 = torch.nn.Conv1d(
            channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels, dim=1)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels, dim=1)
        self.conv_3 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_3 = LayerNorm(filter_channels, dim=1)
        self.proj = torch.nn.Conv1d(filter_channels, 2, 1)

        if global_channels > 0:
            self.conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        # x = torch.detach(x)

        # multi-singer
        if g is not None:
            g = torch.detach(g)
            x = x + self.conv(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask
