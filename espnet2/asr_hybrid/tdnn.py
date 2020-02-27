import torch


class ConvNdWithSamePadding(torch.nn.Module):
    convndim: int = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.conv = getattr(torch.nn, f"Conv{self.convndim}d")(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if isinstance(kernel_size, int):
            kernel_size = tuple(kernel_size for _ in range(self.convndim))
        if isinstance(stride, int):
            stride = tuple(stride for _ in range(self.convndim))
        if isinstance(dilation, int):
            dilation = tuple(dilation for _ in range(self.convndim))

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        insize = x.shape[2:]
        ps = [
            (i + 1 - h + s * (h - 1) + d * (k - 1)) // 2
            for h, k, s, d in list(
                zip(insize, self.kernel_size, self.stride, self.dilation)
            )[::-1]
            for i in range(2)
        ]
        # Padding to make the output shape to have the same shape as the input
        x = torch.nn.functional.pad(x, ps, 'constant', 0)
        return self.conv(x)


class Conv1dWithSamePadding(ConvNdWithSamePadding):
    convndim: int = 1


class Conv2dWithSamePadding(ConvNdWithSamePadding):
    convndim: int = 2


class Conv3dWithSamePadding(ConvNdWithSamePadding):
    convndim: int = 3


class Transpose(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class SamePaddingTDNN(torch.nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_num):
        super().__init__(
            # (B, Time, Freq) -> (B, Freq, Time)
            Transpose(1, 2),
            Conv1dWithSamePadding(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=1,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=1,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                dilation=1,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=3,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=3,
            ),
            torch.nn.ReLU(),
            Conv1dWithSamePadding(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                dilation=3,
            ),
            torch.nn.ReLU(),
            # (B, Freq, Time) -> (B, Time, Freq)
            Transpose(1, 2),
            torch.nn.Linear(hidden_dim, output_num),
        )
