"""Extension of convolutional gating (cgMLP) definition with multiple convolutions.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050
    https://arxiv.org/abs/2407.03718
"""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class MultiConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Multi Convolutional Spatial Gating Unit (M-CSGU)."""

    def __init__(
        self,
        size: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)

        kernel_sizes = list(map(int, kernel_sizes.split(",")))
        no_kernels = len(kernel_sizes)

        assert (
            n_channels % no_kernels == 0
        ), f"{n_channels} input channels cannot be divided between {no_kernels} kernels"

        self.arch_type = arch_type
        if arch_type in ["sum", "weighted_sum"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        elif arch_type in ["concat", "concat_fusion"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels // no_kernels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels // no_kernels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        else:
            raise NotImplementedError(
                f"Unknown architecture type for MultiConvCGMLP: {arch_type}"
            )
        self.use_non_linear = use_non_linear
        if arch_type == "weighted_sum":
            self.kernel_prob_gen = torch.nn.Sequential(
                torch.nn.Linear(n_channels * no_kernels, no_kernels),
                torch.nn.Softmax(dim=-1),
            )
            self.depthwise_conv_fusion = None
        elif arch_type == "concat_fusion":
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=n_channels,
                bias=True,
            )
        else:
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = None

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.model_act = activation
        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=1e-6)
            torch.nn.init.ones_(conv.bias)
        if self.depthwise_conv_fusion is not None:
            torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
            torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """
        x_r, x_i = x.chunk(2, dim=-1)

        x_i = self.norm(x_i).transpose(1, 2)  # (N, D/2, T)

        # TODO(gituser): Parallelize this convolution computation
        xs = []
        for conv in self.convs:
            xi = conv(x_i).transpose(1, 2)  # (N, T, D/2)
            if self.arch_type == "sum" and self.use_non_linear:
                xi = self.model_act(xi)
            xs.append(xi)

        if self.arch_type in ["sum", "weighted_sum"]:
            x = torch.stack(xs, dim=-2)
            if self.arch_type == "weighted_sum":
                prob = self.kernel_prob_gen(torch.cat(xs, dim=-1))
                x = prob.unsqueeze(-1) * x

            x_g = x.sum(dim=-2)
        else:
            x_concat = torch.cat(xs, dim=-1)  # (N, T, D)

            if self.arch_type == "concat_fusion":
                x_tmp = x_concat.transpose(1, 2)
                x_tmp = self.depthwise_conv_fusion(x_tmp)
                x_concat = x_concat + x_tmp.transpose(1, 2)

            x_g = x_concat

        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class MultiConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        if arch_type not in ["sum", "weighted_sum", "concat", "concat_fusion"]:
            raise NotImplementedError(f"Unknown MultiConvCGMLP type: {type}")

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = MultiConvolutionalSpatialGatingUnit(
            size=linear_units,
            arch_type=arch_type,
            kernel_sizes=kernel_sizes,
            merge_conv_kernel=merge_conv_kernel,
            use_non_linear=use_non_linear,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out
