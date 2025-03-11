# SKA-TDNN, original code from: https://github.com/msh9184/ska-tdnn
# adapted for ESPnet-SPK by Jee-weon Jung
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=None,
        kernel_sizes=[5, 7],
        dilation=None,
        scale=8,
        group=1,
    ):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        self.skconvs = nn.ModuleList([])
        for i in range(self.nums):
            convs = nn.ModuleList([])
            for k in kernel_sizes:
                convs += [
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv1d(
                                        width,
                                        width,
                                        kernel_size=k,
                                        dilation=dilation,
                                        padding=k // 2 * dilation,
                                        groups=group,
                                    ),
                                ),
                                ("relu", nn.ReLU()),
                                ("bn", nn.BatchNorm1d(width)),
                            ]
                        )
                    )
                ]
            self.skconvs += [convs]
        self.skse = SKAttentionModule(
            channel=width, reduction=4, num_kernels=len(kernel_sizes)
        )
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.se = SEModule(channels=planes)
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.skse(sp, self.skconvs[i])
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        reduction: int = 8,
        skfwse_freq: int = 40,
        skcwse_channel: int = 128,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.skfwse = fwSKAttention(
            freq=skfwse_freq,
            channel=skcwse_channel,
            kernels=[5, 7],
            receptive=[5, 7],
            dilations=[1, 1],
            reduction=reduction,
            groups=1,
        )
        self.skcwse = cwSKAttention(
            freq=skfwse_freq,
            channel=skcwse_channel,
            kernels=[5, 7],
            receptive=[5, 7],
            dilations=[1, 1],
            reduction=reduction,
            groups=1,
        )
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.skfwse(out)
        out = self.skcwse(out)
        out += residual
        out = self.relu(out)
        return out


class SKAttentionModule(nn.Module):
    def __init__(self, channel=128, reduction=4, L=16, num_kernels=2):
        super(SKAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.D = max(L, channel // reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(num_kernels):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, convs):
        """Forward function.

        Input: [B, C, T]
        Split: [K, B, C, T]
        Fues: [B, C, T]
        Attention weight: [B, C, 1]
        Output: [B, C, T]
        """
        bs, c, t = x.size()
        conv_outs = []
        for conv in convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [weight.view(bs, c, 1)]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class fwSKAttention(nn.Module):
    def __init__(
        self,
        freq=40,
        channel=128,
        kernels=[3, 5],
        receptive=[3, 5],
        dilations=[1, 1],
        reduction=8,
        groups=1,
        L=16,
    ):
        super(fwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=r // 2,
                                    dilation=d,
                                    groups=groups,
                                ),
                            ),
                            ("relu", nn.ReLU()),
                            ("bn", nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, freq // reduction)
        self.fc = nn.Linear(freq, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, freq)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """Forward function.

        Input: [B, C, F, T]
        Split: [K, B, C, F, T]
        Fues: [B, C, F, T]
        Attention weight: [K, B, 1, F, 1]
        Output: [B, C, F, T]
        """
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs).permute(0, 2, 3, 1)
        S = self.avg_pool(U).view(bs, f)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [weight.view(bs, 1, f, 1)]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class cwSKAttention(nn.Module):
    def __init__(
        self,
        freq=40,
        channel=128,
        kernels=[3, 5],
        receptive=[3, 5],
        dilations=[1, 1],
        reduction=8,
        groups=1,
        L=16,
    ):
        super(cwSKAttention, self).__init__()
        self.convs = nn.ModuleList([])
        for k, d, r in zip(kernels, dilations, receptive):
            self.convs += [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=r // 2,
                                    dilation=d,
                                    groups=groups,
                                ),
                            ),
                            ("relu", nn.ReLU()),
                            ("bn", nn.BatchNorm2d(channel)),
                        ]
                    )
                )
            ]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.D = max(L, channel // reduction)
        self.fc = nn.Linear(channel, self.D)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs += [nn.Linear(self.D, channel)]
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """Forward Function.

        Input: [B, C, F, T]
        Split: [K, B, C, F, T]
        Fuse: [B, C, F, T]
        Attention weight: [K, B, C, 1, 1]
        Output: [B, C, F, T]
        """
        bs, c, f, t = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs += [conv(x)]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = self.avg_pool(U).view(bs, c)
        Z = self.fc(S)
        Z = self.relu(Z)
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights += [weight.view(bs, c, 1, 1)]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class SkaTdnnEncoder(AbsEncoder):
    """SKA-TDNN encoder. Extracts frame-level SKA-TDNN embeddings from features.

    Paper: S. Mun, J. Jung et al., "Frequency and Multi-Scale Selective Kernel
        Attention for Speaker Verification,' in Proc. IEEE SLT 2022.

    Args:
        input_size: input feature dimension.
        block: type of encoder block class to use.
        model_scale: scale value of the Res2Net architecture.
        ndim: dimensionality of the hidden representation.
        output_size: ouptut embedding dimension.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: str = "Bottle2neck",
        ndim: int = 1024,
        model_scale: int = 8,
        skablock: str = "ResBlock",
        ska_dim: int = 128,
        output_size: int = 1536,
        **kwargs,
    ):
        super().__init__()

        if block == "Bottle2neck":
            block: type = Bottle2neck
        else:
            raise ValueError(f"unsupported block, got: {block}")

        if skablock == "ResBlock":
            ska_block = ResBlock
        else:
            raise ValueError(f"unsupported block, got: {ska_block}")

        self.frt_conv1 = nn.Conv2d(
            1, ska_dim, kernel_size=(3, 3), stride=(2, 1), padding=1
        )
        self.frt_bn1 = nn.BatchNorm2d(ska_dim)
        self.frt_block1 = ska_block(
            ska_dim,
            ska_dim,
            stride=(1, 1),
            skfwse_freq=input_size // 2,
            skcwse_channel=ska_dim,
        )
        self.frt_block2 = ska_block(
            ska_dim,
            ska_dim,
            stride=(1, 1),
            skfwse_freq=input_size // 2,
            skcwse_channel=ska_dim,
        )
        self.frt_conv2 = nn.Conv2d(
            ska_dim, ska_dim, kernel_size=(3, 3), stride=(2, 2), padding=1
        )
        self.frt_bn2 = nn.BatchNorm2d(ska_dim)
        self.conv1 = nn.Conv1d(
            ska_dim * input_size // 4, ndim, kernel_size=5, stride=1, padding=2
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(ndim)
        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=2, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=3, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)
        self._output_size = output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, S, D) -> (B, D, S)
        x = x.unsqueeze(1)  # (B, D, S) -> (B, 1, D, S)

        # the fcwSKA block
        x = self.frt_conv1(x)
        x = self.relu(x)
        x = self.frt_bn1(x)
        x = self.frt_block1(x)
        x = self.frt_block2(x)
        x = self.frt_conv2(x)
        x = self.relu(x)
        x = self.frt_bn2(x)

        x = x.reshape((x.size()[0], -1, x.size()[-1]))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        return x
