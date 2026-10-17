# Copyright 2023 Jee-weon Jung, 2025 Qingzheng Wang, 2026 ESPnet contributors
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Portable inference layers for the released GeoLID checkpoints.

These layers mirror espnet2.spk's ECAPA, identity encoder, channel-attentive
pooling and RawNet3 projector. They are bundled into the checkpoint so inference
needs only PyTorch and Transformers. Regression tests compare them with ESPnet.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
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


class EcapaBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

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
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
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


class EcapaTdnnEncoder(nn.Module):
    """ECAPA-TDNN encoder. Extracts frame-level ECAPA-TDNN embeddings from

    mel-filterbank energy or MFCC features.
    Paper: B Desplanques at el., ``ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification,''
        in Proc. INTERSPEECH, 2020.

    Args:
        input_size: input feature dimension.
        block: type of encoder block class to use.
        model_scale: scale value of the Res2Net architecture.
        ndim: dimensionality of the hidden representation.
        output_size: output embedding dimension.
        num_blocks: number of SE-Res2 blocks, with dilations 2, 3, ...,
            num_blocks + 1. The default is the paper's three blocks (2, 3, 4).
            Four blocks (2, 3, 4, 5) follow the IDLab VoxSRC-20 system, which
            adds a fourth SE-Res2Block with dilation 5 (J. Thienpondt et al.,
            ``The IDLAB VoxSRC-20 Submission: Large Margin Fine-Tuning and
            Quality-Aware Score Calibration in DNN Based Speaker
            Verification,'' arXiv:2010.11255, Sec. 2). For five or more,
            choosing the dilations with ``dilations`` is recommended.
        dilations: dilation of each SE-Res2 block, such as ``[2, 3, 4, 2, 3, 4]``;
            the number of blocks is its length. Dilations do not change the
            parameter shapes, so build the model from the configuration it
            was trained with (the ``encoder_conf`` saved in its
            ``config.yaml``).

    Every block takes the first convolution's output plus the outputs of all
    earlier blocks, and the 1x1 convolution ``layer4`` aggregates all block
    outputs. The first three blocks keep the names ``layer1``-``layer3`` and
    further blocks are ``extra_layers``, so three-block checkpoints load
    unchanged.
    """

    def __init__(
        self,
        input_size: int,
        block: str = "EcapaBlock",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        num_blocks: Optional[int] = None,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if block == "EcapaBlock":
            block: type = EcapaBlock
        else:
            raise ValueError(f"unsupported block, got: {block}")
        self._output_size = output_size

        self.conv = nn.Conv1d(input_size, ndim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ndim)

        if dilations is None:
            dilations = range(2, (3 if num_blocks is None else num_blocks) + 2)
        elif num_blocks is not None and num_blocks != len(dilations):
            raise ValueError(
                f"num_blocks ({num_blocks}) differs from len(dilations) "
                f"({len(dilations)})"
            )
        dilations = list(dilations)
        if len(dilations) < 3:
            raise ValueError(f"at least 3 blocks are required, got: {dilations}")

        d1, d2, d3 = dilations[:3]
        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=d1, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=d2, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=d3, scale=model_scale)
        self.extra_layers = nn.ModuleList(
            block(ndim, ndim, kernel_size=3, dilation=d, scale=model_scale)
            for d in dilations[3:]
        )
        self.layer4 = nn.Conv1d(len(dilations) * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Input tensor (#batch, L, input_size).

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).

        """
        x = self.conv(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.bn(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        outputs = [x1, x2, x3]
        if self.extra_layers:
            total = x + x1 + x2 + x3
            for layer in self.extra_layers:
                outputs.append(layer(total))
                total = total + outputs[-1]

        x = self.layer4(torch.cat(outputs, dim=1))
        x = self.relu(x)

        return x


class IdentityEncoder(nn.Module):
    """Identity encoder. Does nothing, just passes frontend feature to the pooling.

    Expected to be used for cases when frontend already has a good
    representation (e.g., SSL features).

    Args:
        input_size: input feature dimension.
    """

    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        return x.transpose(1, 2)


class ChnAttnStatPooling(nn.Module):
    """Aggregates frame-level features to single utterance-level feature.

    Reference:
    ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in TDNN Based Speaker Verification
    https://arxiv.org/pdf/2005.07143

    Args:
        input_size: Dimension of the input frame-level embeddings.
                    The output dimensionality will be 2 × input_size
                    after concatenating mean and std.
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=2)
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of channel-attentive statistical pooling.

        Args:
            x: Input feature tensor of shape (batch_size, feature_dim, seq_len)
            feat_lengths: Optional tensor of shape (batch_size,) containing
                          the valid length of each sequence before padding

        Returns:
            x: Utterance-level embeddings of shape (batch_size, 2 × feature_dim)
        """

        T = x.size(-1)
        if feat_lengths is not None:
            feat_lengths = feat_lengths.to(x.device)
            # Pooling over unpadded frames
            mask = (
                torch.arange(T, device=x.device)[None, None, :]
                < feat_lengths[:, None, None]
            )
            # set padding to 0 for sum
            masked_x = x.masked_fill(~mask, 0)
            # sum over time
            sum_val = masked_x.sum(dim=-1, keepdim=True)
            sum_sq_val = (masked_x**2).sum(dim=-1, keepdim=True)
            # feat_lengths might be 0, add epsilon
            feat_lengths_ = feat_lengths.view(-1, 1, 1).clamp(min=1)
            mean = sum_val / feat_lengths_
            # var = E[X^2] - (E[X])^2
            var = sum_sq_val / feat_lengths_ - mean**2
            # add max clamp to prevent gradient explosion
            std = torch.sqrt(var.clamp(min=torch.finfo(var.dtype).eps, max=1e4))
            # Repeat mean and std to match x's time dimension
            mean = mean.repeat(1, 1, T)
            std = std.repeat(1, 1, T)
            global_x = torch.cat((x, mean, std), dim=1)
        else:
            var = torch.var(x, dim=2, keepdim=True, unbiased=False)
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, T),
                    torch.sqrt(
                        var.clamp(
                            min=torch.finfo(x.dtype).eps, max=1e4
                        )  # clamp max to prevent gradient explosion
                    ).repeat(1, 1, T),
                ),
                dim=1,
            )

        w = self.attention(global_x)
        if feat_lengths is not None:
            # Apply padding mask
            padding_mask = torch.arange(T).expand(x.size(0), T).to(
                x.device
            ) >= feat_lengths.unsqueeze(
                1
            )  # (batch_size, seq_len)
            w = w.masked_fill(padding_mask.unsqueeze(1), torch.finfo(w.dtype).min)
        w = self.softmax(w)

        mu = torch.sum(x * w, dim=2)
        # add max clamp to prevent gradient explosion
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x


class RawNet3Projector(nn.Module):
    def __init__(self, input_size, output_size=192):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size)
        self.fc = torch.nn.Linear(input_size, output_size)

    def output_size(self):
        return self._output_size

    def forward(self, x):
        return self.fc(self.bn(x))


class LegacyChnAttnStatPooling(ChnAttnStatPooling):
    """Pooling used to train the 2025 checkpoints (including its variance floor)."""

    def forward(
        self,
        x: torch.Tensor,
        feat_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of channel-attentive statistical pooling.

        Args:
            x: Input feature tensor of shape (batch_size, feature_dim, seq_len)
            feat_lengths: Optional tensor of shape (batch_size,) containing
                          the valid length of each sequence before padding

        Returns:
            x: Utterance-level embeddings of shape (batch_size, 2 × feature_dim)
        """

        T = x.size(-1)
        if feat_lengths is not None:
            # Pooling over unpadded frames
            mean = torch.stack(
                [
                    torch.mean(x[i, :, : int(l.item())], dim=-1, keepdim=True)
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            ).repeat(1, 1, T)
            var = torch.stack(
                [
                    torch.var(
                        x[i, :, : int(l.item())], dim=-1, unbiased=False, keepdim=True
                    )
                    for i, l in enumerate(feat_lengths)
                ],
                dim=0,
            )
            var = var.clamp(min=1e-4, max=1e4)
            std = torch.sqrt(var).repeat(1, 1, T)
            global_x = torch.cat((x, mean, std), dim=1)
        else:
            global_x = torch.cat(
                (
                    x,
                    torch.mean(x, dim=2, keepdim=True).repeat(1, 1, T),
                    torch.sqrt(
                        torch.var(x, dim=2, keepdim=True, unbiased=False).clamp(
                            min=1e-4, max=1e4
                        )
                    ).repeat(1, 1, T),
                ),
                dim=1,
            )

        w = self.attention(global_x)
        if feat_lengths is not None:
            # Apply padding mask
            padding_mask = torch.arange(T).expand(x.size(0), T).to(
                x.device
            ) >= feat_lengths.unsqueeze(
                1
            )  # (batch_size, seq_len)
            w = w.masked_fill(padding_mask.unsqueeze(1), torch.finfo(w.dtype).min)
        w = self.softmax(w)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)

        return x


def legacy_mean_normalize(features, lengths):
    """Reproduce 2025 UtteranceMVN, including the mean in padded positions."""
    mask = (
        torch.arange(features.size(1), device=features.device)[None] < lengths[:, None]
    )
    features = features.masked_fill(~mask[:, :, None], 0)
    return features - features.sum(1, keepdim=True) / lengths[:, None, None]
