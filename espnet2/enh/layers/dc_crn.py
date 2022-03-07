# Implementation of Densely-connected convolutional recurrent network (DC-CRN)
# [1] Tan et al. "Deep Learning Based Real-Time Speech Enhancement for Dual-Microphone
#     Mobile Phones"
#     https://web.cse.ohio-state.edu/~wang.77/papers/TZW.taslp21.pdf


from typing import List

import torch
import torch.nn as nn

from espnet2.enh.layers.conv_utils import conv2d_output_shape
from espnet2.enh.layers.conv_utils import convtransp2d_output_shape


class GLSTM(nn.Module):
    def __init__(
        self, hidden_size=1024, groups=2, layers=2, bidirectional=False, rearrange=False
    ):
        """Grouped LSTM.

        Reference:
            Efficient Sequence Learning with Group Recurrent Networks; Gao et al., 2018

        Args:
            hidden_size (int): total hidden size of all LSTMs in each grouped LSTM layer
                i.e., hidden size of each LSTM is `hidden_size // groups`
            groups (int): number of LSTMs in each grouped LSTM layer
            layers (int): number of grouped LSTM layers
            bidirectional (bool): whether to use BLSTM or unidirectional LSTM
            rearrange (bool): whether to apply the rearrange operation after each
                grouped LSTM layer
        """
        super().__init__()

        assert hidden_size % groups == 0, (hidden_size, groups)
        hidden_size_t = hidden_size // groups
        if bidirectional:
            assert hidden_size_t % 2 == 0, hidden_size_t

        self.groups = groups
        self.layers = layers
        self.rearrange = rearrange

        self.lstm_list = nn.ModuleList()
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(layers)])
        for layer in range(layers):
            self.lstm_list.append(
                nn.ModuleList(
                    [
                        nn.LSTM(
                            hidden_size_t,
                            hidden_size_t // 2 if bidirectional else hidden_size_t,
                            1,
                            batch_first=True,
                            bidirectional=bidirectional,
                        )
                        for _ in range(groups)
                    ]
                )
            )

    def forward(self, x):
        """Grouped LSTM forward.

        Args:
            x (torch.Tensor): (B, C, T, D)
        Returns:
            out (torch.Tensor): (B, C, T, D)
        """
        out = x
        out = out.transpose(1, 2).contiguous()
        B, T = out.size(0), out.size(1)
        out = out.view(B, T, -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.lstm_list[0][i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln[0](out)

        for layer in range(1, self.layers):
            if self.rearrange:
                out = (
                    out.reshape(B, T, self.groups, -1)
                    .transpose(-1, -2)
                    .contiguous()
                    .view(B, T, -1)
                )
            out = torch.chunk(out, self.groups, dim=-1)
            out = torch.cat(
                [self.lstm_list[layer][i](out[i])[0] for i in range(self.groups)],
                dim=-1,
            )
            out = self.ln[layer](out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        """Conv2d with Gated Linear Units (GLU).

        Input and output shapes are the same as regular Conv2d layers.

        Reference: Section III-B in [1]

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int/tuple): kernel size in Conv2d
            stride (int/tuple): stride size in Conv2d
            padding (int/tuple): padding size in Conv2d
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ConvGLU forward.

        Args:
            x (torch.Tensor): (B, C_in, H_in, W_in)
        Returns:
            out (torch.Tensor): (B, C_out, H_out, W_out)
        """
        out = self.conv1(x)
        gate = self.sigmoid(self.conv2(x))
        return out * gate


class GluConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=(0, 0),
    ):
        """ConvTranspose2d with Gated Linear Units (GLU).

        Input and output shapes are the same as regular ConvTranspose2d layers.

        Reference: Section III-B in [1]

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int/tuple): kernel size in ConvTranspose2d
            stride (int/tuple): stride size in ConvTranspose2d
            padding (int/tuple): padding size in ConvTranspose2d
            output_padding (int/tuple): Additional size added to one side of each
                dimension in the output shape
        """
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """DeconvGLU forward.

        Args:
            x (torch.Tensor): (B, C_in, H_in, W_in)
        Returns:
            out (torch.Tensor): (B, C_out, H_out, W_out)
        """
        out = self.deconv1(x)
        gate = self.sigmoid(self.deconv2(x))
        return out * gate


class DenselyConnectedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hid_channels=8,
        kernel_size=(1, 3),
        padding=(0, 1),
        last_kernel_size=(1, 4),  # use (1, 4) to alleviate the checkerboard artifacts
        last_stride=(1, 2),
        last_padding=(0, 1),
        last_output_padding=(0, 0),
        layers=5,
        transposed=False,
    ):
        """Densely-Connected Convolutional Block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            hid_channels (int): number of output channels in intermediate Conv layers
            kernel_size (tuple): kernel size for all but the last Conv layers
            padding (tuple): padding for all but the last Conv layers
            last_kernel_size (tuple): kernel size for the last GluConv layer
            last_stride (tuple): stride for the last GluConv layer
            last_padding (tuple): padding for the last GluConv layer
            last_output_padding (tuple): output padding for the last GluConvTranspose2d
                 (only used when `transposed=True`)
            layers (int): total number of Conv layers
            transposed (bool): True to use GluConvTranspose2d in the last layer
                               False to use GluConv2d in the last layer
        """
        super().__init__()

        assert layers > 1, layers
        self.conv = nn.ModuleList()
        in_channel = in_channels
        # here T=42 and D=127 are random integers that should not be changed after Conv
        T, D = 42, 127
        hidden_sizes = [127]
        for _ in range(layers - 1):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel,
                        hid_channels,
                        kernel_size=kernel_size,
                        stride=(1, 1),
                        padding=padding,
                    ),
                    nn.BatchNorm2d(hid_channels),
                    nn.ELU(inplace=True),
                )
            )
            in_channel = in_channel + hid_channels
            # make sure the last two dimensions will not be changed after this layer
            tdim, hdim = conv2d_output_shape(
                (T, D),
                kernel_size=kernel_size,
                stride=(1, 1),
                pad=padding,
            )
            hidden_sizes.append(hdim)
            assert tdim == T and hdim == D, (tdim, hdim, T, D)

        if transposed:
            self.conv.append(
                GluConvTranspose2d(
                    in_channel,
                    out_channels,
                    kernel_size=last_kernel_size,
                    stride=last_stride,
                    padding=last_padding,
                    output_padding=last_output_padding,
                )
            )
        else:
            self.conv.append(
                GluConv2d(
                    in_channel,
                    out_channels,
                    kernel_size=last_kernel_size,
                    stride=last_stride,
                    padding=last_padding,
                )
            )

    def forward(self, input):
        """DenselyConnectedBlock forward.

        Args:
            input (torch.Tensor): (B, C, T_in, F_in)
        Returns:
            out (torch.Tensor): (B, C, T_out, F_out)
        """
        out = self.conv[0](input)
        outputs = [input, out]
        num_layers = len(self.conv)
        for idx, layer in enumerate(self.conv[1:]):
            out = layer(torch.cat(outputs, dim=1))
            if idx < num_layers - 1:
                outputs.append(out)
        return out


class DC_CRN(nn.Module):
    def __init__(
        self,
        input_dim,
        input_channels: List = [2, 16, 32, 64, 128, 256],
        enc_hid_channels=8,
        enc_kernel_size=(1, 3),
        enc_padding=(0, 1),
        enc_last_kernel_size=(1, 4),
        enc_last_stride=(1, 2),
        enc_last_padding=(0, 1),
        enc_layers=5,
        skip_last_kernel_size=(1, 3),
        skip_last_stride=(1, 1),
        skip_last_padding=(0, 1),
        glstm_groups=2,
        glstm_layers=2,
        glstm_bidirectional=False,
        glstm_rearrange=False,
        output_channels=2,
    ):
        """Densely-Connected Convolutional Recurrent Network (DC-CRN).

        Reference: Fig. 3 and Section III-B in [1]

        Args:
            input_dim (int): input feature dimension
            input_channels (list): number of input channels for the stacked
                DenselyConnectedBlock layers
                Its length should be (`number of DenselyConnectedBlock layers`).
                It is recommended to use even number of channels to avoid AssertError
                when `glstm_bidirectional=True`.
            enc_hid_channels (int): common number of intermediate channels for all
                DenselyConnectedBlock of the encoder
            enc_kernel_size (tuple): common kernel size for all DenselyConnectedBlock
                of the encoder
            enc_padding (tuple): common padding for all DenselyConnectedBlock
                of the encoder
            enc_last_kernel_size (tuple): common kernel size for the last Conv layer
                in all DenselyConnectedBlock of the encoder
            enc_last_stride (tuple): common stride for the last Conv layer in all
                DenselyConnectedBlock of the encoder
            enc_last_padding (tuple): common padding for the last Conv layer in all
                DenselyConnectedBlock of the encoder
            enc_layers (int): common total number of Conv layers for all
                DenselyConnectedBlock layers of the encoder
            skip_last_kernel_size (tuple): common kernel size for the last Conv layer
                in all DenselyConnectedBlock of the skip pathways
            skip_last_stride (tuple): common stride for the last Conv layer in all
                DenselyConnectedBlock of the skip pathways
            skip_last_padding (tuple): common padding for the last Conv layer in all
                DenselyConnectedBlock of the skip pathways
            glstm_groups (int): number of groups in each Grouped LSTM layer
            glstm_layers (int): number of Grouped LSTM layers
            glstm_bidirectional (bool): whether to use BLSTM or unidirectional LSTM
                in Grouped LSTM layers
            glstm_rearrange (bool): whether to apply the rearrange operation after each
                grouped LSTM layer
            output_channels (int): number of output channels (must be an even number to
                recover both real and imaginary parts)
        """
        super().__init__()

        assert output_channels % 2 == 0, output_channels
        self.conv_enc = nn.ModuleList()
        # here T=42 is a random integer that should not be changed after Conv
        T = 42
        hidden_sizes = [input_dim]
        hdim = input_dim
        for i in range(1, len(input_channels)):
            self.conv_enc.append(
                DenselyConnectedBlock(
                    in_channels=input_channels[i - 1],
                    out_channels=input_channels[i],
                    hid_channels=enc_hid_channels,
                    kernel_size=enc_kernel_size,
                    padding=enc_padding,
                    last_kernel_size=enc_last_kernel_size,
                    last_stride=enc_last_stride,
                    last_padding=enc_last_padding,
                    layers=enc_layers,
                    transposed=False,
                )
            )
            tdim, hdim = conv2d_output_shape(
                (T, hdim),
                kernel_size=enc_last_kernel_size,
                stride=enc_last_stride,
                pad=enc_last_padding,
            )
            hidden_sizes.append(hdim)
            assert tdim == T, (tdim, hdim)

        hs = hdim * input_channels[-1]
        assert hs >= glstm_groups, (hs, glstm_groups)
        self.glstm = GLSTM(
            hidden_size=hs,
            groups=glstm_groups,
            layers=glstm_layers,
            bidirectional=glstm_bidirectional,
            rearrange=glstm_rearrange,
        )

        self.skip_pathway = nn.ModuleList()
        self.deconv_dec = nn.ModuleList()
        for i in range(len(input_channels) - 1, 0, -1):
            self.skip_pathway.append(
                DenselyConnectedBlock(
                    in_channels=input_channels[i],
                    out_channels=input_channels[i],
                    hid_channels=enc_hid_channels,
                    kernel_size=enc_kernel_size,
                    padding=enc_padding,
                    last_kernel_size=skip_last_kernel_size,
                    last_stride=skip_last_stride,
                    last_padding=skip_last_padding,
                    layers=enc_layers,
                    transposed=False,
                )
            )
            # make sure the last two dimensions will not be changed after this layer
            enc_hdim = hidden_sizes[i]
            tdim, hdim = conv2d_output_shape(
                (T, enc_hdim),
                kernel_size=skip_last_kernel_size,
                stride=skip_last_stride,
                pad=skip_last_padding,
            )
            assert tdim == T and hdim == enc_hdim, (tdim, hdim, T, enc_hdim)

            if i != 1:
                out_ch = input_channels[i - 1]
            else:
                out_ch = output_channels
            # make sure the last but one dimension will not be changed after this layer
            tdim, hdim = convtransp2d_output_shape(
                (T, enc_hdim),
                kernel_size=enc_last_kernel_size,
                stride=enc_last_stride,
                pad=enc_last_padding,
            )
            assert tdim == T, (tdim, hdim)
            hpadding = hidden_sizes[i - 1] - hdim
            assert hpadding >= 0, (hidden_sizes[i - 1], hdim)
            self.deconv_dec.append(
                DenselyConnectedBlock(
                    in_channels=input_channels[i] * 2,
                    out_channels=out_ch,
                    hid_channels=enc_hid_channels,
                    kernel_size=enc_kernel_size,
                    padding=enc_padding,
                    last_kernel_size=enc_last_kernel_size,
                    last_stride=enc_last_stride,
                    last_padding=enc_last_padding,
                    last_output_padding=(0, hpadding),
                    layers=enc_layers,
                    transposed=True,
                )
            )

        self.fc_real = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.fc_imag = nn.Linear(in_features=input_dim, out_features=input_dim)

    def forward(self, x):
        """DC-CRN forward.

        Args:
            x (torch.Tensor): Concatenated real and imaginary spectrum features
                (B, input_channels[0], T, F)
        Returns:
            out (torch.Tensor): (B, 2, output_channels, T, F)
        """
        out = x
        conv_out = []
        for idx, layer in enumerate(self.conv_enc):
            out = layer(out)
            conv_out.append(out)

        num_out = len(conv_out)
        out = self.glstm(conv_out[-1])
        res = self.skip_pathway[0](conv_out[-1])
        out = torch.cat((out, res), dim=1)

        for idx in range(len(self.deconv_dec) - 1):
            deconv_out = self.deconv_dec[idx](out)
            res = self.skip_pathway[idx + 1](conv_out[num_out - idx - 2])
            out = torch.cat((deconv_out, res), dim=1)
        out = self.deconv_dec[-1](out)

        dout_real, dout_imag = torch.chunk(out, 2, dim=1)

        out_real = self.fc_real(dout_real)
        out_imag = self.fc_imag(dout_imag)
        out = torch.stack([out_real, out_imag], dim=1)

        return out
