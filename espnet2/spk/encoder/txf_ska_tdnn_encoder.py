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
            weights += [(weight.view(bs, c, 1))]
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
            weights += [(weight.view(bs, 1, f, 1))]
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
            weights += [(weight.view(bs, c, 1, 1))]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V

class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
      return x + self.pe[:, :x.size(1)]

class DualInputTransformerLayer(nn.Module):
    ''' Transformer Layer with dual input support '''

    def __init__(self, d_model, attention_heads, ff_dim=2048, attention_dropout_rate=0.1, dropout_rate=0.1):
        super().__init__()
    
        # Multi-Head Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=attention_heads,
                                                dropout=attention_dropout_rate,
                                                batch_first=True)
        # Multi-Head Cross-Attention Layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=attention_heads,
                                                dropout=attention_dropout_rate,
                                                batch_first=True)

        # Position-wise Feed-Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model))
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def padmask(self, padded_input, input_lengths=None):
        """ Create a mask to identify non-padding positions.

        Args:
            padded_input: The input tensor with padding, shape (N, T, ...)
            input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).

        Returns:
            A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
        """
        N, T, _ = padded_input.shape
        mask = torch.ones((N, T), dtype=torch.bool, device=padded_input.device) # Initialize mask with True
        if input_lengths is not None:
            for i in range(N):
                mask[i, :input_lengths[i]] = False       # Set non-padding positions to False
        else:
            raise ValueError("input_lengths must be provided")

        return mask

    def forward(self, x, precomp_x, precomp_x_lengths):

        residual = x

        # Self-Attention
        self_attn_output , self_attn_weights = self.self_attn(query=x,
                                                                key=x,
                                                                value=x,
                                                                key_padding_mask=None,
                                                                need_weights=True,
                                                                attn_mask=None,
                                                                average_attn_weights=True,
                                                                is_causal=False)
        self_attn_output = self.dropout1(self_attn_output)
        x = self.layer_norm1(residual + self_attn_output)
        residual = x

        # Cross-Attention
        cross_attn_key_mask = self.padmask(padded_input=precomp_x, input_lengths=precomp_x_lengths)
        cross_attn_output , cross_attn_weights = self.cross_attn(query=x,
                                                                key=precomp_x,
                                                                value=precomp_x,
                                                                key_padding_mask=cross_attn_key_mask,
                                                                need_weights=True,
                                                                attn_mask=None,
                                                                average_attn_weights=True,
                                                                is_causal=False)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = self.layer_norm2(residual + cross_attn_output)
        residual = x

        # Position-wise Feed-Forward
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout3(feed_forward_output)
        x = self.layer_norm3(residual + feed_forward_output)

        return x, self_attn_weights, cross_attn_weights

class SingleInputTransformerLayer(nn.Module):
    ''' Transformer Layer with single input support '''

    def __init__(self, d_model, attention_heads, ff_dim=2048, attention_dropout_rate=0.1, dropout_rate=0.1):
        super().__init__()
    
        # Multi-Head Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=attention_heads,
                                                dropout=attention_dropout_rate,
                                                batch_first=True)

        # Position-wise Feed-Forward Layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model))
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):

        residual = x

        # Self-Attention
        self_attn_output , self_attn_weights = self.self_attn(query=x,
                                                                key=x,
                                                                value=x,
                                                                key_padding_mask=None,
                                                                need_weights=True,
                                                                attn_mask=None,
                                                                average_attn_weights=True,
                                                                is_causal=False)
        self_attn_output = self.dropout1(self_attn_output)
        x = self.layer_norm1(residual + self_attn_output)
        residual = self_attn_output

        # Position-wise Feed-Forward
        feed_forward_output = self.feed_forward(x)
        feed_forward_output = self.dropout2(feed_forward_output)
        x = self.layer_norm2(residual + feed_forward_output)

        return x, self_attn_weights

class TxfSkaTdnnEncoder(AbsEncoder):
    """SKA-TDNN encoder with transformer layers.

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
        oned_module: str = "tdnn", # 1dmodule can take: transfromer, tdnn
        precomp_input_stage = "none", # precomp_input_stage can take: oned, none
        precomp_dim: int = 2048, # dimension of the precomputed frame-level features
        num_blocks: int = 6, # number of transformer blocks
        d_model: int = 512, # dimension of the model
        attention_heads: int = 8, # number of attention heads
        ff_dim: int = 2048, # dimension of the feedforward network model
        attention_dropout_rate: float = 0.1, # dropout rate for the attention layer
        dropout_rate: float = 0.1, # dropout rate for the model
        max_len: int = 512, # maximum length of the input sequence
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

        self.oned_module = oned_module
        if self.oned_module != "tdnn" and self.oned_module != "transformer":
            raise ValueError(f"unsupported oned_module, got: {self.oned_module}, should be tdnn or transformer")
        
        self.precomp_input_stage = precomp_input_stage
        if self.precomp_input_stage != "none" and self.precomp_input_stage != "oned":
            raise ValueError(f"unsupported precomp_input_stage, got: {precomp_input_stage}, should be none or oned")     

        self.precomp_dim = precomp_dim       

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
        self.relu = nn.ReLU()

        if self.oned_module == "tdnn":
            self.conv1 = nn.Conv1d(
                ska_dim * input_size // 4, ndim, kernel_size=5, stride=1, padding=2
            )
            self.bn1 = nn.BatchNorm1d(ndim)

            self.tdnn_layer1 = block(ndim, ndim, kernel_size=3, dilation=2, scale=model_scale)
            self.tdnn_layer2 = block(ndim, ndim, kernel_size=3, dilation=3, scale=model_scale)
            self.tdnn_layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
            self.tdnn_layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)

        elif self.oned_module == "transformer":
            self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

            self.x_embedding_conv = nn.Conv1d(in_channels=(ska_dim * input_size // 4),
                                        out_channels=d_model,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
            self.precomp_embedding_conv = nn.Conv1d(in_channels=precomp_dim,
                                            out_channels=d_model,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1)

        if self.oned_module == "transformer" and self.precomp_input_stage == "oned":
            self.transformer_layers = nn.ModuleList([
                DualInputTransformerLayer(d_model=d_model,
                                attention_heads=attention_heads,
                                ff_dim=ff_dim,
                                attention_dropout_rate=attention_dropout_rate,
                                dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ])
        elif self.oned_module == "transformer" and self.precomp_input_stage == "none":
            self.transformer_layers = nn.ModuleList([
                SingleInputTransformerLayer(d_model=d_model,
                                attention_heads=attention_heads,
                                ff_dim=ff_dim,
                                attention_dropout_rate=attention_dropout_rate,
                                dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ])
        else:
            raise ValueError(f"unsupported oned_module and precomp_input_stage combination, got: {self.oned_module} and {self.precomp_input_stage}, should be transformer and oned or none")
        self.transformer_final_layer = nn.Linear(d_model, output_size)

        self._output_size = output_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor, precomp_x: torch.Tensor = None, precomp_x_lengths: torch.Tensor = None):
        if self.precomp_input_stage != "none":
            if precomp_x is None:
                raise ValueError("Precomputed frame-level features are required for this encoder")
            precomp_x_dim = precomp_x.size()[-1]
        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        x = x.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)

        # the fcwSKA block
        x = self.frt_conv1(x) # (B, 1, F, T) -> (B, ska_dim, F/2, T)
        x = self.relu(x)
        x = self.frt_bn1(x)
        x = self.frt_block1(x)
        x = self.frt_block2(x)
        x = self.frt_conv2(x) # (B, ska_dim, F/2, T) -> (B, ska_dim, F/4, T/2)
        x = self.relu(x)
        x = self.frt_bn2(x)
        x = x.reshape((x.size()[0], -1, x.size()[-1])) # (B, ska_dim, F/4, T/2) -> (B, ska_dim*F/4, T/2)

        if self.oned_module == "tdnn":
            if self.precomp_input_stage == "none":
                x = self.conv1(x) # (B, ska_dim*F/4, T/2) -> (B, ndim, T/2)
                x = self.relu(x)
                x = self.bn1(x)
                x1 = self.tdnn_layer1(x)
                x2 = self.tdnn_layer2(x + x1)
                x3 = self.tdnn_layer3(x + x1 + x2)
                x = self.tdnn_layer4(torch.cat((x1, x2, x3), dim=1)) # (B, 3*ndim, T/2) -> (B, output_size, T/2)
                x = self.relu(x)
                return x

            elif self.precomp_input_stage == "oned":
                raise NotImplementedError("Precomputed frame-level features are not supported for TDNN module")
        
        elif self.oned_module == "transformer":
            attn_weights = {}
            x = self.x_embedding_conv(x) # (B, ska_dim*F/4, T/2) -> (B, d_model, T/2)
            x = x.permute(0, 2, 1) # (B, d_model, T/2) -> (B, T/2, d_model)
            x = self.positional_encoding(x)

            if self.precomp_input_stage == "none":
                for i, layer in enumerate(self.transformer_layers):
                    x, self_attn_weights = layer(x)
                    attn_weights[f"layer_{i}_self_attn"] = self_attn_weights

                x = self.transformer_final_layer(x) # (B, T/2, d_model) -> (B, T/2, output_size)
                x = x.permute(0, 2, 1) # (B, T/2, output_size) -> (B, output_size, T/2)
                return x, attn_weights
            elif self.precomp_input_stage == "oned":
                precomp_x = precomp_x.permute(0, 2, 1)  # (B, T_feat, precomp_dim) -> (B, precomp_dim, T_feat)
                precomp_x = self.precomp_embedding_conv(precomp_x) # (B, precomp_dim, T_feat) -> (B, d_model, T_feat/2)
                precomp_x = precomp_x.permute(0, 2, 1)  # (B, d_model, T_feat/2) -> (B, T_feat/2, d_model)
                precomp_x = self.positional_encoding(precomp_x)

                for i, layer in enumerate(self.transformer_layers):
                    x, self_attn_weights, cross_attn_weights = layer(x, precomp_x, precomp_x_lengths)
                    attn_weights[f"layer_{i}_self_attn"] = self_attn_weights
                    attn_weights[f"layer_{i}_cross_attn"] = cross_attn_weights

                x = self.transformer_final_layer(x) # (B, T/2, d_model) -> (B, T/2, output_size)
                x = x.permute(0, 2, 1) # (B, T/2, output_size) -> (B, output_size, T/2)
                return x, attn_weights
            else:
                raise ValueError(f"unsupported precomp_input_stage, got: {self.precomp_input_stage}, should be none or oned")
                
        else:
            raise ValueError(f"unsupported oned_module, got: {self.oned_module}, should be tdnn or transformer")
