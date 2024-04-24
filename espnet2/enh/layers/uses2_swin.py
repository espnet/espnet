import math

import torch
import torch.nn as nn

from espnet2.enh.layers.tcn import ChannelwiseLayerNorm
from espnet2.enh.layers.swin_transformer import BasicLayer
from espnet2.enh.layers.uses import ChannelAttention, ChannelTAC, LayerNormalization

EPS = torch.finfo(torch.float32).eps
if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class USES2_Swin(nn.Module):
    """Unconstrained Speech Enhancement and Separation v2 (USES2-Swin) Network.

    Reference:
        [1] W. Zhang, J.-w. Jung, and Y. Qian, “Improving Design of Input
            Condition Invariant Speech Enhancement,” in Proc. ICASSP, 2024.
        [2] W. Zhang, K. Saijo, Z.-Q., Wang, S. Watanabe, and Y. Qian,
            “Toward Universal Speech Enhancement for Diverse Input Conditions,”
            in Proc. ASRU, 2023.

    args:
        input_size (int): dimension of the input feature.
        output_size (int): dimension of the output.
        bottleneck_size (int): dimension of the bottleneck feature.
            Must be a multiple of `att_heads`.
        num_blocks (int): number of ResSwinBlock blocks.
        num_spatial_blocks (int): number of ResSwinBlock blocks with channel modeling.
        swin_block_depth (Tuple[int]): depth of each ResSwinBlock.
        input_resolution (tuple): frequency and time dimension of the input feature.
            Only used for efficient training.
            Should be close to the actual spectrum size (F, T) of training samples.
        window_size (tuple): size of the Time-Frequency window in Swin-Transformer.
        mlp_ratio (int): ratio of the MLP hidden size to embedding size in BasicLayer.
        qkv_bias (bool): If True, add a learnable bias to query, key, value in
            BasicLayer.
        qk_scale (float): Override default qk scale of head_dim ** -0.5 in BasicLayer
            if set.
        att_heads (int): number of attention heads in Transformer.
        dropout (float): dropout ratio in BasicLayer. Default is 0.
        att_dropout (float): attention dropout ratio in BasicLayer.
        drop_path (float): drop-path ratio in BasicLayer.
        activation (str): non-linear activation function applied in each block.
        use_checkpoint (bool): whether to use checkpointing to save memory.
        ch_mode (str): mode of channel modeling. Select from "att", "tac", and "att_tac"
        ch_att_dim (int): dimension of the channel attention.
        eps (float): epsilon for layer normalization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        bottleneck_size=64,
        num_blocks=3,
        num_spatial_blocks=2,
        # Transformer-related arguments
        swin_block_depth=(4, 4, 4, 4),
        input_resolution=(130, 256),
        window_size=(10, 8),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        att_heads=4,
        dropout=0.0,
        att_dropout=0.0,
        drop_path=0.0,
        activation="relu",
        use_checkpoint=False,
        ch_mode="att_tac",
        ch_att_dim=256,
        eps=1e-5,
    ):
        super().__init__()

        # [B, input_size, T] -> [B, input_size, T]
        self.layer_norm = ChannelwiseLayerNorm(input_size)
        # [B, input_size, T] -> [B, bottleneck_size, T]
        self.bottleneck_conv1x1 = nn.Conv1d(input_size, bottleneck_size, 1, bias=False)

        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.output_size = output_size

        assert num_blocks >= num_spatial_blocks, (num_blocks, num_spatial_blocks)
        if isinstance(ch_mode, str):
            assert ch_mode in ("att", "tac", "att_tac"), ch_mode
            ch_mode = [ch_mode for _ in range(num_spatial_blocks)]
        else:
            assert isinstance(ch_mode, (tuple, list))
            assert len(ch_mode) == num_spatial_blocks, (ch_mode, num_spatial_blocks)
            assert all([ch in ("att", "tac", "att_tac") for ch in ch_mode])
        if not isinstance(swin_block_depth, (list, tuple)):
            swin_block_depth = tuple(swin_block_depth)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            opt = {"ch_mode": ch_mode[i]} if i < num_spatial_blocks else {}
            self.blocks.append(
                ResSwinBlock(
                    input_size=bottleneck_size,
                    input_resolution=input_resolution,
                    swin_block_depth=swin_block_depth,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    att_dropout=att_dropout,
                    drop_path=drop_path,
                    activation=activation,
                    att_heads=att_heads,
                    use_checkpoint=use_checkpoint,
                    ch_att_dim=ch_att_dim,
                    eps=eps,
                    with_channel_modeling=i < num_spatial_blocks,
                    **opt,
                )
            )

        # output layer
        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(bottleneck_size, output_size, 1)
        )

    def forward(self, input, ref_channel=None):
        """USES2-Swin forward.

        Args:
            input (torch.Tensor): input feature (batch, mics, input_size, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output feature (batch, output_size, freq, time)
        """
        B, C, N, F, T = input.shape
        output = self.layer_norm(input.reshape(B * C, N, -1))
        # B, C, bn, F, T
        output = self.bottleneck_conv1x1(output).reshape(B, C, -1, F, T)

        for block in self.blocks:
            output = block(output, ref_channel=ref_channel)

        with torch.cuda.amp.autocast(enabled=False):
            output = self.output(output.mean(1))  # B, output_size, F, T
        return output


class ResSwinBlock(nn.Module):
    def __init__(
        self,
        input_size,
        input_resolution=(130, 256),
        swin_block_depth=(4, 4, 4, 4),
        window_size=(10, 8),
        mlp_ratio=2,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.0,
        att_dropout=0.0,
        drop_path=0.0,
        activation="relu",
        att_heads=4,
        use_checkpoint=False,
        ch_mode="att_tac",
        ch_att_dim=256,
        eps=1e-5,
        with_channel_modeling=True,
    ):
        """Container module for a single Residual Shifted-Window Transformer Block.

        Args:
            input_size (int): dimension of the input feature.
            input_resolution (tuple): frequency and time dimension of the input feature.
                Only used for efficient training.
                Should be close to the actual spectrum size (F, T) of training samples.
            swin_block_depth (Tuple[int]): depth of each ResSwinBlock.
            window_size (tuple): size of the Time-Frequency window in Swin-Transformer.
            mlp_ratio (int): ratio of the MLP hidden size to embedding size in
                BasicLayer.
            qkv_bias (bool): If True, add a learnable bias to query, key, value in
                BasicLayer.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 in BasicLayer
                if set.
            dropout (float): dropout ratio in BasicLayer. Default is 0.
            att_dropout (float): attention dropout ratio in BasicLayer. Default is 0.
            drop_path (float): drop-path ratio in BasicLayer. Default is 0.
            activation (str): non-linear activation function applied in each block.
            att_heads (int): number of attention heads.
            use_checkpoint (bool): whether to use checkpointing to save memory.
            ch_mode (str): mode of channel modeling.
                Select from "att", "tac" and "att_tac".
            ch_att_dim (int): dimension of the channel attention.
            eps (float): epsilon for layer normalization.
            with_channel_modeling (bool): whether to use channel attention.
        """
        super().__init__()

        self.input_size = input_size

        self.window_size = window_size
        self.freq_temporal_nn = nn.ModuleList(
            [
                BasicLayer(
                    dim=input_size,
                    input_resolution=input_resolution,
                    depth=depth,
                    num_heads=att_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=dropout,
                    attn_drop=att_dropout,
                    drop_path=drop_path,
                    use_checkpoint=use_checkpoint,
                )
                for depth in swin_block_depth
            ]
        )

        self.with_channel_modeling = with_channel_modeling
        self.ch_mode = ch_mode
        if with_channel_modeling:
            if ch_mode == "att":
                self.channel_nn = ChannelAttention(
                    input_dim=input_size,
                    att_heads=att_heads,
                    att_dim=ch_att_dim,
                    activation=activation,
                    eps=eps,
                )
            elif ch_mode == "tac":
                self.channel_nn = ChannelTAC(input_dim=input_size, eps=eps)
            elif ch_mode == "att_tac":
                self.channel_nn = ChannelAttentionTAC(
                    input_dim=input_size,
                    att_heads=att_heads,
                    att_dim=ch_att_dim,
                    activation=activation,
                    eps=eps,
                )
            else:
                raise NotImplementedError(ch_mode)

    def pad_to_window_multiples(self, input, window_size):
        """Pad the input feature to multiples of the window size.

        Args:
            input (torch.Tensor): input feature (batch, C, N, freq, time)
            window_size (tuple): size of the window (H, W).
        Returns:
            output (torch.Tensor): padded input feature (batch, C, N, n * H, m * W)
        """
        freq, time = input.shape[-2:]
        H, W = window_size
        n = math.ceil(freq / H)
        m = math.ceil(time / W)
        return nn.functional.pad(input, (0, m * W - time, 0, n * H - freq))

    def forward(self, input, ref_channel=None):
        """Forward.

        Args:
            input (torch.Tensor): feature sequence (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output sequence (batch, C, N, freq, time)
        """
        if not self.with_channel_modeling:
            if input.size(1) > 1 and ref_channel is not None:
                input = input[:, ref_channel].unsqueeze(1)
            else:
                input = input.mean(dim=1, keepdim=True)
        B, C, N, F, T = input.shape
        output = input.reshape(B * C, N, F, T).contiguous()

        # (batch, C, N, F2, T2)
        output = self.pad_to_window_multiples(output, self.window_size)
        F2, T2 = output.shape[-2:]
        # (batch * C, F2 * T2, N)
        output = output.reshape(B * C, N, F2 * T2).transpose(1, 2)

        for layer in self.freq_temporal_nn:
            output = layer(output, (F2, T2))

        output = output.contiguous().transpose(1, 2).reshape(B, C, N, F2, T2)
        output = output[..., :F, :T].contiguous()
        if self.with_channel_modeling and C > 1:
            output = self.channel_nn(output, ref_channel=ref_channel)
        return output


class ChannelAttentionTAC(nn.Module):
    def __init__(
        self, input_dim, att_heads=4, att_dim=256, activation="relu", eps=1e-5
    ):
        """Channel Transform-Attention-Concatenate (TAttC) module.

        Args:
            input_dim (int): dimension of the input feature.
            att_heads (int): number of attention heads in self-attention.
            att_dim (int): projection dimension for query and key before self-attention.
            activation (str): non-linear activation function.
            eps (float): epsilon for layer normalization.
        """
        super().__init__()
        hidden_dim = input_dim * 3
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.PReLU())
        self.attention = nn.Sequential(
            ChannelAttention(
                hidden_dim,
                att_heads=att_heads,
                att_dim=att_dim,
                activation=activation,
                eps=eps,
            ),
            nn.PReLU(),
        )
        self.concat = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.PReLU(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, ref_channel=None):
        """TAttC Forward.

        Args:
            x (torch.Tensor): input feature (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output feature (batch, C, N, freq, time)
        """
        batch = x.contiguous().permute(0, 4, 1, 3, 2)  # [B, T, C, F, N]
        out = self.transform(batch)
        out_att = self.attention(out.permute(0, 2, 4, 3, 1)).permute(0, 4, 1, 3, 2)
        out = self.concat(torch.cat([out, out_att], dim=-1))
        out = out.permute(0, 2, 4, 3, 1) + x
        return out
