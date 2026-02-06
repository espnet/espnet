import math

import torch
import torch.nn as nn

from espnet2.enh.layers.dptnet import ImprovedTransformerLayer as SingleTransformer
from espnet2.enh.layers.swin_transformer import BasicLayer
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm
from espnet2.enh.layers.uses import ChannelAttention, ChannelTAC
from espnet2.enh.layers.uses2_swin import ChannelAttentionTAC

EPS = torch.finfo(torch.float32).eps
if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class USES2_Comp(nn.Module):
    """Unconstrained Speech Enhancement and Separation v2 (USES2-Comp) Network.

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
        num_blocks (int): number of processing blocks.
        num_spatial_blocks (int): number of processing blocks with channel modeling.
        segment_size (int): number of frames in each non-overlapping segment.
            This is used to segment long utterances into smaller segments for
            efficient processing.
        memory_size (int): group size of global memory tokens.
            The basic use of memory tokens is to store the history information from
            previous segments.
            The memory tokens are updated by the output of the last block after
            processing each segment.
        memory_types (int): numbre of memory token groups.
            Each group corresponds to a different type of processing, i.e.,
                the first group is used for denoising without dereverberation,
                the second group is used for denoising with dereverberation.
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
        dropout (float): dropout ratio. Default is 0.
        att_dropout (float): dropout ratio in attention in BasicLayer.
        drop_path (float): drop-path ratio in BasicLayer.
        use_checkpoint (bool): whether to use checkpointing to save memory.
        rnn_type (str): type of the RNN cell in the improved Transformer layer.
        hidden_size (int): hidden dimension of the RNN cell.
        activation (str): non-linear activation function applied in each block.
        bidirectional (bool): whether the RNN layers are bidirectional.
        norm_type (str): normalization type in the improved Transformer layer.
        ch_mode (str): mode of channel modeling.
            Select from "att", "tac", and "att_tac".
        ch_att_dim (int): dimension of the channel attention.
        eps (float): epsilon for layer normalization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        bottleneck_size=64,
        num_blocks=4,
        num_spatial_blocks=2,
        segment_size=64,
        memory_size=20,
        memory_types=1,
        # Transformer-related arguments
        input_resolution=(130, 64),
        window_size=(10, 8),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        att_heads=4,
        dropout=0.0,
        att_dropout=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        rnn_type="lstm",
        hidden_size=128,
        activation="relu",
        bidirectional=True,
        norm_type="cLN",
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
        assert ch_mode in ("att", "tac", "att_tac"), ch_mode
        self.atf_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.atf_blocks.append(
                ATFBlock(
                    input_size=bottleneck_size,
                    input_resolution=input_resolution,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=dropout,
                    att_dropout=att_dropout,
                    drop_path=drop_path,
                    use_checkpoint=use_checkpoint,
                    rnn_type=rnn_type,
                    hidden_size=hidden_size,
                    att_heads=att_heads,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                    ch_mode=ch_mode,
                    ch_att_dim=ch_att_dim,
                    eps=eps,
                    with_channel_modeling=i < num_spatial_blocks,
                )
            )

        self.segment_size = segment_size
        self.memory_size = memory_size
        self.memory_types = memory_types
        if memory_types == 1:
            # (B=1, C=1, bottleneck_size, F=1, memory_size)
            self.memory_tokens = nn.Parameter(
                torch.randn(1, 1, bottleneck_size, 1, memory_size)
            )
        else:
            self.memory_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 1, bottleneck_size, 1, memory_size))
                    for _ in range(memory_types)
                ]
            )

        # output layer
        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(bottleneck_size, output_size, 1)
        )

    def forward(self, input, ref_channel=None, mem_idx=None):
        """USES2-Comp forward.

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
        bn = output.size(2)

        # Divide the input into non-overlapping segments
        num_seg, res = divmod(T, self.segment_size)
        if res > 0:
            output = nn.functional.pad(output, (0, self.segment_size - res))
            num_seg += 1
        if self.training and num_seg < 2:
            raise ValueError(f"The input is too short for training: {T}")
        output = output.reshape(B, C, bn, F, num_seg, self.segment_size)

        # Segment-by-segment processing
        ret = []
        mem = None
        for n in range(num_seg):
            out = output[..., n, :]
            if mem is None:
                if mem_idx is not None:
                    out = torch.cat(
                        [self.memory_tokens[mem_idx].repeat(B, C, 1, F, 1), out], dim=-1
                    )
                elif self.memory_types > 1:
                    out = torch.cat(
                        [self.memory_tokens[0].repeat(B, C, 1, F, 1), out], dim=-1
                    )
                else:
                    out = torch.cat(
                        [self.memory_tokens.repeat(B, C, 1, F, 1), out], dim=-1
                    )
            else:
                # reuse memory tokens from the last segment if possible
                if mem.size(1) < C:
                    mem = mem.repeat(1, C // mem.size(1), 1, 1, 1)
                out = torch.cat([mem, out], dim=-1)
            for block in self.atf_blocks:
                out = block(out, ref_channel=ref_channel, mem_size=self.memory_size)
            mem, out = out[..., : self.memory_size], out[..., self.memory_size :]
            ret.append(out)

        output = torch.cat(ret, dim=-1)[..., :T]
        with torch.cuda.amp.autocast(enabled=False):
            output = self.output(output.mean(1))  # B, output_size, F, T
        return output


class ATFBlock(nn.Module):
    def __init__(
        self,
        input_size,
        input_resolution=(130, 64),
        window_size=(10, 8),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.0,
        att_dropout=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        rnn_type="lstm",
        hidden_size=128,
        att_heads=4,
        activation="relu",
        bidirectional=True,
        norm_type="cLN",
        ch_mode="att",
        ch_att_dim=256,
        eps=1e-5,
        with_channel_modeling=True,
    ):
        """Container module for a single Attentive Time-Frequency Block.

        Args:
            input_size (int): dimension of the input feature.
            input_resolution (tuple): frequency and time dimension of the input feature.
                Only used for efficient training.
                Should be close to the actual spectrum size (F, T) of training samples.
            window_size (tuple): size of the Time-Frequency window in Swin-Transformer.
            mlp_ratio (int): ratio of the MLP hidden size to embedding size in
                BasicLayer.
            qkv_bias (bool): If True, add a learnable bias to query, key, value in
                BasicLayer.
            qk_scale (float): Override default qk scale of head_dim ** -0.5 in
                BasicLayer if set.
            dropout (float): dropout ratio. Default is 0.
            att_dropout (float): attention dropout ratio in BasicLayer. Default is 0.
            drop_path (float): drop-path ratio in BasicLayer. Default is 0.
            use_checkpoint (bool): whether to use checkpointing to save memory.
            rnn_type (str): type of the RNN cell in the improved Transformer layer.
            hidden_size (int): hidden dimension of the RNN cell.
            att_heads (int): number of attention heads in Transformer.
            dropout (float): dropout ratio. Default is 0.
            activation (str): non-linear activation function applied in each block.
            bidirectional (bool): whether the RNN layers are bidirectional.
            norm_type (str): normalization type in the improved Transformer layer.
            ch_mode (str): mode of channel modeling.
                Select from "att", "tac", and "att_tac".
            ch_att_dim (int): dimension of the channel attention.
            eps (float): epsilon for layer normalization.
            with_channel_modeling (bool): whether to use channel attention.
        """
        super().__init__()

        kwargs = dict(
            rnn_type=rnn_type,
            input_size=input_size,
            att_heads=att_heads,
            hidden_size=hidden_size,
            dropout=dropout,
            activation="linear",
            bidirectional=bidirectional,
            norm=norm_type,
        )
        self.freq_nn = SingleTransformer(**kwargs)
        self.temporal_nn = SingleTransformer(**kwargs)
        self.tf_nn = BasicLayer(
            dim=input_size,
            input_resolution=input_resolution,
            depth=1,
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
        self.window_size = window_size

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

    def forward(self, input, ref_channel=None, mem_size=20):
        """Forward.

        Args:
            input (torch.Tensor): feature sequence (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
            mem_size (int): length of the memory tokens
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
        out = self.time_freq_process(output[..., mem_size:])
        output = torch.cat([output[..., :mem_size], out], dim=-1)
        output = self.freq_path_process(output)
        output = self.time_path_process(output)
        output = output.reshape(B, C, N, F, T)
        if self.with_channel_modeling and C > 1:
            output = self.channel_nn(output, ref_channel=ref_channel)
        return output

    def freq_path_process(self, x):
        batch, N, freq, time = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch * time, freq, N)
        x = self.freq_nn(x)
        x = x.reshape(batch, time, freq, N).permute(0, 3, 2, 1)
        return x.contiguous()

    def time_path_process(self, x):
        batch, N, freq, time = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * freq, time, N)
        x = self.temporal_nn(x)
        x = x.reshape(batch, freq, time, N).permute(0, 3, 1, 2)
        return x.contiguous()

    def time_freq_process(self, x):
        batch, N, freq, time = x.shape
        # (batch, N, F2, T2)
        x = self.pad_to_window_multiples(x, self.window_size)
        F2, T2 = x.shape[-2:]

        # (batch, F2 * T2, N)
        x = x.reshape(batch, N, F2 * T2).transpose(1, 2)
        x = self.tf_nn(x, (F2, T2))

        x = x.contiguous().transpose(1, 2).reshape(batch, N, F2, T2)
        return x[..., :freq, :time].contiguous()

    def pad_to_window_multiples(self, input, window_size):
        """Pad the input feature to multiples of the window size.

        Args:
            input (torch.Tensor): input feature (..., freq, time)
            window_size (tuple): size of the window (H, W).
        Returns:
            output (torch.Tensor): padded input feature (..., n * H, m * W)
        """
        freq, time = input.shape[-2:]
        H, W = window_size
        n = math.ceil(freq / H)
        m = math.ceil(time / W)
        return nn.functional.pad(input, (0, m * W - time, 0, n * H - freq))
