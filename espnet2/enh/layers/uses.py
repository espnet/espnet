import warnings

import torch
import torch.nn as nn

from espnet2.enh.layers.dptnet import ImprovedTransformerLayer as SingleTransformer
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm
from espnet2.torch_utils.get_layer_from_string import get_layer


class USES(nn.Module):
    """Unconstrained Speech Enhancement and Separation (USES) Network.

    Reference:
        [1] W. Zhang, K. Saijo, Z.-Q., Wang, S. Watanabe, and Y. Qian,
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
                the second group is used for denoising with dereverberation,
        rnn_type (str): type of the RNN cell in the improved Transformer layer.
        hidden_size (int): hidden dimension of the RNN cell.
        att_heads (int): number of attention heads in Transformer.
        dropout (float): dropout ratio. Default is 0.
        activation (str): non-linear activation function applied in each block.
        bidirectional (bool): whether the RNN layers are bidirectional.
        norm_type (str): normalization type in the improved Transformer layer.
        ch_mode (str): mode of channel modeling.
            Select from "att" and "tac".
        ch_att_dim (int): dimension of the channel attention.
        eps (float): epsilon for layer normalization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        bottleneck_size=64,
        num_blocks=6,
        num_spatial_blocks=3,
        segment_size=64,
        memory_size=20,
        memory_types=1,
        # Transformer-related arguments
        rnn_type="lstm",
        hidden_size=128,
        att_heads=4,
        dropout=0.0,
        activation="relu",
        bidirectional=True,
        norm_type="cLN",
        ch_mode="att",
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
        assert ch_mode in ("att", "tac"), ch_mode
        self.atf_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.atf_blocks.append(
                ATFBlock(
                    input_size=bottleneck_size,
                    rnn_type=rnn_type,
                    hidden_size=hidden_size,
                    att_heads=att_heads,
                    dropout=dropout,
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
            # single group of memory tokens (only used to provide history information)
            # (B=1, C=1, bottleneck_size, F=1, memory_size)
            self.memory_tokens = nn.Parameter(
                torch.randn(1, 1, bottleneck_size, 1, memory_size)
            )
        else:
            # >1 groups of memory tokens (used to also control processing behavior)
            self.memory_tokens = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 1, bottleneck_size, 1, memory_size))
                    for _ in range(memory_types)
                ]
            )

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(bottleneck_size, output_size, 1)
        )

    def forward(self, input, ref_channel=None, mem_idx=None):
        """USES forward.

        Args:
            input (torch.Tensor): input feature (batch, mics, input_size, freq, time)
            ref_channel (None or int): index of the reference channel.
                if None, simply average all channels.
                if int, take the specified channel instead of averaging.
            mem_idx (None or int): index of the memory token group.
                if None, use the only group of memory tokens in the model.
                if int, use the specified group from multiple existing groups.
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
            # pad the last segment if necessary
            output = nn.functional.pad(output, (0, self.segment_size - res))
            num_seg += 1
        if self.training and num_seg < 2:
            warnings.warn(f"The input is too short for training: {T}")
        output = output.reshape(B, C, bn, F, num_seg, self.segment_size)

        # Segment-by-segment processing for memory-efficient processing
        ret = []
        mem = None
        for n in range(num_seg):
            out = output[..., n, :]
            if mem is None:
                # initialize memory tokens for the first segment
                if mem_idx is not None:
                    mem = self.memory_tokens[mem_idx].repeat(B, C, 1, F, 1)
                elif self.memory_types > 1:
                    mem = self.memory_tokens[0].repeat(B, C, 1, F, 1)
                else:
                    mem = self.memory_tokens.repeat(B, C, 1, F, 1)
                out = torch.cat([mem, out], dim=-1)
            else:
                # reuse memory tokens from the last segment
                if mem.size(1) < C:
                    mem = mem.repeat(1, C // mem.size(1), 1, 1, 1)
                out = torch.cat([mem, out], dim=-1)
            for block in self.atf_blocks:
                out = block(out, ref_channel=ref_channel)
            mem, out = out[..., : self.memory_size], out[..., self.memory_size :]
            ret.append(out)

        output = torch.cat(ret, dim=-1)[..., :T]
        with torch.amp.autocast("cuda", enabled=False):
            output = self.output(output.mean(1))  # B, output_size, F, T
        return output


class ATFBlock(nn.Module):
    def __init__(
        self,
        input_size,
        rnn_type="lstm",
        hidden_size=128,
        att_heads=4,
        dropout=0.0,
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
            rnn_type (str): type of the RNN cell in the improved Transformer layer.
            hidden_size (int): hidden dimension of the RNN cell.
            att_heads (int): number of attention heads in Transformer.
            dropout (float): dropout ratio. Default is 0.
            activation (str): non-linear activation function applied in each block.
            bidirectional (bool): whether the RNN layers are bidirectional.
            norm_type (str): normalization type in the improved Transformer layer.
            ch_mode (str): mode of channel modeling. Select from "att" and "tac".
            ch_att_dim (int): dimension of the channel attention.
            eps (float): epsilon for layer normalization.
            with_channel_modeling (bool): whether to use channel modeling.
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
            else:
                raise NotImplementedError(ch_mode)

    def forward(self, input, ref_channel=None):
        """Forward.

        Args:
            input (torch.Tensor): feature sequence (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
                if None, simply average all channels.
                if int, take the specified channel instead of averaging.
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
        output = self.freq_path_process(output)
        output = self.time_path_process(output)
        output = output.contiguous().reshape(B, C, N, F, T)
        if self.with_channel_modeling:
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


class ChannelAttention(nn.Module):
    def __init__(
        self, input_dim, att_heads=4, att_dim=256, activation="relu", eps=1e-5
    ):
        """Channel Attention module.

        Args:
            input_dim (int): dimension of the input feature.
            att_heads (int): number of attention heads in self-attention.
            att_dim (int): projection dimension for query and key before self-attention.
            activation (str): non-linear activation function.
            eps (float): epsilon for layer normalization.
        """
        super().__init__()
        self.att_heads = att_heads
        self.att_dim = att_dim
        self.activation = activation
        assert input_dim % att_heads == 0, (input_dim, att_heads)
        self.attn_conv_Q = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            get_layer(activation)(),
            LayerNormalization(att_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_conv_K = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            get_layer(activation)(),
            LayerNormalization(att_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_conv_V = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            get_layer(activation)(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_concat_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            get_layer(activation)(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x, ref_channel=None):
        """ChannelAttention Forward.

        Args:
            x (torch.Tensor): input feature (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output feature (batch, C, N, freq, time)
        """
        B, C, N, F, T = x.shape
        batch = x.permute(0, 4, 1, 3, 2)  # [B, T, C, F, N]

        Q = (
            self.attn_conv_Q(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D]
        K = (
            self.attn_conv_K(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D]
        V = (
            self.attn_conv_V(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D']

        emb_dim = V.size(-2) * V.size(-1)
        attn_mat = torch.einsum("bhtcfn,bhtefn->bhce", Q / T, K / emb_dim**0.5)
        attn_mat = nn.functional.softmax(attn_mat, dim=-1)  # [B, head, C, C]
        V = torch.einsum("bhce,bhtefn->bhtcfn", attn_mat, V)  # [B, head, T, C, F, D']

        batch = torch.cat(V.unbind(dim=1), dim=-1)  # [B, T, C, F, D]
        batch = self["attn_concat_proj"](batch)  # [B, T, C, F, N]

        return batch.permute(0, 2, 4, 3, 1) + x


class ChannelTAC(nn.Module):
    def __init__(self, input_dim, eps=1e-5):
        """Channel Transform-Average-Concatenate (TAC) module.

        Args:
            input_dim (int): dimension of the input feature.
            eps (float): epsilon for layer normalization.
        """
        super().__init__()
        hidden_dim = input_dim * 3
        self.transform = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.PReLU())
        self.average = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU())
        self.concat = nn.Sequential(
            nn.Linear(hidden_dim * 2, input_dim),
            nn.PReLU(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x, ref_channel=None):
        """ChannelTAC Forward.

        Args:
            x (torch.Tensor): input feature (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output feature (batch, C, N, freq, time)
        """
        batch = x.contiguous().permute(0, 4, 1, 3, 2)  # [B, T, C, F, N]
        out = self.transform(batch)
        out_mean = self.average(out.mean(dim=2, keepdim=True)).expand_as(out)
        out = self.concat(torch.cat([out, out_mean], dim=-1))
        out = out.permute(0, 2, 4, 3, 1) + x
        return out


class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
