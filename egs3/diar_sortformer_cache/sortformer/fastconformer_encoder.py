"""NeMo-free port of NVIDIA's FastConformer encoder used by Sortformer.

This module reimplements (in plain PyTorch, with no dependency on the NeMo
toolkit) the subset of ``nemo.collections.asr.modules.ConformerEncoder`` that is
required for the *offline* Sortformer speaker-diarization model
(``nvidia/diar_sortformer_4spk-v1``):

* ``dw_striding`` convolutional subsampling (8x time reduction),
* relative-positional (Transformer-XL) multi-head self-attention,
* Conformer blocks (macaron FFN -> MHSA -> conv module -> macaron FFN).

The submodule / parameter names are chosen to mirror the Hugging Face
``SortformerOffline`` checkpoint (``fc_encoder.*``) so that converting the
released weights into this implementation is a near-identity key remap.  See
``egs3/librispeech_sortformer/diar/src/convert_hf_sortformer.py``.

Reference (Apache-2.0):
    NVIDIA/NeMo nemo/collections/asr/parts/submodules/{subsampling,
    conformer_modules, multi_head_attention}.py
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.asr.encoder.abs_encoder import AbsEncoder


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Compute the output length after ``repeat_num`` identical conv/pool layers.

    Applies the standard convolution output-length formula
    ``floor/ceil((L + all_paddings - kernel_size) / stride + 1)`` once per
    repeated layer. Used by :class:`ConvSubsampling` to track valid (non-padded)
    frame counts through the subsampling stack so masks stay correct.

    Args:
        lengths: Tensor of input lengths (any shape; cast to float internally).
        all_paddings: Total padding added across both sides (left + right).
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        ceil_mode: If True use ``ceil`` rounding, else ``floor`` (matches the
            ``ceil_mode`` of the conv/pool layers being modeled).
        repeat_num: Number of identical layers the input passes through.

    Returns:
        Tensor of output lengths with dtype ``torch.int``.

    Example:
        >>> calc_length(torch.tensor([100]), all_paddings=2, kernel_size=3,
        ...             stride=2, ceil_mode=False, repeat_num=3)
        tensor([13], dtype=torch.int32)
    """
    add_pad = all_paddings - kernel_size
    one = 1.0
    for _ in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class ConvSubsampling(nn.Module):
    """``dw_striding`` convolutional subsampling (factor must be a power of 2).

    For ``subsampling_factor=8`` the layer stack is::

        Conv2d(1, C, 3, stride=2, pad=1)            # full conv
        ReLU
        Conv2d(C, C, 3, stride=2, pad=1, groups=C)  # depthwise
        Conv2d(C, C, 1)                             # pointwise
        ReLU
        Conv2d(C, C, 3, stride=2, pad=1, groups=C)  # depthwise
        Conv2d(C, C, 1)                             # pointwise
        ReLU

    followed by a ``Linear(C * F', d_model)`` projection, where ``F'`` is the
    sub-sampled frequency dimension.

    The forward pass reduces the time axis by ``subsampling_factor`` (8x by
    default) and projects the (channel, frequency) plane to ``feat_out``
    (``d_model``). The ``dw_striding`` design (one full conv followed by
    depthwise+pointwise pairs) mirrors the released Sortformer checkpoint, so
    converting weights is a near-identity key remap.

    Args:
        subsampling_factor: Time reduction factor; must be a power of 2.
        feat_in: Number of input feature (mel) bins.
        feat_out: Output dimension of the final linear projection (``d_model``).
        conv_channels: Channel count used inside the conv stack.

    Raises:
        ValueError: If ``subsampling_factor`` is not a multiple of 2.

    Example:
        >>> sub = ConvSubsampling(8, feat_in=80, feat_out=512, conv_channels=256)
        >>> x = torch.randn(2, 200, 80)
        >>> y, ylens = sub(x, torch.tensor([200, 160]))
        >>> y.shape  # time reduced ~8x, projected to d_model
        torch.Size([2, 25, 512])
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        conv_channels: int,
    ):
        super().__init__()
        if subsampling_factor % 2 != 0:
            raise ValueError("subsampling_factor should be a multiple of 2")
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self._feat_in = feat_in
        self._feat_out = feat_out
        self._conv_channels = conv_channels
        self._kernel_size = 3
        self._stride = 2
        self._left_padding = 1
        self._right_padding = 1
        self._ceil_mode = False

        layers = []
        in_channels = 1
        # Layer 0: full Conv2d
        layers.append(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1)
        )
        in_channels = conv_channels
        layers.append(nn.ReLU(True))
        # Remaining stages: depthwise + pointwise
        for _ in range(self._sampling_num - 1):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                )
            )
            layers.append(
                nn.Conv2d(
                    in_channels, conv_channels, kernel_size=1, stride=1, padding=0
                )
            )
            layers.append(nn.ReLU(True))
            in_channels = conv_channels
        self.layers = nn.Sequential(*layers)

        out_length = calc_length(
            torch.tensor(feat_in, dtype=torch.float),
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.linear = nn.Linear(conv_channels * int(out_length), feat_out)

    def get_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Map input frame lengths to sub-sampled output lengths."""
        return calc_length(
            lengths,
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subsample and project features.

        Args:
            x: Input features ``(B, T, feat_in)``.
            lengths: Valid frame counts ``(B,)`` before subsampling.

        Returns:
            Tuple ``(out, out_lengths)`` where ``out`` is ``(B, T', feat_out)``
            and ``out_lengths`` are the corresponding sub-sampled lengths.
        """
        out_lengths = self.get_out_lengths(lengths)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.layers(x)  # (B, C, T', F')
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).reshape(b, t, c * f))  # (B, T', d_model)
        return x, out_lengths


class RelPositionalEncoding(nn.Module):
    """Transformer-XL relative positional encoding (NeMo convention).

    ``forward`` returns ``(x * xscale, pos_emb)`` where ``pos_emb`` has shape
    ``(1, 2*T-1, d_model)`` ordered from position ``T-1`` down to ``-(T-1)``.
    The positional table is a non-persistent buffer (regenerated on the fly),
    matching the released checkpoint which does not store it.

    Args:
        d_model: Model (embedding) dimension.
        dropout_rate: Dropout applied to the (scaled) input embeddings.
        max_len: Maximum sequence length pre-computed for the positional table;
            it is extended automatically if a longer input is seen.
        xscale: Optional scale applied to the input (typically
            ``sqrt(d_model)``); ``None`` disables scaling.

    Example:
        >>> pe = RelPositionalEncoding(512, dropout_rate=0.1, xscale=512 ** 0.5)
        >>> x = torch.randn(2, 30, 512)
        >>> x_scaled, pos_emb = pe(x)
        >>> pos_emb.shape  # (1, 2T-1, d_model)
        torch.Size([1, 59, 512])
    """

    def __init__(
        self,
        d_model: int,
        dropout_rate: float,
        max_len: int = 5000,
        xscale: Optional[float] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(dropout_rate)
        self.max_len = max_len
        self.pe = None
        self.extend_pe(max_len, torch.device("cpu"), torch.float32)

    def create_pe(self, positions: torch.Tensor, dtype):
        """Build the sinusoidal table ``self.pe`` for the given positions."""
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=torch.float32, device=positions.device
            )
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(dtype)
        self.pe = pe

    def extend_pe(self, length: int, device, dtype):
        """Grow the positional table to cover ``length`` if it is too short.

        The table needs ``2*length - 1`` entries (positions ``length-1`` down to
        ``-(length-1)``). If the cached table is already large enough it is only
        moved to the requested device/dtype; otherwise it is rebuilt.
        """
        needed = 2 * length - 1
        if self.pe is not None and self.pe.size(1) >= needed:
            self.pe = self.pe.to(device=device, dtype=dtype)
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions, dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale the input and slice the relative-position embedding.

        Args:
            x: Input embeddings ``(B, T, d_model)``.

        Returns:
            Tuple ``(x_out, pos_emb)`` where ``x_out`` is the (optionally scaled
            and dropped-out) input and ``pos_emb`` is ``(1, 2*T-1, d_model)``.
        """
        self.extend_pe(x.size(1), x.device, x.dtype)
        if self.xscale:
            x = x * self.xscale
        input_len = x.size(1)
        center = self.pe.size(1) // 2 + 1
        start = center - input_len
        end = center + input_len - 1
        pos_emb = self.pe[:, start:end]
        return self.dropout(x), pos_emb


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with Transformer-XL relative positional encoding.

    Implements the Transformer-XL decomposition of the attention score into a
    content term (``matrix_ac``, query+``bias_u`` against keys) and a
    position term (``matrix_bd``, query+``bias_v`` against the relative-position
    projection, then :meth:`rel_shift`-ed into alignment).

    Parameter names match the HF checkpoint::

        q_proj, k_proj, v_proj, o_proj, relative_k_proj, bias_u, bias_v

    Args:
        n_head: Number of attention heads (must divide ``n_feat``).
        n_feat: Model dimension.
        dropout_rate: Dropout on the attention weights.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.q_proj = nn.Linear(n_feat, n_feat)
        self.k_proj = nn.Linear(n_feat, n_feat)
        self.v_proj = nn.Linear(n_feat, n_feat)
        self.o_proj = nn.Linear(n_feat, n_feat)
        self.relative_k_proj = nn.Linear(n_feat, n_feat, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.dropout = nn.Dropout(dropout_rate)

    def forward_qkv(self, query, key, value):
        """Project and reshape inputs into ``(B, h, T, d_k)`` query/key/value."""
        n = query.size(0)
        q = self.q_proj(query).view(n, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(n, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(n, -1, self.h, self.d_k).transpose(1, 2)
        return q, k, v

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift the position-score matrix so relative offsets align with keys.

        Pads, reshapes and slices ``matrix_bd`` (the Transformer-XL trick) so
        that entry ``(i, j)`` holds the score for relative offset ``i - j``.
        """
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, x, pos_emb, mask):
        """Compute relative-position multi-head self-attention.

        Args:
            x: Input sequence ``(B, T, d_model)`` (used as query, key, value).
            pos_emb: Relative-position embedding ``(1, 2*T-1, d_model)`` from
                :class:`RelPositionalEncoding`.
            mask: Boolean attention mask ``(B, T, T)`` where True marks
                disallowed (key, query) pairs; ``None`` for full attention.

        Returns:
            Attention output ``(B, T, d_model)``.
        """
        q, k, v = self.forward_qkv(x, x, x)
        q = q.transpose(1, 2)  # (B, T, h, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.relative_k_proj(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (B, h, 2T-1, d_k)

        q_with_bias_u = (q + self.bias_u).transpose(1, 2)  # (B, h, T, d_k)
        q_with_bias_v = (q + self.bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k  # (B, h, T, T)

        return self._forward_attention(v, scores, mask)

    def _forward_attention(self, value, scores, mask):
        """Mask, softmax and apply attention weights, then output-project."""
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T)
            scores = scores.masked_fill(mask, -10000.0)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (B, h, T, d_k)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.o_proj(x)


class ConformerFeedForward(nn.Module):
    """Position-wise feed-forward module (Linear -> SiLU -> dropout -> Linear)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ConformerConvolution(nn.Module):
    """Conformer conv module: pointwise -> GLU -> depthwise -> BN -> SiLU -> pointwise."""

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)  # (B, d, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    """A single Conformer block (macaron-FFN, MHSA, conv, macaron-FFN).

    Each sub-module uses pre-LayerNorm with residual connections; the two FFNs
    are scaled by ``fc_factor=0.5`` (macaron style) and a final LayerNorm is
    applied. The self-attention is :class:`RelPositionMultiHeadAttention`, which
    :class:`FastConformerEncoder` may swap for an efficient sliding-window
    variant when ``att_context_size`` is set (detected at runtime via the
    ``is_local`` attribute in :meth:`forward`).

    Args:
        d_model: Model dimension.
        d_ff: Hidden size of the feed-forward modules.
        n_heads: Number of attention heads.
        conv_kernel_size: Depthwise conv kernel size (must be odd).
        dropout: Dropout for FFN/conv and residual paths.
        dropout_att: Dropout on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel_size: int,
        dropout: float,
        dropout_att: float,
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout_att)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, att_mask, pos_emb, pad_mask, n_global=0):
        """Apply the Conformer block.

        Args:
            x: Input ``(B, T, d_model)``.
            att_mask: Full-attention mask ``(B, T, T)`` (True = masked); unused
                on the sliding-window path.
            pos_emb: Relative-position embedding for the attention module.
            pad_mask: Padding mask ``(B, T)`` (True = padded) used to zero
                padded positions before the depthwise conv and as the key mask
                on the sliding-window path.
            n_global: Number of leading frames (speaker-cache prefix) that stay
                globally attended on the sliding-window path; ignored otherwise.

        Returns:
            Block output ``(B, T, d_model)``.
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if getattr(self.self_attn, "is_local", False):
            # Efficient O(N*W) sliding-window attention; cache prefix is global.
            x = self.self_attn(x, pos_emb=pos_emb, pad_mask=pad_mask, n_global=n_global)
        else:
            x = self.self_attn(x, pos_emb=pos_emb, mask=att_mask)
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        return self.norm_out(residual)


class FastConformerEncoder(AbsEncoder):
    """FastConformer encoder (offline) for Sortformer.

    Stacks :class:`ConvSubsampling` (8x time reduction by default),
    :class:`RelPositionalEncoding`, and ``n_layers`` :class:`ConformerLayer`
    blocks. Subclasses :class:`AbsEncoder`, so :meth:`output_size` reports
    ``d_model``.

    Input: log-mel features ``(B, T, feat_in)`` with ``ilens``.
    Output: ``(B, T', d_model)`` with sub-sampled lengths (8x by default).

    Attention modes:
        * Full attention (default, ``att_context_size=None``): an O(T^2) mask is
          built per forward and standard relative-position attention is used.
        * Efficient local attention (``att_context_size=(left, right)``): the
          attention modules are replaced at construction time by O(T*W)
          sliding-window variants (Longformer-style chunked rel-pos). The
          replacement loads the original weights as-is, so checkpoints stay
          compatible. Each side may be ``-1`` for "unlimited on that side", and
          window units are sub-sampled (80-ms) frames.

    Streaming / speaker cache:
        :meth:`pre_encode` exposes the subsampling stage so a streaming caller
        can cache pre-encoded embeddings. The ``full_prefix_len`` argument of
        :meth:`forward` (passed through as ``n_global`` to each layer) marks the
        leading frames (speaker cache + FIFO) that every query may always attend
        to, bypassing the local window.

    Args:
        feat_in: Number of input mel bins.
        d_model: Model dimension (also the output size).
        n_layers: Number of Conformer blocks.
        n_heads: Attention heads per block.
        ff_expansion_factor: ``d_ff = d_model * ff_expansion_factor``.
        subsampling_factor: Time reduction factor (power of 2).
        subsampling_conv_channels: Channels inside the subsampling stack.
        conv_kernel_size: Depthwise conv kernel size in each block (odd).
        dropout: Dropout for FFN/conv/residual paths.
        dropout_pre_encoder: Dropout in the positional encoding.
        dropout_att: Dropout on attention weights.
        xscaling: If True scale embeddings by ``sqrt(d_model)``.
        pos_emb_max_len: Pre-computed positional table length.
        att_context_size: Optional ``(left, right)`` local-attention window in
            sub-sampled frames; ``None`` keeps full attention.

    Example:
        >>> enc = FastConformerEncoder(feat_in=80, d_model=512, n_layers=18)
        >>> x = torch.randn(2, 800, 80)
        >>> out, olens, _ = enc(x, torch.tensor([800, 640]))
        >>> out.shape  # time reduced 8x
        torch.Size([2, 100, 512])
        >>> # Streaming with a 188-frame global prefix (speaker cache + FIFO)
        >>> enc_local = FastConformerEncoder(att_context_size=(188, 13))
        >>> emb, elens = enc_local.pre_encode(x, torch.tensor([800, 640]))
        >>> out, olens, _ = enc_local(
        ...     emb, elens, bypass_pre_encode=True, full_prefix_len=188)
    """

    def __init__(
        self,
        feat_in: int = 80,
        d_model: int = 512,
        n_layers: int = 18,
        n_heads: int = 8,
        ff_expansion_factor: int = 4,
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        conv_kernel_size: int = 9,
        dropout: float = 0.1,
        dropout_pre_encoder: float = 0.1,
        dropout_att: float = 0.1,
        xscaling: bool = True,
        pos_emb_max_len: int = 5000,
        att_context_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self._output_size = d_model
        self.subsampling_factor = subsampling_factor
        # Local attention window [left, right] (in 80-ms frames); -1 = unlimited
        # on that side; None = full attention. In streaming, frames in the
        # speaker-cache prefix are ALWAYS attendable regardless of this window.
        self.att_context_size = att_context_size
        d_ff = d_model * ff_expansion_factor

        self.subsampling = ConvSubsampling(
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
        )
        xscale = math.sqrt(d_model) if xscaling else None
        self.pos_enc = RelPositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_pre_encoder,
            max_len=pos_emb_max_len,
            xscale=xscale,
        )
        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
                for _ in range(n_layers)
            ]
        )

        if att_context_size is not None:
            # Efficient O(N*W) sliding-window attention (Longformer-style chunked
            # rel-pos). Same weights as full attention -> checkpoints load as-is.
            # Lazy import avoids a circular import with sliding_window_attention.
            from .sliding_window_attention import (
                LocalAttRelPositionalEncoding,
                RelPositionLocalAttention,
            )

            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size,
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=xscale,
            )
            for layer in self.layers:
                local = RelPositionLocalAttention(
                    n_heads, d_model, dropout_att, att_context_size
                )
                local.load_state_dict(layer.self_attn.state_dict())
                layer.self_attn = local

    def output_size(self) -> int:
        """Return the encoder output dimension (``d_model``)."""
        return self._output_size

    def pre_encode(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run subsampling only: log-mel ``(B, T, feat_in)`` -> ``(B, T', d_model)``.

        Used by streaming inference: the speaker cache stores these pre-encode
        embeddings and re-feeds them through the conformer layers each chunk
        (call :meth:`forward` with ``bypass_pre_encode=True``).

        Args:
            x: Log-mel features ``(B, T, feat_in)``.
            lengths: Valid frame counts ``(B,)``.

        Returns:
            Tuple ``(emb, out_lengths)`` with ``emb`` of shape
            ``(B, T', d_model)`` and the sub-sampled lengths.
        """
        return self.subsampling(x, lengths)

    @staticmethod
    def _create_masks(lengths: torch.Tensor, max_len: int):
        """Build full-attention masks ``(pad_mask, att_mask)``.

        Returns a padding mask ``(B, T)`` and a square attention mask
        ``(B, T, T)`` (both True = masked). Local windowing is handled
        efficiently inside the sliding-window attention, not via an O(T^2) mask.
        """
        device = lengths.device
        valid = torch.arange(max_len, device=device).expand(
            lengths.size(0), max_len
        ) < lengths.unsqueeze(1)
        pad_mask = ~valid  # True = padded
        att_valid = valid.unsqueeze(1) & valid.unsqueeze(2)  # (B, T, T)
        att_mask = ~att_valid  # True = masked
        return pad_mask, att_mask

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        bypass_pre_encode: bool = False,
        full_prefix_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the conformer encoder.

        Args:
            xs_pad: ``(B, T, feat_in)`` log-mel features when
                ``bypass_pre_encode=False``; otherwise ``(B, T', d_model)``
                pre-encoded embeddings (subsampling already applied).
            ilens: Valid frame counts ``(B,)``; interpreted as input lengths
                (pre-subsampling) or as ``T'`` lengths when
                ``bypass_pre_encode=True``.
            prev_states: Unused; kept for the :class:`AbsEncoder` interface.
            bypass_pre_encode: If True, skip subsampling and treat ``xs_pad`` as
                already pre-encoded embeddings (see :meth:`pre_encode`).
            full_prefix_len: Number of leading frames (speaker cache + FIFO) that
                every query may always attend to, bypassing ``att_context_size``
                (forwarded as ``n_global`` to each layer; only meaningful on the
                local-attention path).

        Returns:
            Tuple ``(out, olens, None)`` where ``out`` is ``(B, T', d_model)``,
            ``olens`` are the (clamped) sub-sampled lengths, and the third item
            is always ``None`` (no recurrent state).
        """
        if bypass_pre_encode:
            x, olens = xs_pad, ilens
        else:
            x, olens = self.subsampling(xs_pad, ilens)
        x, pos_emb = self.pos_enc(x)
        max_len = x.size(1)
        olens = olens.clamp(max=max_len)
        local = self.att_context_size is not None
        if local:
            # Efficient sliding-window path: only the padding mask is needed; the
            # band kernel handles windowing, and the prefix is global (n_global).
            pad_mask = ~(
                torch.arange(max_len, device=olens.device).expand(
                    olens.size(0), max_len
                )
                < olens.unsqueeze(1)
            )
            att_mask = None
        else:
            pad_mask, att_mask = self._create_masks(olens, max_len)
        for layer in self.layers:
            x = layer(
                x,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                n_global=full_prefix_len,
            )
        return x, olens, None
