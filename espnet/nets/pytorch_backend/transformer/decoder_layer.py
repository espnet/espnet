import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concate_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout_rate,
                 normalize_before=True, concate_after=False):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        if self.concate_after:
            self.concate_linear1 = nn.Linear(size + size, size)
            self.concate_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if self.concate_after:
            tgt_concat = torch.cat((tgt, self.self_attn(tgt, tgt, tgt, tgt_mask)), dim=-1)
            x = residual + self.concate_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt, tgt, tgt, tgt_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concate_after:
            x_concat = torch.cat(x, self.src_attn(x, memory, memory, memory_mask))
            x = residual + self.concate_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask, memory, memory_mask
