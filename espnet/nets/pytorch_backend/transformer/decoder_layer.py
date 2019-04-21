from torch import nn

from .layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        x = tgt
        nx = self.norm1(x)
        x = x + self.dropout(self.self_attn(nx, nx, nx, tgt_mask))
        nx = self.norm2(x)
        x = x + self.dropout(self.src_attn(nx, memory, memory, memory_mask))
        nx = self.norm3(x)
        return x + self.dropout(self.feed_forward(nx)), tgt_mask, memory, memory_mask
