import torch

from .attention import MultiHeadedAttention
from .decoder_layer import DecoderLayer
from .embedding import PositionalEncoding
from .feedforward import PositionwiseFeedForward
from .layer_norm import LayerNorm
from .sequential import repeat


class Decoder(torch.nn.Module):
    def __init__(self, odim, args):
        super(Decoder, self).__init__()
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(odim, args.adim),
            PositionalEncoding(args.adim, args.dropout_rate)
        )
        self.decoders = repeat(
            args.dlayers,
            lambda: DecoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.transformer_attn_dropout_rate),
                MultiHeadedAttention(args.aheads, args.adim, args.transformer_attn_dropout_rate),
                PositionwiseFeedForward(args.adim, args.dunits, args.dropout_rate),
                args.dropout_rate
            )
        )
        self.output_norm = LayerNorm(args.adim)
        self.output_layer = torch.nn.Linear(args.adim, odim)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """forward decoder

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory_mask: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        x = self.output_layer(self.output_norm(x))
        return x, tgt_mask
