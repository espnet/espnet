import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Encoder module

    :param int idim: input dim
    :param argparse.Namespace args: experiment config
    """

    def __init__(self, idim, args):
        super(Encoder, self).__init__()
        if args.transformer_input_layer == "linear":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(idim, args.adim),
                torch.nn.LayerNorm(args.adim),
                torch.nn.Dropout(args.dropout_rate),
                torch.nn.ReLU(),
                PositionalEncoding(args.adim, args.dropout_rate)
            )
        elif args.transformer_input_layer == "conv2d":
            self.input_layer = Conv2dSubsampling(idim, args.adim, args.dropout_rate)
        elif args.transformer_input_layer == "embed":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Embedding(idim, args.adim),
                PositionalEncoding(args.adim, args.dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + args.transformer_input_layer)

        self.encoders = repeat(
            args.elayers,
            lambda: EncoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.transformer_attn_dropout_rate),
                PositionwiseFeedForward(args.adim, args.eunits, args.dropout_rate),
                args.dropout_rate
            )
        )
        self.norm = LayerNorm(args.adim)

    def forward(self, x, mask):
        """Embed positions in tensor

        :param torch.Tensor x: input tensor
        :param torch.Tensor mask: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.input_layer, Conv2dSubsampling):
            x, mask = self.input_layer(x, mask)
        else:
            x = self.input_layer(x)
        x, mask = self.encoders(x, mask)
        return self.norm(x), mask
