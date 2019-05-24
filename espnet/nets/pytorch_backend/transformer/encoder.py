import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Encoder(torch.nn.Module):
    """Transformer encoder module

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    """

    def __init__(self, idim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer="conv2d"):
        super(Encoder, self).__init__()
        if input_layer == "linear":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                PositionalEncoding(attention_dim, dropout_rate)
            )
        elif input_layer == "conv2d":
            self.input_layer = Conv2dSubsampling(idim, attention_dim, dropout_rate)
        elif input_layer == "embed":
            self.input_layer = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim),
                PositionalEncoding(attention_dim, dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.input_layer = torch.nn.Sequential(
                input_layer,
                PositionalEncoding(attention_dim, dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate
            )
        )
        self.norm = LayerNorm(attention_dim)

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
