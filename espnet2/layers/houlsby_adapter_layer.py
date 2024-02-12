import torch
import torch.nn as nn

try:
    import s3prl
    from s3prl.upstream.wav2vec2.wav2vec2_model import TransformerSentenceEncoderLayer
except Exception:
    s3prl = None


class Houlsby_Adapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        bottleneck: int,
    ):
        super(Houlsby_Adapter, self).__init__()

        self.houlsby_adapter = nn.Sequential(
            nn.Linear(input_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, input_size),
        )

    def forward(self, x):
        return self.houlsby_adapter(x)


class HoulsbyTransformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        bottleneck: int = 32,
        **kwargs,
    ) -> None:
        if s3prl is None:
            print("Error: S3PRL is not properly installed.")
            print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
            raise RuntimeError("Requiring S3PRL. ")
        super(HoulsbyTransformerSentenceEncoderLayer, self).__init__(**kwargs)

        self.adapter = Houlsby_Adapter(
            input_size=self.fc2.out_features,
            bottleneck=bottleneck,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))

            x = self.dropout2(x)
            x = self.fc2(x)

            # add adapter input
            houlsby_input = x

            layer_result = x

            x = self.dropout3(x)

            x = x + residual + self.adapter(houlsby_input)
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            houlsby_input = x

            layer_result = x

            x = self.dropout3(x)
            x = x + residual + self.adapter(houlsby_input)
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)
