import torch
import torch.nn as nn

try:
    import s3prl  # noqa
    from s3prl.upstream.wav2vec2.wav2vec2_model import TransformerSentenceEncoderLayer

    is_s3prl_available = True
except ImportError:
    is_s3prl_available = False


class Houlsby_Adapter(nn.Module):
    """
    Implements the Houlsby Adapter mechanism for model adaptation.

    The Houlsby Adapter is a lightweight module that allows for efficient
    parameterization of the model by adding a bottleneck layer between the
    input and output layers. It can be utilized to adapt pre-trained models
    to specific tasks without requiring full retraining.

    Attributes:
        bottleneck (int): The size of the bottleneck layer.
        houlsby_adapter (nn.Sequential): The sequential model containing
            the linear layers and activation function.

    Args:
        input_size (int): The size of the input features.
        bottleneck (int): The size of the bottleneck layer.

    Returns:
        torch.Tensor: The output of the adapter, which has the same size as
        the input.

    Examples:
        >>> adapter = Houlsby_Adapter(input_size=768, bottleneck=32)
        >>> input_tensor = torch.randn(10, 768)  # Batch size of 10
        >>> output_tensor = adapter(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 768])

    Raises:
        ValueError: If input_size or bottleneck is not a positive integer.
    """

    def __init__(
        self,
        input_size: int,
        bottleneck: int,
    ):
        super(Houlsby_Adapter, self).__init__()
        self.bottleneck = bottleneck
        self.houlsby_adapter = nn.Sequential(
            nn.Linear(input_size, self.bottleneck),
            nn.GELU(),
            nn.Linear(self.bottleneck, input_size),
        )

    def forward(self, x):
        """
            Applies the Houlsby Adapter to the input tensor.

        This method processes the input tensor `x` through the Houlsby Adapter,
        which consists of a linear layer followed by a GELU activation and another
        linear layer. The purpose of the adapter is to reduce the dimensionality
        of the input before passing it through the final layer, enabling
        efficient parameter usage in the model.

        Args:
            x (torch.Tensor): The input tensor to be processed. The expected shape
                is (batch_size, input_size).

        Returns:
            torch.Tensor: The output tensor after applying the Houlsby Adapter.
            The output shape will be the same as the input shape (batch_size,
            input_size).

        Examples:
            >>> adapter = Houlsby_Adapter(input_size=768, bottleneck=32)
            >>> input_tensor = torch.randn(10, 768)  # Batch of 10 samples
            >>> output_tensor = adapter(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 768])
        """
        return self.houlsby_adapter(x)


if not is_s3prl_available:
    HoulsbyTransformerSentenceEncoderLayer = None
else:

    class HoulsbyTransformerSentenceEncoderLayer(TransformerSentenceEncoderLayer):
        """Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained

        models.
        """

        def __init__(
            self,
            bottleneck: int = 32,
            **kwargs,
        ) -> None:

            super(HoulsbyTransformerSentenceEncoderLayer, self).__init__(**kwargs)
            self.bottleneck = bottleneck
            self.adapter = Houlsby_Adapter(
                input_size=self.fc2.out_features,
                bottleneck=self.bottleneck,
            )

        def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
        ):
            """LayerNorm is applied either before or after the self-attention/ffn

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
