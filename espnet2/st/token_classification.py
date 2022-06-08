from typing import Optional
import logging
import torch
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

class LinearDecoder(TransformerEncoder):
    """Linear decoder for token classification"""

    def __init__(
        self,
        encoder_output_size: int,
        num_labels: int = 2,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        speech_attn: bool = False,
    ):
        super().__init__(
            input_size = encoder_output_size,
            output_size = encoder_output_size,
            attention_heads = attention_heads,
            linear_units = linear_units,
            num_blocks = num_blocks,
            dropout_rate = dropout_rate,
            positional_dropout_rate = positional_dropout_rate,
            attention_dropout_rate = attention_dropout_rate,
            input_layer = input_layer,
            pos_enc_class=pos_enc_class,
            positionwise_layer_type = positionwise_layer_type,
            positionwise_conv_kernel_size = positionwise_conv_kernel_size,
            padding_idx = padding_idx,
            speech_attn=speech_attn,
        )

        self.speech_attn=speech_attn
        logging.info(f"Speech Attn: {speech_attn}")

        self._num_labels = num_labels
        self.linear_decoder = torch.nn.Linear(encoder_output_size, num_labels)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, speech: Optional[torch.Tensor] = None, speech_lens: Optional[torch.Tensor] = None):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        if self.speech_attn:
            encoder_out, encoder_out_lens, _ = super().forward(input, ilens, speech=speech, speech_lens=speech_lens)
        else:
            encoder_out, encoder_out_lens, _ = super().forward(input, ilens)
        output = self.linear_decoder(encoder_out)

        return output

    @property
    def num_labels(self):
        return self._num_labels
