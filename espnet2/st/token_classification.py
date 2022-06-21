from typing import Optional
import logging
import torch
import copy
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
try:
    from transformers import AutoModel
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

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
        model_name_or_path: str = "na",
<<<<<<< Updated upstream
        use_bert: bool = False,
=======
>>>>>>> Stashed changes
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
        if model_name_or_path!="na":
            if not is_transformers_available:
                raise ImportError(
                    "`transformers` is not available. Please install it via `pip install"
                    " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                    " && ./installers/install_transformers.sh`."
                )

            model = AutoModel.from_pretrained(model_name_or_path)

            config = model.config
            self.config = config
            if hasattr(model, "encoder"):
                self.transformer = model.encoder
            else:
                self.transformer = model
            if hasattr(self.transformer, "embed_tokens"):
                del self.transformer.embed_tokens
            if hasattr(self.transformer, "wte"):
                del self.transformer.wte
            if hasattr(self.transformer, "word_embedding"):
                del self.transformer.word_embedding

            self.use_inputs_embeds = False
            self.extend_attention_mask = True
            self.use_bert = use_bert
            if not self.use_bert:
                self.linear_in = torch.nn.Linear(
                    encoder_output_size, config.hidden_size
                )

        self.model_name_or_path = model_name_or_path
        self.speech_attn=speech_attn
        logging.info(f"Speech Attn: {speech_attn}")

        self._num_labels = num_labels
        if model_name_or_path!="na":
            self.linear_decoder = torch.nn.Linear(config.hidden_size, num_labels)
        else:
            self.linear_decoder = torch.nn.Linear(encoder_output_size, num_labels)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor, speech: Optional[torch.Tensor] = None, speech_lens: Optional[torch.Tensor] = None):
        """Forward.

        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        if self.model_name_or_path!="na":
<<<<<<< Updated upstream
            if not self.use_bert:
                input = self.linear_in(input)
=======
            input = self.linear_in(input)
>>>>>>> Stashed changes
            args = {"return_dict": True}

            mask = (~make_pad_mask(ilens)).to(input.device).float()

            args["attention_mask"] = _extend_attention_mask(mask)
            args["hidden_states"] = input
            encoder_out = self.transformer(**args).last_hidden_state
        else:
            if self.speech_attn:
                encoder_out, encoder_out_lens, _ = super().forward(input, ilens, speech=speech, speech_lens=speech_lens)
            else:
                encoder_out, encoder_out_lens, _ = super().forward(input, ilens)
        output = self.linear_decoder(encoder_out)

        return output

    @property
    def num_labels(self):
        return self._num_labels

def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
