import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet.nets.pytorch_backend.transformer.embedding import (
    LearnableFourierPosEnc,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)


class ConditionalChnAttnStatPooling(AbsPooling):
    """
    Aggregates frame-level features to single utterance-level feature.
    Proposed in B.Desplanques et al., "ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification"

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
        vocab_size: vocabulary size (i.e., task number).
        pos_enc_class: type of positional encoding.
        input_layer: type of embedding the task token.
        token_dim: token embedding dimensionality.
        positional_dropout_rate: positional dropout ratio for 'linear' embedding
        dropout_rate: dropout ratio for 'linear' embedding layer
    """

    def __init__(
        self,
        input_size: int = 1536,
        vocab_size: int = 1,
        pos_enc_class: str = "rel_pos",
        input_layer: str = "embed",
        token_dim: int = 128,
        positional_dropout_rate: float = 0.1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3 + token_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self._output_size = input_size * 2

        # related to condition
        if pos_enc_class == "pos":
            pos_enc_layer = PositionalEncoding
        elif pos_enc_class == "rel_pos":
            pos_enc_layer = RelPositionalEncoding
        elif pos_enc_class == "learnable_fourier_pos":
            pos_enc_layer = LearnableFourierPosEnc
        elif pose_enc_class == "scale_pos":
            pos_enc_layer = ScaledPositionalEncoding

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, token_dim),
                pos_enc_layer(token_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, token_dim),
                torch.nn.LayerNorm(token_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_layer(token_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        if task_tokens is None:
            raise ValueError(
                "ConcitionalChannelAttentiveStatisticsPooling requires"
                "task_tokens"
            )

        # manipulate tokens to (bs, dim, seq)
        ttokens = self.embed(task_tokens).transpose(1, 2)
        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4),
                ).repeat(1, 1, t),
                ttokens.repeat(1, 1, t),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4)
        )

        x = torch.cat((mu, sg), dim=1)

        return x
