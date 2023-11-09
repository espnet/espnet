import torch
import torch.nn as nn

from espnet2.spk.pooling.abs_pooling import AbsPooling


class ChnAttnStatPooling(AbsPooling):
    """
    Aggregates frame-level features to single utterance-level feature.
    Proposed in B.Desplanques et al., "ECAPA-TDNN: Emphasized Channel
    Attention, Propagation and Aggregation in TDNN Based Speaker Verification"

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_size * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, input_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        if task_tokens is not None:
            raise ValueError(
                "ChannelAttentiveStatisticsPooling is not adequate for task_tokens"
            )
        t = x.size()[-1]
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)
                ).repeat(1, 1, t),
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
