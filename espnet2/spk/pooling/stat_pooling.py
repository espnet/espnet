import torch

from espnet2.spk.pooling.abs_pooling import AbsPooling


class StatsPooling(AbsPooling):
    """Aggregates frame-level features to single utterance-level feature.

    Proposed in D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker
    recognition"

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
            For this pooling layer, the output dimensionality will be double of
            the input_size
    """

    def __init__(self, input_size: int = 1536):
        super().__init__()
        self._output_size = input_size * 2

    def output_size(self):
        return self._output_size

    def forward(self, x, task_tokens: torch.Tensor = None):
        if task_tokens is not None:
            raise ValueError("StatisticsPooling is not adequate for task_tokens")
        mu = torch.mean(x, dim=-1)
        st = torch.std(x, dim=-1)

        x = torch.cat((mu, st), dim=1)

        return x
