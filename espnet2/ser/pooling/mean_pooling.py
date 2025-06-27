import torch

from espnet2.ser.pooling.abs_pooling import AbsPooling


class MeanPooling(AbsPooling):
    """Average frame-level features to a single utterance-level feature.

    args:
        input_size: dimensionality of the input frame-level embeddings.
            Determined by encoder hyperparameter.
    """

    def __init__(self, input_size: int = 1024):
        super().__init__()
        self._output_size = input_size

    def output_size(self):
        return self._output_size

    def forward(self, x):
        x = torch.mean(x, dim=1)

        return x
