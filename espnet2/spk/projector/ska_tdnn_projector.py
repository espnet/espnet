import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class SkaTdnnProjector(AbsProjector):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size)
        self.fc = torch.nn.Linear(input_size, output_size)
        self.bn2 = torch.nn.BatchNorm1d(output_size)

    def output_size(self):
        return self._output_size

    def forward(self, x):
        return self.bn2(self.fc(self.bn(x)))
