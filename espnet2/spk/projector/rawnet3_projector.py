import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class RawNet3Projector(AbsProjector):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size

        self.bn = torch.nn.BatchNorm1d(input_size)
        self.fc = torch.nn.Linear(input_size, output_size)

    def output_size(self):
        return self._output_size

    def forward(self, x):
        return self.fc(self.bn(x))
