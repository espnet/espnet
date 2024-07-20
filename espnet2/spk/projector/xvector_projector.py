import torch

from espnet2.spk.projector.abs_projector import AbsProjector


class XvectorProjector(AbsProjector):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size

        self.fc1 = torch.nn.Linear(input_size, output_size)
        self.fc2 = torch.nn.Linear(output_size, output_size)
        self.act = torch.nn.ReLU()

    def output_size(self):
        return self._output_size

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
