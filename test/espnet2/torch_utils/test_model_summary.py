import torch

from espnet2.torch_utils.model_summary import model_summary


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1000, 1000)
        self.l2 = torch.nn.Linear(1000, 1000)
        self.l3 = torch.nn.Linear(1000, 1000)


def test_model_summary():
    print(model_summary(Model()))
