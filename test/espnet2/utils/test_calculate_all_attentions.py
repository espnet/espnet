from collections import defaultdict

import numpy as np
import torch

from espnet.nets.pytorch_backend.rnn.attentions import AttAdd
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.train.abs_e2e import AbsE2E
from espnet2.utils.calculate_all_attentions import calculate_all_attentions


class Dummy(AbsE2E):
    def __init__(self):
        super().__init__()
        self.att1 = MultiHeadedAttention(2, 10, 0.0)
        self.att2 = AttAdd(10, 20, 15)
        self.desired = defaultdict(list)

    def forward(self, x, x_lengths, y, y_lengths):
        a1 = self.att1(y, x, x, None)
        _, a2 = self.att2(x, x_lengths, y, None)
        self.desired["att1"].append(a1)
        self.desired["att2"].append(a2)


def test_calculate_all_attentions():
    model = Dummy()
    bs = 2
    batch = {
        "x": torch.randn(bs, 3, 10),
        "x_lengths": torch.tensor([3, 2], dtype=torch.long),
        "y": torch.randn(bs, 2, 10),
        "y_lengths": torch.tensor([4, 4], dtype=torch.long),
    }
    t = calculate_all_attentions(model, batch)
    for k in model.desired:
        for i in range(bs):
            np.testing.assert_array_equal(t[k][i].numpy(), model.desired[k][i].numpy())
