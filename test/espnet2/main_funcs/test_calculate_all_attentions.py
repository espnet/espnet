from collections import defaultdict

import numpy as np
import pytest
import torch

from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.main_funcs.calculate_all_attentions import \
    calculate_all_attentions
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.rnn.attentions import AttAdd
from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention


class Dummy(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.att1 = MultiHeadedAttention(2, 10, 0.0)
        self.att2 = AttAdd(10, 20, 15)
        self.desired = defaultdict(list)

    def forward(self, x, x_lengths, y, y_lengths):
        self.att1(y, x, x, None)
        _, a2 = self.att2(x, x_lengths, y, None)
        self.desired["att1"].append(self.att1.attn.squeeze(0))
        self.desired["att2"].append(a2)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


class Dummy2(AbsESPnetModel):
    def __init__(self, atype):
        super().__init__()
        self.decoder = RNNDecoder(50, 128, att_conf=dict(atype=atype))

    def forward(self, x, x_lengths, y, y_lengths):
        self.decoder(x, x_lengths, y, y_lengths)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


def test_calculate_all_attentions_MultiHeadedAttention():
    model = Dummy()
    bs = 2
    batch = {
        "x": torch.randn(bs, 3, 10),
        "x_lengths": torch.tensor([3, 2], dtype=torch.long),
        "y": torch.randn(bs, 2, 10),
        "y_lengths": torch.tensor([4, 4], dtype=torch.long),
    }
    t = calculate_all_attentions(model, batch)
    print(t)
    for k in model.desired:
        for i in range(bs):
            np.testing.assert_array_equal(t[k][i].numpy(), model.desired[k][i].numpy())


@pytest.mark.parametrize(
    "atype",
    [
        "noatt",
        "dot",
        "add",
        "location",
        "location2d",
        "location_recurrent",
        "coverage",
        "coverage_location",
        "multi_head_dot",
        "multi_head_add",
        "multi_head_loc",
        "multi_head_multi_res_loc",
    ],
)
def test_calculate_all_attentions(atype):
    model = Dummy2(atype)
    bs = 2
    batch = {
        "x": torch.randn(bs, 20, 128),
        "x_lengths": torch.tensor([20, 17], dtype=torch.long),
        "y": torch.randint(0, 50, [bs, 7]),
        "y_lengths": torch.tensor([7, 5], dtype=torch.long),
    }
    t = calculate_all_attentions(model, batch)
    for k, o in t.items():
        for i, att in enumerate(o):
            print(att.shape)
            if att.dim() == 2:
                att = att[None]
            for a in att:
                assert a.shape == (batch["y_lengths"][i], batch["x_lengths"][i])
