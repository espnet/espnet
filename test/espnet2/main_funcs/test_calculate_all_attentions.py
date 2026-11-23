from collections import defaultdict

import numpy as np
import pytest
import torch

from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.legacy.nets.pytorch_backend.rnn.attentions import AttAdd
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,
)
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.train.abs_espnet_model import AbsESPnetModel


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


class DummySDPA(AbsESPnetModel):
    """Self-attention with SDPA, plain and relative-position, and its explicit copy."""

    def __init__(self):
        super().__init__()
        self.att = MultiHeadedAttention(2, 10, 0.0, use_sdpa=True)
        self.rel = RelPositionMultiHeadedAttention(2, 10, 0.0, use_sdpa=True)
        self.pos = RelPositionalEncoding(10, 0.0)

    def forward(self, x, x_lengths):
        self.att(x, x, x, None)
        x, pos = self.pos(x)
        self.rel(x, x, x, pos, None)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


class NoWeights(MultiHeadedAttention):
    """An attention layer whose forward never sets the weights."""

    def forward(self, query, key, value, mask):
        return query


class DummyNoWeights(AbsESPnetModel):
    def __init__(self):
        super().__init__()
        self.att = NoWeights(2, 10, 0.0)

    def forward(self, x, x_lengths):
        self.att(x, x, x, None)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


def test_calculate_all_attentions_sdpa_plots_the_explicit_weights():
    torch.manual_seed(0)
    model = DummySDPA().eval()
    batch = {"x": torch.randn(2, 5, 10), "x_lengths": torch.tensor([5, 3])}
    t = calculate_all_attentions(model, batch)
    # the same layers on the explicit path give the plotted weights
    for name in ("att", "rel"):
        layer = getattr(model, name)
        assert layer.use_sdpa and layer.attn is None  # restored
        layer.use_sdpa = False
        for i, n in enumerate(batch["x_lengths"]):
            x = batch["x"][i : i + 1, :n]
            if name == "att":
                layer(x, x, x, None)
            else:
                xp, pos = model.pos(x)
                layer(xp, xp, xp, pos, None)
            assert torch.allclose(t[name][i], layer.attn[0])
        layer.use_sdpa = True


def test_calculate_all_attentions_skips_layers_without_weights():
    model = DummyNoWeights()
    batch = {"x": torch.randn(2, 5, 10), "x_lengths": torch.tensor([5, 3])}
    assert calculate_all_attentions(model, batch) == {}


class DummyCausalDecoder(AbsESPnetModel):
    """A transformer decoder: its self-attention is built with causal=True."""

    def __init__(self):
        super().__init__()
        self.decoder = TransformerDecoder(
            7, 10, attention_heads=2, linear_units=16, num_blocks=1, use_flash_attn=True
        )
        # the decoder keeps use_flash_attn only where flash_attn is installed; set
        # it as on such a machine (the plotting turns it off, so flash never runs)
        self.decoder.decoders[0].self_attn.use_flash_attn = True

    def forward(self, x, x_lengths, y, y_lengths):
        self.decoder(x, x_lengths, y, y_lengths)

    def collect_feats(self, **batch: torch.Tensor):
        return {}


def test_calculate_all_attentions_causal_flash_decoder_plots_causal_weights():
    torch.manual_seed(0)
    model = DummyCausalDecoder().eval()
    layer = model.decoder.decoders[0].self_attn
    assert layer.use_flash_attn and layer.causal
    batch = {
        "x": torch.randn(2, 5, 10),
        "x_lengths": torch.tensor([5, 3]),
        "y": torch.randint(0, 7, (2, 4)),
        "y_lengths": torch.tensor([4, 2]),
    }
    t = calculate_all_attentions(model, batch)
    assert layer.use_flash_attn and layer.attn is None  # restored
    # the explicit path gets the causal constraint from the decoder's mask
    # (padding & subsequent_mask), as flash attention gets it from causal=True
    for i, n in enumerate(batch["y_lengths"]):
        for a in t["decoder.decoders.0.self_attn"][i]:
            assert a.shape == (n, n)
            assert torch.all(torch.triu(a, diagonal=1) == 0)
            assert torch.allclose(a.sum(-1), torch.ones(n))
