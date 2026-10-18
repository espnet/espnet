import pytest
import torch

import espnet2.spk.loss.aamsoftmax as aam
import espnet2.spk.loss.aamsoftmax_subcenter_intertopk as sub
from espnet2.spk.loss.aamsoftmax import AAMSoftmax, sine_from_cosine
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)

NOUT = 8
NCLASSES = 4


def _aamsoftmax(easy_margin=False):
    return AAMSoftmax(nout=NOUT, nclasses=NCLASSES, easy_margin=easy_margin)


def _subcenter_intertopk():
    return ArcMarginProduct_intertopk_subcenter(
        nout=NOUT, nclasses=NCLASSES, K=2, mp=0.06, k_top=2
    )


LOSSES = [
    pytest.param(_aamsoftmax, id="aamsoftmax"),
    pytest.param(lambda: _aamsoftmax(easy_margin=True), id="aamsoftmax_em"),
    pytest.param(_subcenter_intertopk, id="subcenter_intertopk"),
]


def test_sine_from_cosine_matches_clamped_sqrt():
    torch.manual_seed(0)
    cosine = torch.cat(
        [
            torch.rand(1000) * 2 - 1,
            torch.tensor([-1.0, 0.0, 1.0, 1.0 + 1e-6, -1.0 - 1e-6]),
        ]
    )
    expected = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
    assert torch.equal(sine_from_cosine(cosine), expected)


@pytest.mark.parametrize("value", [1.0, -1.0, 1.0 + 1e-6])
def test_sine_from_cosine_finite_grad_at_boundary(value):
    cosine = torch.tensor([value], requires_grad=True)
    sine = sine_from_cosine(cosine)
    sine.sum().backward()
    assert torch.isfinite(sine).all()
    assert torch.isfinite(cosine.grad).all()


@pytest.mark.parametrize("loss_fn", LOSSES)
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16"])
def test_aligned_embedding_gives_finite_grads(loss_fn, autocast):
    torch.manual_seed(0)
    loss_module = loss_fn()
    # Each embedding equals the (first sub-center) weight of its own class,
    # so the target cosine is exactly 1.
    stride = loss_module.weight.size(0) // NCLASSES
    label = torch.arange(NCLASSES)
    emb = loss_module.weight.detach()[label * stride].clone().requires_grad_(True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, _, _ = loss_module(emb, label)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(emb.grad).all()
    assert torch.isfinite(loss_module.weight.grad).all()


@pytest.mark.parametrize("loss_fn", LOSSES)
def test_ordinary_inputs_unchanged(loss_fn):
    """The loss equals the one computed with the previous sqrt expression."""
    torch.manual_seed(0)
    loss_module = loss_fn()
    emb = torch.randn(6, NOUT)
    label = torch.randint(0, NCLASSES, (6,))

    loss, _, _ = loss_module(emb, label)

    def reference(cosine):
        return torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(aam, "sine_from_cosine", reference)
        mp.setattr(sub, "sine_from_cosine", reference)
        ref_loss, _, _ = loss_module(emb, label)

    assert torch.equal(loss, ref_loss)
