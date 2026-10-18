import pytest
import torch

import espnet2.spk.loss.aamsoftmax as aam
from espnet2.spk.loss.aamsoftmax import AAMSoftmax, sine_from_cosine

NOUT = 8
NCLASSES = 4


def _reference_sine(cosine):
    """Return the sine as computed before sine_from_cosine was introduced."""
    return torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))


def test_sine_from_cosine_matches_clamped_sqrt():
    """sine_from_cosine is bit-identical to the clamped sqrt, incl. |cos| >= 1."""
    torch.manual_seed(0)
    cosine = torch.cat(
        [
            torch.rand(1000) * 2 - 1,
            torch.tensor([-1.0, 0.0, 1.0, 1.0 + 1e-6, -1.0 - 1e-6]),
        ]
    )
    assert torch.equal(sine_from_cosine(cosine), _reference_sine(cosine))


@pytest.mark.parametrize("value", [1.0, -1.0, 1.0 + 1e-6])
def test_sine_from_cosine_finite_grad_at_boundary(value):
    """The gradient of sine_from_cosine is finite at and beyond |cos| == 1."""
    cosine = torch.tensor([value], requires_grad=True)
    sine = sine_from_cosine(cosine)
    sine.sum().backward()
    assert torch.isfinite(sine).all()
    assert torch.isfinite(cosine.grad).all()


@pytest.mark.parametrize("easy_margin", [False, True])
@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16"])
def test_aligned_embedding_gives_finite_grads(easy_margin, autocast):
    """An embedding equal to its class weight (cos == 1) gives finite grads."""
    torch.manual_seed(0)
    loss_module = AAMSoftmax(nout=NOUT, nclasses=NCLASSES, easy_margin=easy_margin)
    label = torch.arange(NCLASSES)
    emb = loss_module.weight.detach()[label].clone().requires_grad_(True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, _, _ = loss_module(emb, label)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(emb.grad).all()
    assert torch.isfinite(loss_module.weight.grad).all()


@pytest.mark.parametrize("easy_margin", [False, True])
def test_ordinary_inputs_unchanged(easy_margin):
    """The loss equals the one computed with the previous sqrt expression."""
    torch.manual_seed(0)
    loss_module = AAMSoftmax(nout=NOUT, nclasses=NCLASSES, easy_margin=easy_margin)
    emb = torch.randn(6, NOUT)
    label = torch.randint(0, NCLASSES, (6,))

    loss, _, _ = loss_module(emb, label)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(aam, "sine_from_cosine", _reference_sine)
        ref_loss, _, _ = loss_module(emb, label)

    assert torch.equal(loss, ref_loss)
