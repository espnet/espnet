import pytest
import torch

import espnet2.spk.loss.aamsoftmax_subcenter_intertopk as sub
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)

NOUT = 8
NCLASSES = 4
K = 2


def _make_loss():
    """Build a small sub-center inter-topk loss for testing."""
    return ArcMarginProduct_intertopk_subcenter(
        nout=NOUT, nclasses=NCLASSES, K=K, mp=0.06, k_top=2
    )


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16"])
def test_aligned_embedding_gives_finite_grads(autocast):
    """An embedding equal to a class sub-center (cos == 1) gives finite grads."""
    torch.manual_seed(0)
    loss_module = _make_loss()
    label = torch.arange(NCLASSES)
    # Weight rows are laid out as (class, sub-center); take sub-center 0.
    emb = loss_module.weight.detach()[label * K].clone().requires_grad_(True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        loss, _, _ = loss_module(emb, label)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(emb.grad).all()
    assert torch.isfinite(loss_module.weight.grad).all()


def test_ordinary_inputs_unchanged():
    """The loss equals the one computed with the previous sqrt expression."""
    torch.manual_seed(0)
    loss_module = _make_loss()
    emb = torch.randn(6, NOUT)
    label = torch.randint(0, NCLASSES, (6,))

    def reference(cosine):
        """Return the sine as computed before sine_from_cosine was introduced."""
        return torch.sqrt(1.0 - torch.pow(cosine, 2))

    loss, _, _ = loss_module(emb, label)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sub, "sine_from_cosine", reference)
        ref_loss, _, _ = loss_module(emb, label)

    assert torch.equal(loss, ref_loss)
