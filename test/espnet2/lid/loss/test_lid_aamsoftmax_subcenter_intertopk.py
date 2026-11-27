"""Tests for the LID soft-target and multi-label sub-center ArcFace losses."""

import pytest
import torch
import torch.nn.functional as F

from espnet2.lid.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter_multilabel,
    ArcMarginProduct_intertopk_subcenter_softtarget,
    _margin_logits,
    _subcenter_cosine,
)
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
)

NOUT = 8
NCLASSES = 4
K = 2


def _arcface_logits(module, inputs, target):
    """Return scale * cos(theta + m) for the positive classes, scale * cos else."""
    cosine = F.linear(F.normalize(inputs), F.normalize(module.weight))
    cosine = cosine.view(len(inputs), -1, module.K).amax(dim=2)
    margin = torch.cos(torch.acos(cosine) + module.margin)
    return module.scale * torch.where(target > 0, margin, cosine)


class _CaptureCE(torch.nn.CrossEntropyLoss):
    """Cross-entropy that records the logits it is given."""

    def forward(self, logits, label):
        self.logits = logits
        return super().forward(logits, label)


@pytest.mark.parametrize("easy_margin", [False, True])
def test_hard_labels_match_upstream(easy_margin):
    """Integer labels reproduce the upstream margin logits bit for bit."""
    torch.manual_seed(0)
    conf = dict(nout=NOUT, nclasses=8, K=K, k_top=5, easy_margin=easy_margin)
    hard = ArcMarginProduct_intertopk_subcenter(**conf)
    hard.ce = _CaptureCE()
    hard.update(0.3)
    inputs = torch.randn(6, NOUT)
    labels = torch.randint(0, 8, (6,))
    one_hot = F.one_hot(labels, 8).float()
    hard_in = inputs.clone().requires_grad_()
    hard_loss, hard_acc, hard_pred = hard(hard_in, labels)
    hard_loss.backward()
    ref_logits = hard.ce.logits.detach()

    for cls in (
        ArcMarginProduct_intertopk_subcenter_softtarget,
        ArcMarginProduct_intertopk_subcenter_multilabel,
    ):
        lid = cls(**conf)
        lid.load_state_dict(hard.state_dict(), strict=True)
        lid.update(0.3)
        with torch.no_grad():
            cosine = _subcenter_cosine(lid, inputs)
            assert torch.equal(_margin_logits(lid, inputs, cosine, one_hot), ref_logits)

        lid_in = inputs.clone().requires_grad_()
        loss, acc, pred = lid(lid_in, labels)
        assert torch.equal(pred, hard_pred)
        if cls is ArcMarginProduct_intertopk_subcenter_softtarget:
            expected = F.kl_div(
                F.log_softmax(ref_logits, 1), one_hot, reduction="batchmean"
            )
            assert torch.equal(acc, hard_acc)
            # KL and CE reduce in a different order, so agree to rounding only
            torch.testing.assert_close(loss, hard_loss)
            loss.backward()
            torch.testing.assert_close(lid_in.grad, hard_in.grad)
            torch.testing.assert_close(lid.weight.grad, hard.weight.grad)
        else:
            expected = F.binary_cross_entropy_with_logits(ref_logits, one_hot)
        assert torch.equal(loss.detach(), expected)


def test_multilabel_loss_is_bce_and_predictions_ignore_labels():
    """The multi-label loss is BCE with a margin on every positive class."""
    torch.manual_seed(0)
    loss_module = ArcMarginProduct_intertopk_subcenter_multilabel(
        nout=NOUT, nclasses=NCLASSES, K=K, k_top=0, pos_weight=5.0
    )
    inputs = torch.randn(2, NOUT)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    loss, _, preds = loss_module(inputs, targets)
    expected = F.binary_cross_entropy_with_logits(
        _arcface_logits(loss_module, inputs, targets),
        targets,
        pos_weight=torch.full((NCLASSES,), 5.0),
    )
    torch.testing.assert_close(loss, expected)

    no_label_loss, no_label_accuracy, no_label_preds = loss_module(inputs)
    assert no_label_loss is None and no_label_accuracy is None
    assert preds.shape == (2,)
    assert torch.equal(preds, no_label_preds)
    zero_target = torch.zeros_like(targets)
    assert torch.equal(
        preds, _arcface_logits(loss_module, inputs, zero_target).argmax(1)
    )


def test_multilabel_accuracy_thresholds_scores_without_margin():
    """Subset accuracy thresholds sigmoid(scale * cos), not the margin logits."""
    loss_module = ArcMarginProduct_intertopk_subcenter_multilabel(
        nout=3, nclasses=3, K=1, k_top=0, scale=10.0, margin=1.0, threshold=0.5
    )
    with torch.no_grad():
        loss_module.weight.copy_(torch.eye(3))
    inputs = torch.tensor([[1.0, 1.0, -1.0], [1.0, -1.0, -1.0]])
    targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    _, accuracy, _ = loss_module(inputs, targets)

    # Only the first row is correct; with the margin no row would be
    assert accuracy.item() == 0.5


def test_softtarget_loss_is_batchmean_kl():
    """The soft-target loss is KL(target || softmax) with a margin on positives."""
    torch.manual_seed(0)
    loss_module = ArcMarginProduct_intertopk_subcenter_softtarget(
        nout=NOUT, nclasses=NCLASSES, K=K, k_top=0
    )
    inputs = torch.randn(2, NOUT)
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    loss, accuracy, preds = loss_module(inputs, targets)
    normalized = targets / targets.sum(1, keepdim=True)
    logits = _arcface_logits(loss_module, inputs, targets)
    expected = F.kl_div(F.log_softmax(logits, 1), normalized, reduction="batchmean")

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        accuracy, (targets.gather(1, preds[:, None]) > 0).float().mean()
    )


def test_multilabel_pos_weight_round_trips_through_state_dict():
    """pos_weight is saved as bce.pos_weight and strict-loads into a new module."""
    loss_module = ArcMarginProduct_intertopk_subcenter_multilabel(
        nout=4, nclasses=3, K=1, pos_weight=50.0
    )
    state_dict = loss_module.state_dict()
    assert sorted(state_dict) == ["bce.pos_weight", "weight"]

    reloaded = ArcMarginProduct_intertopk_subcenter_multilabel(
        nout=4, nclasses=3, K=1, pos_weight=1.0
    )
    reloaded.load_state_dict(state_dict, strict=True)
    torch.testing.assert_close(reloaded.bce.pos_weight, torch.full((3,), 50.0))


@pytest.mark.parametrize(
    "loss_class",
    [
        ArcMarginProduct_intertopk_subcenter_softtarget,
        ArcMarginProduct_intertopk_subcenter_multilabel,
    ],
)
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16])
def test_dense_targets_have_finite_grads_at_cosine_boundaries(loss_class, dtype):
    """Gradients stay finite at cosine +/-1, including under AMP dtypes."""
    loss_module = loss_class(nout=4, nclasses=3, K=1, k_top=0)
    with torch.no_grad():
        loss_module.weight.copy_(
            torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        )
    inputs = torch.tensor([[1.0, 0, 0, 0]], requires_grad=True)
    with torch.autocast("cpu", enabled=dtype is not None, dtype=dtype):
        loss, _, _ = loss_module(inputs, torch.tensor([[1.0, 0, 0]]))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(loss_module.weight.grad).all()
