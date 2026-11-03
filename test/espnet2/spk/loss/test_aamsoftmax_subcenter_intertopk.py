import pytest
import torch
import torch.nn.functional as F

import espnet2.spk.loss.aamsoftmax_subcenter_intertopk as sub
from espnet2.spk.loss.aamsoftmax_subcenter_intertopk import (
    ArcMarginProduct_intertopk_subcenter,
    ArcMarginProduct_intertopk_subcenter_multilabel,
    ArcMarginProduct_intertopk_subcenter_softtarget,
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


def _arcface_logits(module, inputs, target):
    """Return scale * cos(theta + m) for the positive classes, scale * cos else."""
    cosine = F.linear(F.normalize(inputs), F.normalize(module.weight))
    cosine = cosine.view(len(inputs), -1, module.K).amax(dim=2)
    margin = torch.cos(torch.acos(cosine) + module.margin)
    return module.scale * torch.where(target > 0, margin, cosine)


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


def test_softtarget_hard_labels_match_original_loss_and_gradients():
    """Hard labels give the same loss and gradients as the original loss."""
    torch.manual_seed(7)
    hard = ArcMarginProduct_intertopk_subcenter(nout=4, nclasses=8)
    soft = ArcMarginProduct_intertopk_subcenter_softtarget(nout=4, nclasses=8)
    soft.load_state_dict(hard.state_dict(), strict=True)
    hard.update(0.3)
    soft.update(0.3)
    inputs = torch.randn(3, 4, requires_grad=True)
    soft_inputs = inputs.detach().clone().requires_grad_()
    labels = torch.tensor([[1], [4], [6]])
    loss, acc, pred = hard(inputs, labels)
    soft_loss, soft_acc, soft_pred = soft(soft_inputs, labels)
    loss.backward()
    soft_loss.backward()
    torch.testing.assert_close(soft_loss, loss)
    torch.testing.assert_close(soft_acc, acc)
    torch.testing.assert_close(soft_pred, pred)
    torch.testing.assert_close(soft_inputs.grad, inputs.grad)
    torch.testing.assert_close(soft.weight.grad, hard.weight.grad)


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
