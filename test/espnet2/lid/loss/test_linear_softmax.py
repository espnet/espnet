import torch

from espnet2.lid.loss.linear_softmax import LinearSoftmax
from espnet2.tasks.lid import LIDTask


def test_loss_accuracy_and_inference():
    torch.manual_seed(0)
    loss_fn = LinearSoftmax(nout=8, nclasses=5)
    x = torch.randn(4, 8)
    label = torch.tensor([0, 1, 2, 3])
    loss, acc, preds = loss_fn(x, label)
    assert loss.ndim == 0 and torch.isfinite(loss) and 0 <= acc <= 1
    assert torch.equal(preds, loss_fn.linear(x).argmax(1))
    assert loss_fn(x, label[:, None])[0] == loss
    assert loss_fn(x)[:2] == (None, None)


def test_registered_in_lid_task():
    (losses,) = [c for c in LIDTask.class_choices_list if c.name == "loss"]
    assert losses.get_class("linear_softmax") is LinearSoftmax
