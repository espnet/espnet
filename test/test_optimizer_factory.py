import chainer
import torch

from espnet.optimizer.factory import dynamic_import_optimizer


def test_pytorch_optimizer_factory():
    model = torch.nn.Linear(2, 1)
    opt_class = dynamic_import_optimizer("adam", "pytorch")
    optimizer = opt_class.build(model.parameters(), lr=0.9)
    for g in optimizer.param_groups:
        assert g["lr"] == 0.9

    opt_class = dynamic_import_optimizer("sgd", "pytorch")
    optimizer = opt_class.build(model.parameters(), lr=0.9)
    for g in optimizer.param_groups:
        assert g["lr"] == 0.9

    opt_class = dynamic_import_optimizer("adadelta", "pytorch")
    optimizer = opt_class.build(model.parameters(), rho=0.9)
    for g in optimizer.param_groups:
        assert g["rho"] == 0.9


def test_chainer_optimizer_factory():
    model = chainer.links.Linear(2, 1)
    opt_class = dynamic_import_optimizer("adam", "chainer")
    optimizer = opt_class.build(model, lr=0.9)
    assert optimizer.alpha == 0.9

    opt_class = dynamic_import_optimizer("sgd", "chainer")
    optimizer = opt_class.build(model, lr=0.9)
    assert optimizer.lr == 0.9

    opt_class = dynamic_import_optimizer("adadelta", "chainer")
    optimizer = opt_class.build(model, rho=0.9)
    assert optimizer.rho == 0.9
