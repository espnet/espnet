import pytest
import chainer
import numpy
try:
    import torch
except ImportError:
    pytest.skip("torch not installed")


class ChModel(chainer.Chain):
    def __init__(self):
        super(ChModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(3, 1)

    def __call__(self, x):
        return chainer.functions.sum(self.a(x))


class ThModel(torch.nn.Module):
    def __init__(self):
        super(ThModel, self).__init__()
        self.a = torch.nn.Linear(3, 1)

    def forward(self, x):
        return self.a(x).sum()


@pytest.mark.parametrize("ch_opt_t,th_opt_t", [
    (chainer.optimizers.SGD, lambda ps: torch.optim.SGD(ps, lr=0.01)),
    (chainer.optimizers.Adam, torch.optim.Adam),
    (chainer.optimizers.AdaDelta, lambda ps: torch.optim.Adadelta(ps, rho=0.95))
])
def test_optimizer(ch_opt_t, th_opt_t):
    # model construction
    ch_model = ChModel()
    th_model = ThModel()

    # copy params
    th_model.a.weight.data = torch.from_numpy(numpy.copy(ch_model.a.W.data))
    th_model.a.bias.data = torch.from_numpy(numpy.copy(ch_model.a.b.data))

    # optimizer setup
    ch_opt = ch_opt_t()
    ch_opt.setup(ch_model)
    th_opt = th_opt_t(th_model.parameters())

    # forward
    ch_model.cleargrads()
    data = numpy.random.randn(2, 3).astype(numpy.float32)
    v = chainer.Variable(data)
    ch_loss = ch_model(data)
    th_loss = th_model(torch.autograd.Variable(torch.from_numpy(data)))
    numpy.testing.assert_allclose(ch_loss.data, th_loss.data.numpy())

    chainer.functions.sum(ch_loss).backward()
    th_loss.backward()
    ch_opt.update()
    th_opt.step()
    numpy.testing.assert_allclose(ch_model.a.W.data, th_model.a.weight.data.numpy())
    numpy.testing.assert_allclose(ch_model.a.b.data, th_model.a.bias.data.numpy())
    
