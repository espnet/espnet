from collections import defaultdict

import chainer
import numpy

from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.tensorboard_logger import TensorboardLogger


class DummyWriter:
    def __init__(self):
        self.data = defaultdict(dict)

    def add_scalar(self, k, v, n):
        self.data[k][n] = v


def test_tensorboard_evaluator():
    model = chainer.links.Classifier(chainer.links.Linear(3, 2))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    data_size = 7
    xs = numpy.random.randn(data_size, 3).astype(numpy.float32)
    ys = (numpy.random.randn(data_size) > 1).astype(numpy.int32)
    data = chainer.datasets.TupleDataset(xs, ys)
    batch_size = 2
    epoch = 10
    trainer = chainer.training.Trainer(
        chainer.training.StandardUpdater(
            chainer.iterators.SerialIterator(data, batch_size), optimizer),
        (epoch, "epoch"))
    trainer.extend(BaseEvaluator(
        chainer.iterators.SerialIterator(data, batch_size, repeat=False),
        model))
    # test runnable without tensorboard logger
    trainer.run()

    # test runnable with tensorboard logger
    trainer = chainer.training.Trainer(
        chainer.training.StandardUpdater(
            chainer.iterators.SerialIterator(data, batch_size), optimizer),
        (epoch, "epoch"))
    trainer.extend(BaseEvaluator(
        chainer.iterators.SerialIterator(data, batch_size, repeat=False),
        model))
    writer = DummyWriter()
    log_interval = 3
    trainer.extend(TensorboardLogger(writer), trigger=(log_interval, "iteration"))
    trainer.run()

    assert TensorboardLogger.default_name in trainer._extensions
    # TODO(karita): what is the correct number of len(train_loss)?
    # train_loss = writer.data["main/loss"]
    # assert len(train_loss) == (epoch * (data_size // log_interval + 1))
    val_loss = writer.data["validation/main/loss"]
    assert len(val_loss) == epoch, val_loss
