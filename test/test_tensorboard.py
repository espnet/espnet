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
    # setup model
    model = chainer.links.Classifier(chainer.links.Linear(3, 2))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # setup data
    data_size = 6
    xs = numpy.random.randn(data_size, 3).astype(numpy.float32)
    ys = (numpy.random.randn(data_size) > 1).astype(numpy.int32)
    data = chainer.datasets.TupleDataset(xs, ys)
    batch_size = 2
    epoch = 10

    # test runnable without tensorboard logger
    trainer = chainer.training.Trainer(
        chainer.training.StandardUpdater(
            chainer.iterators.SerialIterator(data, batch_size), optimizer
        ),
        (epoch, "epoch"),
    )
    trainer.extend(
        BaseEvaluator(
            chainer.iterators.SerialIterator(data, batch_size, repeat=False), model
        )
    )
    trainer.run()

    # test runnable with tensorboard logger
    for log_interval in [1, 3]:
        trainer = chainer.training.Trainer(
            chainer.training.StandardUpdater(
                chainer.iterators.SerialIterator(data, batch_size), optimizer
            ),
            (epoch, "epoch"),
        )
        trainer.extend(
            BaseEvaluator(
                chainer.iterators.SerialIterator(data, batch_size, repeat=False), model
            )
        )
        writer = DummyWriter()
        trainer.extend(TensorboardLogger(writer), trigger=(log_interval, "iteration"))
        trainer.run()

        # test the number of log entries
        assert TensorboardLogger.default_name in trainer._extensions
        assert (
            len(writer.data["main/loss"]) == trainer.updater.iteration // log_interval
        )
        assert len(writer.data["validation/main/loss"]) == epoch
