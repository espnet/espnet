
from chainer.training import StandardUpdater

from espnet.utils.training.tensorboard_logger import TensorboardLogger


class TensorboardStandardUpdater(StandardUpdater):
    def set_tb_trigger(self, trainer, tb_trigger):
        self.trainer = trainer
        self.tb_trigger = tb_trigger

    def update(self):
        TensorboardLogger.reset_trigger(self.trainer, self.tb_trigger)
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1
