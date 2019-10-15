"""Language model interface."""

from chainer import link

from espnet.nets.lm_interface import LMInterface


class ChainerLMInterface(LMInterface, link.Chain):
    """LM Interface for Chainer model implementation."""

    @staticmethod
    def custom_updater(*args, **kw):
        from espnet.nets.chainer_backend.lm.default import BPTTUpdater
        train_iter, optimizer, gpuid = args
        return BPTTUpdater(train_iter, optimizer, gpuid)
