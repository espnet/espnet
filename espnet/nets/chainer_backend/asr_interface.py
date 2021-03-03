"""ASR Interface module."""
import chainer

from espnet.nets.asr_interface import ASRInterface


class ChainerASRInterface(ASRInterface, chainer.Chain):
    """ASR Interface for ESPnet model implementation."""

    @staticmethod
    def custom_converter(*args, **kw):
        """Get customconverter of the model (Chainer only)."""
        raise NotImplementedError("custom converter method is not implemented")

    @staticmethod
    def custom_updater(*args, **kw):
        """Get custom_updater of the model (Chainer only)."""
        raise NotImplementedError("custom updater method is not implemented")

    @staticmethod
    def custom_parallel_updater(*args, **kw):
        """Get custom_parallel_updater of the model (Chainer only)."""
        raise NotImplementedError("custom parallel updater method is not implemented")

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        raise NotImplementedError(
            "get_total_subsampling_factor method is not implemented"
        )
