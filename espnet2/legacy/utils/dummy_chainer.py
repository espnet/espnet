"""Dummy classes of chainer required for CI test."""


class StandardUpdater:
    """A dummy StandardUpdater wrapper."""

    def __init__(self, *args, **kwargs):
        """Initliaze Dummy StandardUpdater."""
        pass

    def update_core(self, *args, **kwargs):
        """Update at every step."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class Extension:
    """A dummy Extension wrapper."""

    def __init__(self, *args, **kwargs):
        """Initliaze Dummy Extension."""
        self.log_attentions = self.__call__
        self.get_attention_weights = self.__call__
        self.get_attention_weight = self.__call__
        self.draw_attention_plot = self.__call__
        self._plot_and_save_attention = self.__call__
        pass

    def __call__(self, *args, **kwargs):
        """Plot and save imaged matrix of att_ws."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class Iterator:
    """A dummy Iterator wrapper."""

    def __init__(self, *args, **kwargs):
        """Initliaze Dummy Iterator."""
        self.__next__ = self.serialize
        self.start_shuffle = self.serialize
        pass

    def serialize(self, *args, **kwargs):
        """Append values to serializer."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class Reporter:
    """A dummy chainer reporter wrapper."""

    def report(self, *args, **kwargs):
        """Report at every step."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class Evaluator:
    """A dummy Evaluator wrapper."""

    def __call__(self, *args, **kwargs):
        """Report at every step."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class SerialIterator:
    """A dummy SerialIterator wrapper."""

    def start_shuffle(self, *args, **kwargs):
        """Report at every step."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )


class MultiprocessIterator:
    """A dummy MultiprocessIterator wrapper."""

    def start_shuffle(self, *args, **kwargs):
        """Report at every step."""
        raise NotImplementedError(
            "This is a dummy object to solve version compatibility issues.\n"
            "You need to install `chainer` if you want work with this class."
        )
