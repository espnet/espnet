"""Interface of encoders whose look-ahead can be changed at inference."""

from abc import abstractmethod

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class AbsLookAheadEncoder(AbsEncoder):
    """An encoder whose output may read a chosen number of future frames.

    One set of weights serves every look-ahead from 0 (online: an output frame
    depends on the current and past input frames only) to ``max_look_ahead``
    (offline). ``set_look_ahead`` chooses it, for training as for inference.
    A convolutional encoder implements it by switching off future kernel taps;
    a self-attention encoder can implement it by masking the attention to keys
    after t + look_ahead.
    """

    @property
    @abstractmethod
    def max_look_ahead(self) -> int:
        """Largest look-ahead, in input frames."""
        raise NotImplementedError

    @property
    @abstractmethod
    def look_ahead(self) -> int:
        """Current look-ahead, in input frames."""
        raise NotImplementedError

    @abstractmethod
    def set_look_ahead(self, frames: int):
        """Make each output frame t read input frames up to t + frames.

        Raises:
            ValueError: frames is negative, larger than max_look_ahead, or not
                reachable by this encoder.
        """
        raise NotImplementedError
