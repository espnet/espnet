from abc import ABC, abstractmethod

import numpy as np


class AbsFrameVAD(ABC):
    """A streaming frame scorer: the speech probability of each frame.

    TurnDetector (espnet2.sds.vad.turn_detector) turns these scores into the
    user's turns, so a new VAD for the dialogue system only implements this
    class. Frame i starts i * frame_shift seconds into the stream, and its
    score is ready once window_s seconds of audio from its start are in.

    What an implementation holds must stay bounded however long the stream
    runs: it keeps no history of scores, and only the audio its next frames
    need.
    """

    @property
    @abstractmethod
    def frame_shift(self) -> float:
        """Seconds between frames."""
        raise NotImplementedError

    @property
    @abstractmethod
    def window_s(self) -> float:
        """Seconds of audio from a frame's start until its score is ready."""
        raise NotImplementedError

    @abstractmethod
    def accept(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Add audio; return the scores of the frames it completes.

        Args:
            audio (np.ndarray): The next audio, 1-D float32 in [-1, 1].
            sample_rate (int): Its sampling rate, the same until reset().

        Returns:
            np.ndarray: One speech probability in [0, 1] per new frame.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Forget all audio and start a new stream."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Prepare the scorer (e.g. run the model once), then reset it."""
        self.reset()
