from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class AbsVAD(torch.nn.Module, ABC):
    """A streaming VAD that tells a dialogue system when the user's turn ends.

    The dialogue system feeds the user's audio chunk by chunk to forward(). The
    VAD keeps state across calls and returns the audio of the user's turn once
    the turn has ended, and None before that.

    Every VAD follows this contract:

    - forward(speech, sample_rate, binary=False) takes the next chunk, 1-D
      int16 PCM (an implementation may accept more, e.g. float in [-1, 1]).
      The sample rate stays the same until reset().
    - It returns at most one turn per call. When several turns end within one
      chunk, the dialogue system can answer none of them before the call
      returns, so the VAD returns them as one turn.
    - The returned turn is float32 at the VAD's target_sr, or with
      binary=True the samples as they were given, at sample_rate.
    - What the VAD holds stays bounded however long the stream runs: between
      turns by a constant, within a turn by the VAD's limit on a turn's
      length.
    - reset() forgets the stream; warmup() prepares the VAD and leaves it
      as reset() does.

    How a turn starts and ends, and what happens to a turn that is too short
    or too long, is each VAD's own rule; see its class.
    """

    @abstractmethod
    def warmup(self) -> None:
        """Prepare the VAD (e.g. run the model once), then reset it."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget all audio and state, and start a new stream.

        Not abstract, so that a subclass written before reset() was added
        still instantiates; such a subclass raises here until it implements
        it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement reset()")

    @abstractmethod
    def forward(
        self,
        speech: np.ndarray,
        sample_rate: int,
        binary: bool = False,
    ) -> Optional[np.ndarray]:
        """Add the next chunk of audio; return the user's turn once it has ended.

        Args:
            speech (np.ndarray):
                The next chunk, 1-D int16 PCM.
            sample_rate (int):
                The sampling rate of the chunk.
            binary (bool, optional):
                If True, return the turn as the chunks were given, at
                sample_rate. Otherwise return float32 audio at the VAD's
                target_sr. Defaults to False.

        Returns:
            Optional[np.ndarray]:
                The turn, or None while no turn has ended.
        """
        raise NotImplementedError
