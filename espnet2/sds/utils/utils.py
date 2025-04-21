from typing import Union

import numpy as np


def int2float(sound: Union[np.ndarray, list]) -> np.ndarray:
    """Converts an integer PCM audio signal to a floating-point representation.

    This function scales an integer PCM audio
    waveform (typically `int16`)
    to a `float32` format, normalizing the values
    to the range [-1.0, 1.0].

    Args:
        sound (Union[np.ndarray, list]):
            The input audio signal in integer format.
            Typically a NumPy array or a list of integers.

    Returns:
        np.ndarray:
            The audio signal converted to `float32` format and normalized.
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound
