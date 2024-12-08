import numpy as np
def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound