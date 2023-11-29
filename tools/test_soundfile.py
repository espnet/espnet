#!/usr/bin/env python
import tempfile
from pathlib import Path

import numpy as np
import soundfile


def test(audio_format="flac", dtype=None):
    """
    This is a test program designed to avoid a bug in which
    an invalid write/read operation is performed
    on FLAC audio files using soundfile under certain conditions
    when using a specific version of libsndfile.
    """

    N = 20
    r = 16000
    if dtype == np.int16:
        ma = np.iinfo(np.int16).max
        mi = np.iinfo(np.int16).min
        a = ma * np.ones(N, dtype=np.int16)
    else:
        ma = np.iinfo(np.int16).max / abs(np.iinfo(np.int16).min)
        mi = -1
        a = ma * np.ones(N)
    mask = np.random.rand(N) > 0.5
    a[mask] = mi
    with tempfile.TemporaryDirectory() as dd:
        f = Path(dd) / f"a.{audio_format}"
        soundfile.write(f, a, r)
        a2, r = soundfile.read(f, dtype=dtype)
    np.testing.assert_equal(a, a2)


if __name__ == "__main__":
    print("Performing soundfile test...")
    try:
        for audio_format in ["wav", "flac"]:
            for i in range(5):
                test(audio_format=audio_format, dtype=np.int16)
            for i in range(5):
                test(audio_format=audio_format)
    except Exception:
        print(
            "Your libsndfile has a bug for the read/write operation "
            "on flac files. Please install new libsndfile"
        )
        raise
    print("The soundfile test succesfully finished!")
