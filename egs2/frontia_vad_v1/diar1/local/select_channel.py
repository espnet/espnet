#!/usr/bin/env python3
"""Write one channel of a sound file to stdout as 16-bit WAV.

    python3 local/select_channel.py AUDIO CHANNEL

Used in wav.scp pipes ("python3 local/select_channel.py x.wav 0 |") so that a
multi-channel recording is dumped as the single channel that is evaluated.
"""

import io
import sys

import soundfile as sf


def main():
    path, channel = sys.argv[1], int(sys.argv[2])
    wav, fs = sf.read(path, dtype="int16", always_2d=True)
    # written to memory first: the WAV header is completed with a seek, which
    # a pipe does not allow
    buf = io.BytesIO()
    sf.write(buf, wav[:, channel], fs, format="WAV", subtype="PCM_16")
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
