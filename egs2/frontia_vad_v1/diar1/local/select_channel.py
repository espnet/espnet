#!/usr/bin/env python3
"""Write one channel of a sound file to stdout as 16-bit WAV.

    python3 local/select_channel.py AUDIO CHANNEL

Used in wav.scp pipes ("python3 local/select_channel.py x.wav 0 |") so that a
multi-channel recording is dumped as the single channel that is evaluated.
The file is read in blocks of 10 s, so memory holds the selected channel and
one block of all channels, not the whole multi-channel recording.
"""

import io
import sys

import numpy as np
import soundfile as sf

BLOCK_SECONDS = 10


def main():
    path, channel = sys.argv[1], int(sys.argv[2])
    with sf.SoundFile(path) as f:
        if not 0 <= channel < f.channels:
            raise SystemExit(f"{path} has {f.channels} channels, not channel {channel}")
        fs = f.samplerate
        parts = [
            block[:, channel].copy()
            for block in f.blocks(
                blocksize=BLOCK_SECONDS * fs, dtype="int16", always_2d=True
            )
        ]
    wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
    # written to memory first: the WAV header is completed with a seek, which
    # a pipe does not allow
    buf = io.BytesIO()
    sf.write(buf, wav, fs, format="WAV", subtype="PCM_16")
    sys.stdout.buffer.write(buf.getvalue())


if __name__ == "__main__":
    main()
