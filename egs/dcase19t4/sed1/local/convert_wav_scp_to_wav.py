#!/usr/bin/env python3

import os

import fire

from kaldiio import ReadHelper
from scipy.io import wavfile


def convert_wav_scp_to_wav(rspecifier, dumpdir="dump"):
    """Dump wavfiles in wav.scp
    Args:
        rspecifier (str): kaldi format read specifier (e.g. scp:/path/to/wav.scp)
        dumpdir (str): directory to dump wavfiles
    """
    # check directory existence
    if not os.path.exists(dumpdir):
        os.makedirs(dumpdir)

    # write wav file
    with ReadHelper(rspecifier) as reader:
        for uttid, (rate, x) in reader:
            print("writing %s.wav in %s." % (uttid, dumpdir))
            wavfile.write(dumpdir + "/" + uttid + ".wav", rate, x)


if __name__ == "__main__":
    fire.Fire(convert_wav_scp_to_wav)