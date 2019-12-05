#!/usr/bin/env python3

# Copyright 2019 Okayama University (Katsuki Inoue)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import librosa
import numpy as np
import os

def add_sils(indir, outdir, dur, hsil=False, tsil=False):
    files = glob.glob(os.path.join(indir, "*.wav"))

    # make outdir
    if files != None:
        os.makedirs(outdir, exist_ok=True)

    for file in files:
        # get file name
        fname = os.path.basename(file)
        # print(file)

        # load wav
        wf, sr = librosa.load(file)

        # connect sil
        head_sil = np.zeros(np.int(sr*dur), dtype=np.float32) if hsil else np.array([], dtype=np.float32)
        tail_sil = np.zeros(np.int(sr*dur), dtype=np.float32) if tsil else np.array([], dtype=np.float32)
        wf_new = np.hstack([head_sil, wf, tail_sil])
        assert wf.shape[0] != wf_new.shape[0], "nothing is changed"

        # write wav with head sil and/or tail sil
        librosa.output.write_wav(os.path.join(outdir, fname), wf_new, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help='input wav dir path')
    parser.add_argument('--outdir', type=str, help='output wav dir path')
    parser.add_argument('--dur', type=np.float32, help='duration of silence')
    parser.add_argument('--hsil', type=np.bool, default=False, help='add head sil')
    parser.add_argument('--tsil', type=np.bool, default=False, help='add tail sil')
    args = parser.parse_args()

    add_sils(args.indir, args.outdir, args.dur, args.hsil, args.tsil)
