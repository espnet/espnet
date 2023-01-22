#!/usr/bin/env python3
import argparse
import logging
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import kaldiio
import numpy as np
import resampy
import soundfile
import soundfile as sf
from scipy.signal import lfilter
from tqdm import tqdm
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text
from espnet.utils.cli_utils import get_commandline_args


# Credits: code from Fairseq
def rvad(speechproc, data, fs):
    # TODO(jiatong): add arguments to params
    winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres = 0.5
    vadThres = 0.4
    opts = 1

    assert fs == 16_000, "sample rate must be 16khz"
    ft, flen, fsh10, nfr10 = speechproc.sflux(data, fs, winlen, ovrlen, nftt)

    # --spectral flatness --
    pv01 = np.zeros(ft.shape[0])
    pv01[np.less_equal(ft, ftThres)] = 1
    pitch = deepcopy(ft)

    pvblk = speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b = np.array([0.9770, -0.9770])
    a = np.array([1.0000, -0.9540])
    fdata = lfilter(b, a, data, axis=0)

    # --pass 1--
    noise_samp, noise_seg, n_noise_samp = speechproc.snre_highenergy(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk
    )

    # sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

    vad_seg = speechproc.snre_vad(
        fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres
    )
    return vad_seg, data


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='compute vad information from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("rvad_home")
    parser.add_argument("scp")
    parser.add_argument("out_scp")
    parser.add_argument("--vad_stride", default=160)
    parser.add_argument("--vad_fs", default=16000)
    args = parser.parse_args()

    sys.path.append(args.rvad_home)
    import speechproc

    # TODO(jiatong): add argument for stride and fs
    assert (
        args.vad_stride == 160
    ), "due to the limit of rvad, we only support 160 stride"
    assert args.vad_fs == 16000, "due to the limit of rvad, we only support 16000 fs"

    Path(args.out_scp).parent.mkdir(parents=True, exist_ok=True)
    out_vadscp = Path(args.out_scp)

    with Path(args.scp).open("r") as fscp, out_vadscp.open("w") as fout:
        for line in tqdm(fscp):
            uttid, wavpath = line.strip().split(None, 1)

            if wavpath.endswith("|"):
                # Streaming input e.g. cat a.wav |
                with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                    with BytesIO(f.read()) as g:
                        wave, rate = soundfile.read(g, dtype=np.int16)
            else:
                wave, rate = sf.read(wavpath)

            if args.vad_fs is not None and args.vad_fs != rate:
                # FIXME(kamo): To use sox?
                wave = resampy.resample(
                    wave.astype(np.float64), rate, args.vad_fs, axis=0
                )
                rate = args.vad_fs

            vads, wav = rvad(speechproc, wave, rate)

            start = None
            vad_segs = []
            for i, v in enumerate(vads):
                if start is None and v == 1:
                    start = i * args.vad_stride
                elif start is not None and v == 0:
                    vad_segs.append((start, i * args.vad_stride))
                    start = None
            if start is not None:
                vad_segs.append((start, len(wav)))

            vads = " ".join(
                "{:.4f}:{:.4f}".format(v[0] / args.vad_fs, v[1] / args.vad_fs)
                for v in vad_segs
            )
            fout.write("{} {}\n".format(uttid, vads))


if __name__ == "__main__":
    main()
